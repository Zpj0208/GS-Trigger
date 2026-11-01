import argparse
import os,sys
import numpy as np
import torch
import torch.nn as nn
import shutil
sys.path.append('../')
sys.path.append(os.getcwd())

from pprint import  pformat
import yaml
import logging
import time
from copy import deepcopy
import torch.nn.utils.prune as prune

from backdoorbench.defense.base import defense
from backdoorbench.utils.aggregate_block.train_settings_generate import argparser_opt_scheduler
from backdoorbench.utils.trainer_cls import ModelTrainerCLS_v2, BackdoorModelTrainer, Metric_Aggregator, given_dataloader_test, general_plot_for_epoch
from backdoorbench.utils.choose_index import choose_index
from backdoorbench.utils.aggregate_block.fix_random import fix_random
from backdoorbench.utils.aggregate_block.model_trainer_generate import generate_cls_model
from backdoorbench.utils.log_assist import get_git_info
from backdoorbench.utils.aggregate_block.dataset_and_transform_generate import get_input_shape, get_num_classes, get_transform
from backdoorbench.utils.save_load_attack import load_attack_result, save_defense_result
from backdoorbench.utils.bd_dataset_v2 import prepro_cls_DatasetBD_v2, dataset_wrapper_with_transform

class fp(defense):

    def __init__(self,parser=argparse.ArgumentParser()):
        super(fp, self).__init__(parser)

    def add_arguments(self, parser):
        parser = super().add_arguments(parser)
        parser.add_argument('--index', type=str, help='index of clean data')
        parser.add_argument('--ratio', type=float, help='the ratio of clean data loader')
        parser.add_argument("--once_prune_ratio", type = float, help ="how many percent once prune. in 0 to 1")
        # return parser

    def mitigation(self):

        self.attack_result = self.result

        fix_random(self.args.random_seed)

        args = self.args

        netC = generate_cls_model(args.model, args.num_classes)
        netC.load_state_dict(self.attack_result['model'])
        netC.to(args.device)
        netC.eval()
        netC.requires_grad_(False)

        self.netC = netC

        attack_result = self.attack_result
        # clean_train with subset
        clean_train_dataset_with_transform = attack_result['clean_train']
        clean_train_dataset_without_transform = clean_train_dataset_with_transform.wrapped_dataset
        clean_train_dataset_without_transform = prepro_cls_DatasetBD_v2(
            clean_train_dataset_without_transform
        )
        ran_idx = choose_index(args, len(clean_train_dataset_without_transform))
        logging.info(f"get ran_idx for subset clean train dataset, (len={len(ran_idx)}), ran_idx:{ran_idx}")
        clean_train_dataset_without_transform.subset(
            choose_index(args, len(clean_train_dataset_without_transform))
        )
        clean_train_dataset_with_transform.wrapped_dataset = clean_train_dataset_without_transform
        log_index = os.path.join(self.args.log, 'index.txt')
        np.savetxt(log_index, ran_idx, fmt='%d')
        trainloader = torch.utils.data.DataLoader(clean_train_dataset_with_transform, batch_size=args.batch_size, num_workers=args.num_workers,
                                                  shuffle=True)

        clean_test_dataset_with_transform = attack_result['clean_test']
        data_clean_testset = clean_test_dataset_with_transform
        clean_test_dataloader = torch.utils.data.DataLoader(data_clean_testset, batch_size=args.batch_size,
                                                        num_workers=args.num_workers, drop_last=False, shuffle=True,
                                                        pin_memory=args.pin_memory)

        # bd_train_dataset_with_transform = attack_result['bd_train']

        bd_test_dataset_with_transform = attack_result['bd_test']
        data_bd_testset = bd_test_dataset_with_transform
        bd_test_dataset_without_transform = bd_test_dataset_with_transform.wrapped_dataset
        bd_test_dataloader = torch.utils.data.DataLoader(data_bd_testset, batch_size=args.batch_size,
                                                     num_workers=args.num_workers, drop_last=False, shuffle=True,
                                                     pin_memory=args.pin_memory)


        criterion = nn.CrossEntropyLoss()

        # Forward hook for getting last module's input
        global result_mid
        # container = []
        result_mid = torch.tensor(0).to(args.device)
        with torch.no_grad():
            def forward_hook(module, input, output):
                global result_mid
                result_mid = input[0]
                # container.append(input.detach().clone().cpu())
        last_child_name, last_child = list(netC.named_children())[-1]
        logging.info(f"hook on {last_child_name}")
        hook = last_child.register_forward_hook(forward_hook)

        logging.info("Forwarding all the training dataset:")
        with torch.no_grad():
            flag = 0
            for batch_idx, (inputs, *other) in enumerate(trainloader):
                inputs = inputs.to(args.device)
                _ = netC(inputs)
                if flag == 0:
                    activation = torch.zeros(result_mid.size()[1]).to(args.device)
                    flag = 1
                activation += torch.sum(result_mid, dim=[0]) / len(clean_train_dataset_without_transform)
        hook.remove()

        seq_sort = torch.argsort(activation)
        logging.info(f"get seq_sort, (len={len(seq_sort)}), seq_sort:{seq_sort}")
        # del container

        # find the first linear child in last_child.
        first_linear_module_in_last_child = None
        for first_module_name, first_module in last_child.named_modules():
            if isinstance(first_module, nn.Linear):
                logging.info(f"Find the first child be nn.Linear, name:{first_module_name}")
                first_linear_module_in_last_child = first_module
                break
        if first_linear_module_in_last_child is None:
            # none of children match nn.Linear
            raise Exception("None of children in last module is nn.Linear, cannot prune.")

        # init prune_mask, prune_mask is "accumulated"!
        prune_mask = torch.ones_like(first_linear_module_in_last_child.weight)

        prune_info_recorder = Metric_Aggregator()
        test_acc_list = []
        test_asr_list = []
        test_ra_list = []
        # start from 0, so unprune case will also be tested.
        # for num_pruned in range(0, len(seq_sort), 500):
        for num_pruned in range(0, len(seq_sort), int(len(seq_sort) * args.once_prune_ratio)):
            net_pruned = (netC)
            net_pruned.to(args.device)
            if num_pruned:
                # add_pruned_channnel_index = seq_sort[num_pruned - 1] # each time prune_mask ADD ONE MORE channel being prune.
                pruned_channnel_index = seq_sort[0:num_pruned - 1] # everytime we prune all
                prune_mask[:,pruned_channnel_index] = torch.zeros_like(prune_mask[:,pruned_channnel_index])
                prune.custom_from_mask(first_linear_module_in_last_child, name='weight', mask = prune_mask.to(args.device))

                # prune_ratio = 100. * float(torch.sum(first_linear_module_in_last_child.weight_mask == 0)) / float(first_linear_module_in_last_child.weight_mask.nelement())
                # logging.info(f"Pruned {num_pruned}/{len(seq_sort)}  ({float(prune_ratio):.2f}%) filters")

            # test
            test_acc = given_dataloader_test(net_pruned, clean_test_dataloader, criterion, args.non_blocking, args.device)[0]['test_acc']
            test_asr = given_dataloader_test(net_pruned, bd_test_dataloader, criterion, args.non_blocking, args.device)[0]['test_acc']

            # use switch in preprocess bd dataset v2
            bd_test_dataset_without_transform.getitem_all_switch = True
            test_ra = given_dataloader_test(net_pruned, bd_test_dataloader, criterion, args.non_blocking, args.device)[0]['test_acc']
            bd_test_dataset_without_transform.getitem_all_switch = False

            prune_info_recorder({
                "num_pruned":num_pruned,
                "all_filter_num":len(seq_sort),
                "test_acc" : test_acc,
                "test_asr" : test_asr,
                "test_ra" : test_ra,
            })

            test_acc_list.append(float(test_acc))
            test_asr_list.append(float(test_asr))
            test_ra_list.append(float(test_ra))

            if num_pruned == 0:
                test_acc_cl_ori = test_acc
                last_net = (net_pruned)
                last_index = 0
            if abs(test_acc - test_acc_cl_ori) / test_acc_cl_ori < args.acc_ratio:
                last_net = (net_pruned)
                last_index = num_pruned
            else:
                break

        prune_info_recorder.to_dataframe().to_csv(os.path.join(self.args.save_path, "prune_log.csv"))
        prune_info_recorder.summary().to_csv(os.path.join(self.args.save_path, "prune_log_summary.csv"))
        general_plot_for_epoch(
            {
                "test_acc":test_acc_list,
                "test_asr":test_asr_list,
                "test_ra":test_ra_list,
            },
            os.path.join(self.args.save_path, "prune_log_plot.jpg"),
            ylabel='percentage',
            xlabel="num_pruned",
        )

        logging.info(f"End prune. Pruned {num_pruned}/{len(seq_sort)} test_acc:{test_acc:.2f}  test_asr:{test_asr:.2f}  test_ra:{test_ra:.2f}  ")


        # finetune
        last_net.train()
        last_net.requires_grad_()

        optimizer, scheduler = argparser_opt_scheduler(
            last_net,
            self.args,
        )

        self.set_trainer(last_net)
        finetune_trainer = self.trainer

        finetune_trainer.train_with_test_each_epoch_on_mix(
            trainloader,
            clean_test_dataloader,
            bd_test_dataloader,
            args.epochs,
            criterion,
            optimizer,
            scheduler,
            args.amp,
            torch.device(args.device),
            args.frequency_save,
            self.args.save_path,
            "finetune",
            prefetch=False,
            prefetch_transform_attr_name="transform",
            non_blocking=args.non_blocking,
        )

        save_defense_result(
            model_name = args.model,
            num_classes = args.num_classes,
            model = last_net.cpu().state_dict(),
            save_path = self.args.save_path,
        )

        result = {}
        result['model'] = last_net

        return result

    def set_trainer(self, model):
        self.trainer = BackdoorModelTrainer(
            model,
        )

    def defense(self,result_file):
        self.set_result(result_file)
        self.set_logger()
        result = self.mitigation()
        return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=sys.argv[0])
    fp_method = fp(parser)
    result = fp_method.defense(fp_method.args.result_file)
