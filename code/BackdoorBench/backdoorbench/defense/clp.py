'''
This file is modified based on the following source:
link : https://github.com/rkteddy/channel-Lipschitzness-based-pruning.
The defense method is called clp.

The update include:
    1. data preprocess and dataset setting
    2. model setting
    3. args and config
    4. save process
    5. new standard: robust accuracy
    6. draw the corresponding images of asr and acc according to different proportions
basic sturcture for defense method:
    1. basic setting: args
    2. attack result(model, train data, test data)
    3. clp defense:
        a. prune the model depend on the estimate of TAC
    4. test the result and get ASR, ACC, RC with regard to the chosen threshold and interval
'''


import argparse
import copy
import os,sys
import numpy as np
import torch
import torch.nn as nn


sys.path.append('../')
sys.path.append(os.getcwd())

from pprint import  pformat
import yaml
import logging
import time
from backdoorbench.defense.base import defense
from backdoorbench.utils.aggregate_block.train_settings_generate import argparser_criterion
from backdoorbench.utils.trainer_cls import Metric_Aggregator, PureCleanModelTrainer, general_plot_for_epoch
from backdoorbench.utils.aggregate_block.fix_random import fix_random
from backdoorbench.utils.aggregate_block.model_trainer_generate import generate_cls_model
from backdoorbench.utils.log_assist import get_git_info
from backdoorbench.utils.aggregate_block.dataset_and_transform_generate import get_input_shape, get_num_classes, get_transform
from backdoorbench.utils.save_load_attack import load_attack_result, save_defense_result

def CLP_prune(net, u):
    params = net.state_dict()
    # conv = None
    for name, m in net.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            std = m.running_var.sqrt()
            weight = m.weight

            channel_lips = []
            for idx in range(weight.shape[0]):
                # Combining weights of convolutions and BN
                w = conv.weight[idx].reshape(conv.weight.shape[1], -1) * (weight[idx]/std[idx]).abs()
                channel_lips.append(torch.svd(w.cpu())[1].max())
            channel_lips = torch.Tensor(channel_lips)

            index = torch.where(channel_lips>channel_lips.mean() + u*channel_lips.std())[0]

            params[name+'.weight'][index] = 0
            params[name+'.bias'][index] = 0
            # print(index)
        
       # Convolutional layer should be followed by a BN layer by default
        elif isinstance(m, nn.Conv2d):
            conv = m

    net.load_state_dict(params)

class clp(defense):

    def __init__(self, parser=argparse.ArgumentParser()):
        super(clp, self).__init__(parser)

    def add_arguments(self,parser):
        parser = super().add_arguments(parser)
        #set the parameter for the clp defense
        parser.add_argument('--u', type=float, help='the default value of u')
        parser.add_argument('--u_min', type=float, help='the default minimum value of u')
        parser.add_argument('--u_max', type=float, help='the default maximum value of u')
        parser.add_argument('--u_num', type=int, help='the default number of u')

    def set_trainer(self, model):
        self.trainer = PureCleanModelTrainer(
            model,
        )
    
    def set_devices(self):
        # self.device = torch.device(
        #     (
        #         f"cuda:{[int(i) for i in self.args.device[5:].split(',')][0]}" if "," in self.args.device else self.args.device
        #         # since DataParallel only allow .to("cuda")
        #     ) if torch.cuda.is_available() else "cpu"
        # )
        self.device = self.args.device
        
    def mitigation(self):
        self.set_devices()
        fix_random(self.args.random_seed)
        args = self.args
        # Prepare model, optimizer, scheduler
        model = generate_cls_model(self.args.model,self.args.num_classes)
        model.load_state_dict(self.result['model'])
        if "," in self.device:
            model = torch.nn.DataParallel(
                model,
                device_ids=[int(i) for i in self.args.device[5:].split(",")]  # eg. "cuda:2,3,7" -> [2,3,7]
            )
            self.args.device = f'cuda:{model.device_ids[0]}'
            model.to(self.args.device)
        else:
            model.to(self.args.device)
        # criterion = nn.CrossEntropyLoss()
        
        criterion = argparser_criterion(self.args)

        test_tran = get_transform(self.args.dataset, *([self.args.input_height,self.args.input_width]) , train = False)
        data_bd_testset = self.result['bd_test']
        data_bd_testset.wrap_img_transform = test_tran
        data_bd_loader = torch.utils.data.DataLoader(data_bd_testset, batch_size=self.args.batch_size, num_workers=self.args.num_workers,drop_last=False,pin_memory=args.pin_memory)

        data_clean_testset = self.result['clean_test']
        data_clean_testset.wrap_img_transform = test_tran
        data_clean_loader = torch.utils.data.DataLoader(data_clean_testset, batch_size=self.args.batch_size, num_workers=self.args.num_workers,drop_last=False,pin_memory=args.pin_memory)

        # model.eval()
        default_u = np.linspace(self.args.u_min, self.args.u_max, self.args.u_num)
        
        agg_all = Metric_Aggregator()
        clean_test_loss_list = []
        bd_test_loss_list = []
        test_acc_list = []
        test_asr_list = []
        test_ra_list = []
        for u in default_u:
            model_copy = copy.deepcopy(model)
            model_copy.eval()
            CLP_prune(model_copy, u)
            # model.eval()
            model_copy.eval()
            test_dataloader_dict = {}
            test_dataloader_dict["clean_test_dataloader"] = data_clean_loader
            test_dataloader_dict["bd_test_dataloader"] = data_bd_loader

            self.set_trainer(model_copy)
            self.trainer.set_with_dataloader(
                ### the train_dataload has nothing to do with the backdoor defense
                train_dataloader = data_bd_loader,
                test_dataloader_dict = test_dataloader_dict,

                criterion = criterion,
                optimizer = None,
                scheduler = None,
                device = self.args.device,
                amp = self.args.amp,

                frequency_save = self.args.frequency_save,
                save_folder_path = self.args.save_path,
                save_prefix = 'clp',

                prefetch = self.args.prefetch,
                prefetch_transform_attr_name = "ori_image_transform_in_loading",
                non_blocking = self.args.non_blocking,

    
                )

            clean_test_loss_avg_over_batch, \
                    bd_test_loss_avg_over_batch, \
                    test_acc, \
                    test_asr, \
                    test_ra = self.trainer.test_current_model(
                test_dataloader_dict, self.args.device,
            )
            clean_test_loss_list.append(clean_test_loss_avg_over_batch)
            bd_test_loss_list.append(bd_test_loss_avg_over_batch)
            test_acc_list.append(test_acc)
            test_asr_list.append(test_asr)
            test_ra_list.append(test_ra)
            agg_all({
                "u": u,
                "clean_test_loss_avg_over_batch": clean_test_loss_avg_over_batch,
                "bd_test_loss_avg_over_batch": bd_test_loss_avg_over_batch,
                "test_acc": test_acc,
                "test_asr": test_asr,
                "test_ra": test_ra,
            })

            general_plot_for_epoch(
                {
                    "Test C-Acc": test_acc_list,
                    "Test ASR": test_asr_list,
                    "Test RA": test_ra_list,
                },
                save_path=os.path.join(f"{args.save_path}","u_step_acc_like_metric_plots.png"),
                ylabel="percentage",
            )

            general_plot_for_epoch(
                {
                    "Test Clean Loss": clean_test_loss_list,
                    "Test Backdoor Loss": bd_test_loss_list,
                },
                save_path=os.path.join(f"{args.save_path}","u_step_loss_metric_plots.png"),
                ylabel="percentage",
            )

            general_plot_for_epoch(
                {
                    "u": default_u,
                },
                save_path=os.path.join(f"{args.save_path}","u_step_plots.png"),
                ylabel="percentage",
            )

            agg_all.to_dataframe().to_csv(os.path.join(f"{args.save_path}","u_step_df.csv"))

        agg = Metric_Aggregator()
        CLP_prune(model, self.args.u)

        test_dataloader_dict = {}
        test_dataloader_dict["clean_test_dataloader"] = data_clean_loader
        test_dataloader_dict["bd_test_dataloader"] = data_bd_loader

        self.set_trainer(model)
        self.trainer.set_with_dataloader(
            ### the train_dataload has nothing to do with the backdoor defense
            train_dataloader = data_bd_loader,
            test_dataloader_dict = test_dataloader_dict,

            criterion = criterion,
            optimizer = None,
            scheduler = None,
            device = self.args.device,
            amp = self.args.amp,

            frequency_save = self.args.frequency_save,
            save_folder_path = self.args.save_path,
            save_prefix = 'clp',

            prefetch = self.args.prefetch,
            prefetch_transform_attr_name = "ori_image_transform_in_loading",
            non_blocking = self.args.non_blocking,

            # continue_training_path = continue_training_path,
            # only_load_model = only_load_model,
        )

        clean_test_loss_avg_over_batch, \
                bd_test_loss_avg_over_batch, \
                test_acc, \
                test_asr, \
                test_ra = self.trainer.test_current_model(
            test_dataloader_dict, self.args.device,
        )
        agg({
                "u": self.args.u,
                "clean_test_loss_avg_over_batch": clean_test_loss_avg_over_batch,
                "bd_test_loss_avg_over_batch": bd_test_loss_avg_over_batch,
                "test_acc": test_acc,
                "test_asr": test_asr,
                "test_ra": test_ra,
            })
        agg.to_dataframe().to_csv(os.path.join(f"{args.save_path}","clp_df_summary.csv"))

        logging.info(f'the threshold{args.u} clean_loss:{clean_test_loss_avg_over_batch} bd_loss:{bd_test_loss_avg_over_batch} clean_acc:{test_acc} asr:{test_asr} ra:{test_ra}')


        result = {}
        result['model'] = model
        save_defense_result(
            model_name=args.model,
            num_classes=args.num_classes,
            model=model.cpu().state_dict(),
            save_path=args.save_path,
        )
        return result

    def defense(self,result_file):
        self.set_result(result_file)
        self.set_logger()
        result = self.mitigation()
        return result
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=sys.argv[0])
    clp_method = clp(parser)
    result = clp_method.defense(clp_method.args.result_file)