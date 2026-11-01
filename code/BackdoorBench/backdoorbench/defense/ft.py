'''
This file implements the defense method called finetuning (ft), which is a standard fine-tuning that uses clean data to finetune the model.

basic sturcture for defense method:
    1. basic setting: args
    2. attack result(model, train data, test data)
    3. ft defense:
        a. get some clean data
        b. retrain the backdoor model
    4. test the result and get ASR, ACC, RC 
'''

import argparse
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

from backdoorbench.utils.aggregate_block.train_settings_generate import argparser_criterion, argparser_opt_scheduler
from backdoorbench.utils.trainer_cls import PureCleanModelTrainer
from backdoorbench.utils.choose_index import choose_index
from backdoorbench.utils.aggregate_block.fix_random import fix_random
from backdoorbench.utils.aggregate_block.model_trainer_generate import generate_cls_model
from backdoorbench.utils.aggregate_block.dataset_and_transform_generate import get_input_shape, get_num_classes, get_transform
from backdoorbench.utils.save_load_attack import load_attack_result, save_defense_result
from backdoorbench.utils.bd_dataset_v2 import prepro_cls_DatasetBD_v2

class ft(defense):
    def __init__(self,parser=argparse.ArgumentParser()):
        super(ft, self).__init__(parser)

    def add_arguments(self,parser):
        parser = super().add_arguments(parser)
        #set the parameter for the ft defense
        parser.add_argument('--ratio', type=float, help='the ratio of clean data loader')
        parser.add_argument('--index', type=str, help='index of clean data')


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
       

        optimizer, scheduler = argparser_opt_scheduler(model, self.args)
        # criterion = nn.CrossEntropyLoss()
        self.set_trainer(model)
        criterion = argparser_criterion(self.args)

        

        train_tran = get_transform(self.args.dataset, *([self.args.input_height,self.args.input_width]) , train = True)
        clean_dataset = prepro_cls_DatasetBD_v2(self.result['clean_train'].wrapped_dataset)
        data_all_length = len(clean_dataset)
        ran_idx = choose_index(self.args, data_all_length) 
        log_index = os.path.join(self.args.log , 'index.txt')
        np.savetxt(log_index, ran_idx, fmt='%d')
        clean_dataset.subset(ran_idx)
        data_set_without_tran = clean_dataset
        data_set_o = self.result['clean_train']
        data_set_o.wrapped_dataset = data_set_without_tran
        data_set_o.wrap_img_transform = train_tran
      
        data_loader = torch.utils.data.DataLoader(data_set_o, batch_size=self.args.batch_size, num_workers=self.args.num_workers, shuffle=True, pin_memory=self.args.pin_memory)
        trainloader = data_loader
        
        test_tran = get_transform(self.args.dataset, *([self.args.input_height,self.args.input_width]) , train = False)
        data_bd_testset = self.result['bd_test']
        data_bd_testset.wrap_img_transform = test_tran
        data_bd_loader = torch.utils.data.DataLoader(data_bd_testset, batch_size=self.args.batch_size, num_workers=self.args.num_workers,drop_last=False, shuffle=True,pin_memory=self.args.pin_memory)

        data_clean_testset = self.result['clean_test']
        data_clean_testset.wrap_img_transform = test_tran
        data_clean_loader = torch.utils.data.DataLoader(data_clean_testset, batch_size=self.args.batch_size, num_workers=self.args.num_workers,drop_last=False, shuffle=True,pin_memory=self.args.pin_memory)

        self.trainer.train_with_test_each_epoch_on_mix(
            trainloader,
            data_clean_loader,
            data_bd_loader,
            self.args.epochs,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=self.args.device,
            frequency_save=self.args.frequency_save,
            save_folder_path=self.args.save_path,
            save_prefix='ft',
            amp=self.args.amp,
            prefetch=self.args.prefetch,
            prefetch_transform_attr_name="ori_image_transform_in_loading", # since we use the preprocess_bd_dataset
            non_blocking=self.args.non_blocking,
        )
        
        result = {}
        result['model'] = model
        save_defense_result(
            model_name=self.args.model,
            num_classes=self.args.num_classes,
            model=model.cpu().state_dict(),
            save_path=self.args.save_path,
        )
        return result

    def defense(self,result_file):

        self.set_result(result_file)
        self.set_logger()
        result = self.mitigation()
        return result
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=sys.argv[0])
    ft_method = ft(parser)
    result = ft_method.defense(ft_method.args.result_file)