'''
This file is modified based on the following source:
link : https://github.com/RJ-T/NIPS2022_EP_BNP.
The defense method is called mbns or bnp.

The update include:
    1. data preprocess and dataset setting
    2. model setting
    3. args and config
    4. save process
    5. new standard: robust accuracy
    6. reconstruct the layer norm for convnext and transformer
    7. draw the corresponding images of asr and acc according to different proportions
basic sturcture for defense method:
    1. basic setting: args
    2. attack result(model, train data, test data)
    3. dde defense:
        a. calculate the entropy of each norm layer
        b. prune the model depend on the mask
    4. test the result and get ASR, ACC, RC 
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
import backdoorbench.utils.defense_utils.mbns.mbns_model as mbns_model
from backdoorbench.utils.aggregate_block.train_settings_generate import argparser_criterion
from backdoorbench.utils.trainer_cls import Metric_Aggregator, PureCleanModelTrainer, general_plot_for_epoch
from backdoorbench.utils.choose_index import choose_index
from backdoorbench.utils.aggregate_block.fix_random import fix_random
from backdoorbench.utils.aggregate_block.model_trainer_generate import generate_cls_model
from backdoorbench.utils.log_assist import get_git_info
from backdoorbench.utils.aggregate_block.dataset_and_transform_generate import get_input_shape, get_num_classes, get_transform
from backdoorbench.utils.save_load_attack import load_attack_result, save_defense_result
from backdoorbench.utils.bd_dataset_v2 import prepro_cls_DatasetBD_v2

def batch_entropy(x, step_size=0.1):
	n_bars = int((x.max()-x.min())/step_size)
	entropy = 0
	for n in range(n_bars):
		num = ((x > x.min() + n*step_size) * (x < x.min() + (n+1)*step_size)).sum(-1)
		p = num / x.shape[-1]
		entropy += - p * p.log().nan_to_num(0)
	return entropy

def mbns_defense(net, u, trainloader, args):
	clean_data_loader = trainloader['clean_train']
	bd_data_loader = trainloader['bd_train']
	net.eval()
	bd_data = next(iter(bd_data_loader))[0].to(args.device)
	mixture_data = next(iter(clean_data_loader))[0].to(args.device)
	params = net.state_dict()
	for m in net.modules():
		if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
			m.collect_stats = True
		if isinstance(m, nn.LayerNorm):
			m.collect_stats_clean = True
			m.collect_stats_bd = False
	with torch.no_grad():
		net(mixture_data)
	for m in net.modules():
		if isinstance(m, nn.LayerNorm):
			m.collect_stats_bd = True
			m.collect_stats_clean = False
	with torch.no_grad():
		net(bd_data)
		for name, m in net.named_modules():
			if isinstance(m, nn.BatchNorm2d):
				var_2 = m.running_var
				var_1 = m.batch_var
				mean_2 = m.running_mean
				mean_1 = m.batch_mean
				kl_div = (var_2/var_1).log() + (var_1+(mean_1-mean_2).pow(2))/(2*var_2) - 1/2
				index = (kl_div>kl_div.mean() + u*kl_div.std())

				params[name+'.weight'][index] = 0
				params[name+'.bias'][index] = 0
			elif isinstance(m, nn.LayerNorm):
				# We use layer norm to subsitute batch norm in convnext_model and vit_model
				var_2 = m.batch_var_bd
				var_1 = m.batch_var_clean
				mean_2 = m.batch_mean_bd
				mean_1 = m.batch_mean_clean
				kl_div = (var_2/var_1).log() + (var_1+(mean_1-mean_2).pow(2))/(2*var_2) - 1/2
				index = (kl_div>kl_div.mean() + u*kl_div.std())

				params[name+'.weight'][index] = 0
				params[name+'.bias'][index] = 0

	net.load_state_dict(params)

def get_mbns_network(
	model_name: str,
	num_classes: int = 10,
	**kwargs,
):
	if model_name == 'preactresnet18':
		net = mbns_model.preact_mbns.PreActResNet18(num_classes = num_classes, **kwargs)
	elif model_name == 'vgg19_bn':
		net = mbns_model.vgg_mbns.vgg19_bn(num_classes = num_classes,  **kwargs)
	elif model_name == 'densenet161':
		net = mbns_model.den_mbns.densenet161(num_classes= num_classes, **kwargs)
	elif model_name == 'mobilenet_v3_large':
		net = mbns_model.mobilenet_mbns.mobilenet_v3_large(num_classes= num_classes, **kwargs)
	elif model_name == 'efficientnet_b3':
		net = mbns_model.eff_mbns.efficientnet_b3(num_classes= num_classes, **kwargs)
	elif model_name == 'convnext_tiny':
		try :
			net = mbns_model.conv_mbns.convnext_tiny(num_classes= num_classes, 
			)
		except :
			net = mbns_model.conv_new_mbns.convnext_tiny(num_classes= num_classes, 
			)
	elif model_name == 'vit_b_16':
		try :
			from torchvision.transforms import Resize
			net = mbns_model.vit_mbns.vit_b_16(
					pretrained = True,
				)
			net.heads.head = torch.nn.Linear(net.heads.head.in_features, out_features = num_classes, bias=True)
			net = torch.nn.Sequential(
					Resize((224, 224)),
					net,
				)
		except :
			from torchvision.transforms import Resize
			net = mbns_model.vit_new_mbns.vit_b_16(
					pretrained = True,
				)
			net.heads.head = torch.nn.Linear(net.heads.head.in_features, out_features = num_classes, bias=True)
			net = torch.nn.Sequential(
					Resize((224, 224)),
					net,
				)
	else:
		raise SystemError('NO valid model match in function generate_cls_model!')

	return net


class mbns(defense):

	def __init__(self,parser=argparse.ArgumentParser()):
		super(mbns, self).__init__(parser)

	def add_arguments(self,parser):
		parser = super().add_arguments(parser)
		#set the parameter for the mbns defense
		parser.add_argument('--u', type=float, help='u in the mbns defense')
		parser.add_argument('--u_min', type=float, help='the default minimum value of u')
		parser.add_argument('--u_max', type=float, help='the default maximum value of u')
		parser.add_argument('--u_num', type=int, help='the default number of u')
		parser.add_argument('--ratio', type=float, help='the ratio of clean data loader')
		parser.add_argument('--index', type=str, help='index of clean data')

	def set_trainer(self, model):
		self.trainer = PureCleanModelTrainer(
			model,
		)
	
	def set_devices(self):
		# self.device = torch.device(
		# 	(
		# 		f"cuda:{[int(i) for i in self.args.device[5:].split(',')][0]}" if "," in self.args.device else self.args.device
		# 		# since DataParallel only allow .to("cuda")
		# 	) if torch.cuda.is_available() else "cpu"
		# )
		self.device =self.args.device
	def mitigation(self):
		args = self.args
		self.set_devices()
		fix_random(self.args.random_seed)

		# Prepare model, optimizer, scheduler
		
		net = get_mbns_network(self.args.model,self.args.num_classes,norm_layer=mbns_model.BatchNorm2d_MBNS)
		net.load_state_dict(self.result['model'])
		if "," in self.device:
			net = torch.nn.DataParallel(
				net,
				device_ids=[int(i) for i in self.args.device[5:].split(",")]  # eg. "cuda:2,3,7" -> [2,3,7]
			)
			self.args.device = f'cuda:{net.device_ids[0]}'
			net.to(self.args.device)
		else:
			net.to(self.args.device)
		# criterion = nn.CrossEntropyLoss()
		
		criterion = argparser_criterion(args)

		trainloader_all = {}

		train_tran = get_transform(self.args.dataset, *([self.args.input_height,self.args.input_width]) , train = True)
		train_dataset = self.result['bd_train'].wrapped_dataset
		data_set_without_tran = train_dataset
		data_set_o = self.result['bd_train']
		data_set_o.wrapped_dataset = data_set_without_tran
		data_set_o.wrap_img_transform = train_tran
		data_loader = torch.utils.data.DataLoader(data_set_o, batch_size=self.args.batch_size, num_workers=self.args.num_workers, shuffle=True, pin_memory=args.pin_memory)
		trainloader_backdoor = data_loader
		trainloader_all['bd_train'] = trainloader_backdoor

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

		data_loader = torch.utils.data.DataLoader(data_set_o, batch_size=self.args.batch_size, num_workers=self.args.num_workers, shuffle=True, pin_memory=args.pin_memory)
		trainloader_clean = data_loader
		trainloader_all['clean_train'] = trainloader_clean


		test_tran = get_transform(self.args.dataset, *([self.args.input_height,self.args.input_width]) , train = False)
		data_bd_testset = self.result['bd_test']
		data_bd_testset.wrap_img_transform = test_tran
		data_bd_loader = torch.utils.data.DataLoader(data_bd_testset, batch_size=self.args.batch_size, num_workers=self.args.num_workers,drop_last=False,pin_memory=args.pin_memory)

		data_clean_testset = self.result['clean_test']
		data_clean_testset.wrap_img_transform = test_tran
		data_clean_loader = torch.utils.data.DataLoader(data_clean_testset, batch_size=self.args.batch_size, num_workers=self.args.num_workers,drop_last=False,pin_memory=args.pin_memory)

		
		default_u = np.linspace(self.args.u_min, self.args.u_max, self.args.u_num)
		
		agg_all = Metric_Aggregator()
		clean_test_loss_list = []
		bd_test_loss_list = []
		test_acc_list = []
		test_asr_list = []
		test_ra_list = []
		for u in default_u:
			model_copy = copy.deepcopy(net)
			model_copy.eval()
			mbns_defense(model_copy, self.args.u, trainloader_all, args)
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
				save_prefix = 'mbns',

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
				save_path=os.path.join(f"{self.args.save_path}","u_step_acc_like_metric_plots.png"),
				ylabel="percentage",
			)

			general_plot_for_epoch(
				{
					"Test Clean Loss": clean_test_loss_list,
					"Test Backdoor Loss": bd_test_loss_list,
				},
				save_path=os.path.join(f"{self.args.save_path}","u_step_loss_metric_plots.png"),
				ylabel="percentage",
			)

			general_plot_for_epoch(
				{
					"u": default_u,
				},
				save_path=os.path.join(f"{self.args.save_path}","u_step_plots.png"),
				ylabel="percentage",
			)

			agg_all.to_dataframe().to_csv(os.path.join(f"{self.args.save_path}","u_step_df.csv"))

		agg = Metric_Aggregator()
		mbns_defense(net, self.args.u, trainloader_all, args)

		test_dataloader_dict = {}
		test_dataloader_dict["clean_test_dataloader"] = data_clean_loader
		test_dataloader_dict["bd_test_dataloader"] = data_bd_loader
		
		model = generate_cls_model(self.args.model,self.args.num_classes)
		model.load_state_dict(net.state_dict())
		self.set_trainer(model)

		self.trainer.set_with_dataloader(
			train_dataloader = trainloader_backdoor,
			test_dataloader_dict = test_dataloader_dict,

			criterion = criterion,
			optimizer = None,
			scheduler = None,
			device = self.args.device,
			amp = self.args.amp,

			frequency_save = self.args.frequency_save,
			save_folder_path = self.args.save_path,
			save_prefix = 'mbns',

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
		agg({
				"u": self.args.u,
				"clean_test_loss_avg_over_batch": clean_test_loss_avg_over_batch,
				"bd_test_loss_avg_over_batch": bd_test_loss_avg_over_batch,
				"test_acc": test_acc,
				"test_asr": test_asr,
				"test_ra": test_ra,
			})
		agg.to_dataframe().to_csv(os.path.join(f"{self.args.save_path}","mbns_df_summary.csv"))

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
	method = mbns()
	result = method.defense(method.args.result_file)