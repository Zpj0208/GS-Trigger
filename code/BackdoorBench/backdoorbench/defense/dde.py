'''
This file is modified based on the following source:
link : https://github.com/RJ-T/NIPS2022_EP_BNP.
The defense method is called dde or ep.

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
import backdoorbench.utils.defense_utils.dde.dde_model as dde_model
from backdoorbench.utils.aggregate_block.train_settings_generate import argparser_criterion
from backdoorbench.utils.trainer_cls import Metric_Aggregator, PureCleanModelTrainer, general_plot_for_epoch
from backdoorbench.utils.aggregate_block.fix_random import fix_random
from backdoorbench.utils.aggregate_block.model_trainer_generate import generate_cls_model
from backdoorbench.utils.log_assist import get_git_info
from backdoorbench.utils.aggregate_block.dataset_and_transform_generate import get_input_shape, get_num_classes, get_transform
from backdoorbench.utils.save_load_attack import load_attack_result, save_defense_result

def batch_entropy(x, step_size=0.1):
	n_bars = int((x.max()-x.min())/step_size)
	entropy = 0
	for n in range(n_bars):
		num = ((x > x.min() + n*step_size) * (x < x.min() + (n+1)*step_size)).sum(-1)
		p = num / x.shape[-1]
		entropy += - p * p.log().nan_to_num(0)
	return entropy


# This version of dde uses only uses args.batch-size samples in the mixed training dataset for pruning.
def DDE_defense(net, u, mixture_data_loader, args):
	net.eval()
	mixture_data = iter(mixture_data_loader).__next__()[0].to(args.device)
	params = net.state_dict()
	for m in net.modules():
		if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
			m.collect_feats = True

	with torch.no_grad():
		net(mixture_data)
		for name, m in net.named_modules():
			if isinstance(m, nn.BatchNorm2d):
				feats = m.batch_feats
				feats = (feats - feats.mean(-1).unsqueeze(-1)) / feats.std(-1).unsqueeze(-1)

				entropy = batch_entropy(feats)
				index = (entropy<(entropy.mean() - u*entropy.std()))

				params[name+'.weight'][index] = 0
				params[name+'.bias'][index] = 0
			# We use layer norm to subsitute batch norm in convnext_model and vit_model
			elif isinstance(m, nn.LayerNorm):
				feats = m.batch_feats
				feats = (feats - feats.mean(-1).unsqueeze(-1)) / feats.std(-1).unsqueeze(-1)
				# variance is zero
				feats = torch.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=-0.0)
				entropy = batch_entropy(feats)
				index = (entropy<(entropy.mean() - u*entropy.std()))

				params[name+'.weight'][index] = 0
				params[name+'.bias'][index] = 0

	net.load_state_dict(params)


def get_dde_network(
	model_name: str,
	num_classes: int = 10,
	**kwargs,
):
	if model_name == 'preactresnet18':
		net = dde_model.preact_dde.PreActResNet18(num_classes = num_classes, **kwargs)
	elif model_name == 'vgg19_bn':
		net = dde_model.vgg_dde.vgg19_bn(num_classes = num_classes,  **kwargs)
	elif model_name == 'densenet161':
		net = dde_model.den_dde.densenet161(num_classes= num_classes, **kwargs)
	elif model_name == 'mobilenet_v3_large':
		net = dde_model.mobilenet_dde.mobilenet_v3_large(num_classes= num_classes, **kwargs)
	elif model_name == 'efficientnet_b3':
		net = dde_model.eff_dde.efficientnet_b3(num_classes= num_classes, **kwargs)
	elif model_name == 'convnext_tiny':
		try :
			net = dde_model.conv_dde.convnext_tiny(num_classes= num_classes, 
			)
		except:
			net = dde_model.conv_new_dde.convnext_tiny(num_classes= num_classes,
			)
	elif model_name == 'vit_b_16':
		try :
			from torchvision.transforms import Resize
			net = dde_model.vit_dde.vit_b_16(
					pretrained = True,
				)
			net.heads.head = torch.nn.Linear(net.heads.head.in_features, out_features = num_classes, bias=True)
			net = torch.nn.Sequential(
					Resize((224, 224)),
					net,
				)
		except :
			from torchvision.transforms import Resize
			net = dde_model.vit_new_dde.vit_b_16(
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


class dde(defense):

	def __init__(self,parser=argparse.ArgumentParser()):
		super(dde, self).__init__(parser)

	def add_arguments(self,parser):
		parser = super().add_arguments(parser)
		#set the parameter for the dde defense
		parser.add_argument('--u', type=float, help='u in the DDE defense')
		parser.add_argument('--u_min', type=float, help='the default minimum value of u')
		parser.add_argument('--u_max', type=float, help='the default maximum value of u')
		parser.add_argument('--u_num', type=int, help='the default number of u')

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
		self.device = self.args.device
	def mitigation(self):
		self.set_devices()
		fix_random(self.args.random_seed)

		# Prepare model, optimizer, scheduler
		
		net = get_dde_network(self.args.model,self.args.num_classes,norm_layer=dde_model.BatchNorm2d_DDE)
		# net = generate_cls_model(self.args.model,self.args.num_classes)
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
		
		criterion = argparser_criterion(self.args)

		train_tran = get_transform(self.args.dataset, *([self.args.input_height,self.args.input_width]) , train = True)
		train_dataset = self.result['bd_train'].wrapped_dataset
		data_set_without_tran = train_dataset
		data_set_o = self.result['bd_train']
		data_set_o.wrapped_dataset = data_set_without_tran
		data_set_o.wrap_img_transform = train_tran
		data_loader = torch.utils.data.DataLoader(data_set_o, batch_size=self.args.batch_size, num_workers=self.args.num_workers, shuffle=True, pin_memory=self.args.pin_memory)
		trainloader = data_loader


		test_tran = get_transform(self.args.dataset, *([self.args.input_height,self.args.input_width]) , train = False)
		data_bd_testset = self.result['bd_test']
		data_bd_testset.wrap_img_transform = test_tran
		data_bd_loader = torch.utils.data.DataLoader(data_bd_testset, batch_size=self.args.batch_size, num_workers=self.args.num_workers,drop_last=False,pin_memory=self.args.pin_memory)

		data_clean_testset = self.result['clean_test']
		data_clean_testset.wrap_img_transform = test_tran
		data_clean_loader = torch.utils.data.DataLoader(data_clean_testset, batch_size=self.args.batch_size, num_workers=self.args.num_workers,drop_last=False,pin_memory=self.args.pin_memory)


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
			DDE_defense(model_copy, u, trainloader, self.args)
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
				save_prefix = 'dde',

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
		DDE_defense(net, self.args.u, trainloader, self.args)

		test_dataloader_dict = {}
		test_dataloader_dict["clean_test_dataloader"] = data_clean_loader
		test_dataloader_dict["bd_test_dataloader"] = data_bd_loader
		
		model = generate_cls_model(self.args.model,self.args.num_classes)
		model.load_state_dict(net.state_dict())
		self.set_trainer(model)

		self.trainer.set_with_dataloader(
			train_dataloader = trainloader,
			test_dataloader_dict = test_dataloader_dict,

			criterion = criterion,
			optimizer = None,
			scheduler = None,
			device = self.args.device,
			amp = self.args.amp,

			frequency_save = self.args.frequency_save,
			save_folder_path = self.args.save_path,
			save_prefix = 'dde',

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
		agg.to_dataframe().to_csv(os.path.join(f"{self.args.save_path}","dde_df_summary.csv"))

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
	method = dde()
	result = method.defense(method.args.result_file)