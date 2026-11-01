# MIT License

# Copyright (c) 2017 Brandon Tran and Jerry Li

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

'''
This file is modified based on the following source:
link : https://github.com/MadryLab/backdoor_data_poisoning.
The defense method is called spectral.

The update include:
	1. data preprocess and dataset setting
	2. model setting
	3. args and config
	4. save process
	5. new standard: robust accuracy
	6. use the PyTorch environment instead of TensorFlow
	7. add some addtional backbone such as resnet18 and vgg19
	8. the poison ratio can also be preset when the data for each category is small
basic sturcture for defense method:
	1. basic setting: args
	2. attack result(model, train data, test data)
	3. spectral defense:
		a. prepare the model and dataset
		b. get the activation as representation for each data
		c. detect the backdoor data by the SVD decomposition
		d. retrain the model with remaining data
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
from backdoorbench.utils.aggregate_block.fix_random import fix_random
from backdoorbench.utils.aggregate_block.model_trainer_generate import generate_cls_model
from backdoorbench.utils.log_assist import get_git_info
from backdoorbench.utils.aggregate_block.dataset_and_transform_generate import get_input_shape, get_num_classes, get_transform
from backdoorbench.utils.save_load_attack import load_attack_result, save_defense_result

class spectral(defense):

	def __init__(self,parser=argparse.ArgumentParser()):
		super(spectral, self).__init__(parser)

	def add_arguments(self,parser):
		parser = super().add_arguments(parser)
		#set the parameter for the spectral defense
		parser.add_argument('--percentile', type=float)
		parser.add_argument('--target_label', type=int)
	

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
		args = self.args
		self.set_devices()
		fix_random(self.args.random_seed)

		### a. prepare the model and dataset
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
		# Setting up the data and the model
		train_tran = get_transform(self.args.dataset, *([self.args.input_height,self.args.input_width]) , train = True)
		train_dataset = self.result['bd_train'].wrapped_dataset
		data_set_without_tran = train_dataset
		data_set_o = self.result['bd_train']
		data_set_o.wrapped_dataset = data_set_without_tran
		data_set_o.wrap_img_transform = train_tran
		data_set_o.wrapped_dataset.getitem_all = False
		dataset = data_set_o
	
		# initialize data augmentation
		logging.info(f'Dataset Size: {len(dataset)}' )

		if 'target_label' in self.args.__dict__:
			if isinstance(self.args.target_label,(int)):
				poison_labels = [self.args.target_label]
			else:
				poison_labels = self.args.target_label
		else:
			poison_labels = range(self.args.num_classes)

		re_all = []
		for target_label in poison_labels:
			lbl = target_label
			dataset_y = []
			for i in range(len(dataset)):
				dataset_y.append(dataset[i][1])
			cur_indices = [i for i,v in enumerate(dataset_y) if v==lbl]
			cur_examples = len(cur_indices)
			logging.info(f'Label, num ex: {lbl},{cur_examples}' )
			
			model.eval()
			### b. get the activation as representation for each data
			for iex in range(cur_examples):
				cur_im = cur_indices[iex]
				x_batch = dataset[cur_im][0].unsqueeze(0).to(self.args.device)
				y_batch = dataset[cur_im][1]
				with torch.no_grad():
					assert self.args.model in ['preactresnet18', 'vgg19','vgg19_bn', 'resnet18', 'mobilenet_v3_large', 'densenet161', 'efficientnet_b3','convnext_tiny','vit_b_16']
					if self.args.model == 'preactresnet18':
						inps,outs = [],[]
						def layer_hook(module, inp, out):
							outs.append(out.data)
						hook = model.layer4.register_forward_hook(layer_hook)
						_ = model(x_batch)
						batch_grads = outs[0].view(outs[0].size(0), -1).squeeze(0)
						hook.remove()
					elif self.args.model == 'vgg19':
						inps,outs = [],[]
						def layer_hook(module, inp, out):
							outs.append(out.data)
						hook = model.features.register_forward_hook(layer_hook)
						_ = model(x_batch)
						batch_grads = outs[0].view(outs[0].size(0), -1).squeeze(0)
						hook.remove()
					elif self.args.model == 'vgg19_bn':
						inps,outs = [],[]
						def layer_hook(module, inp, out):
							outs.append(out.data)
						hook = model.features.register_forward_hook(layer_hook)
						_ = model(x_batch)
						batch_grads = outs[0].view(outs[0].size(0), -1).squeeze(0)
						hook.remove()
					elif self.args.model == 'resnet18':
						inps,outs = [],[]
						def layer_hook(module, inp, out):
							outs.append(out.data)
						hook = model.layer4.register_forward_hook(layer_hook)
						_ = model(x_batch)
						batch_grads = outs[0].view(outs[0].size(0), -1).squeeze(0)
						hook.remove()
					elif self.args.model == 'mobilenet_v3_large':
						inps,outs = [],[]
						def layer_hook(module, inp, out):
							outs.append(out.data)
						hook = model.avgpool.register_forward_hook(layer_hook)
						_ = model(x_batch)
						batch_grads = outs[0].view(outs[0].size(0), -1).squeeze(0)
						hook.remove()
					elif self.args.model == 'densenet161':
						inps,outs = [],[]
						def layer_hook(module, inp, out):
							outs.append(out.data)
						hook = model.features.register_forward_hook(layer_hook)
						_ = model(x_batch)
						outs[0] = torch.nn.functional.relu(outs[0])
						batch_grads = outs[0].view(outs[0].size(0), -1).squeeze(0)
						hook.remove()
					elif self.args.model == 'efficientnet_b3':
						inps,outs = [],[]
						def layer_hook(module, inp, out):
							outs.append(out.data)
						hook = model.avgpool.register_forward_hook(layer_hook)
						_ = model(x_batch)
						batch_grads = outs[0].view(outs[0].size(0), -1).squeeze(0)
						hook.remove()
					elif self.args.model == 'convnext_tiny':
						inps,outs = [],[]
						def layer_hook(module, inp, out):
							outs.append(out.data)
						hook = model.avgpool.register_forward_hook(layer_hook)
						_ = model(x_batch)
						batch_grads = outs[0].view(outs[0].size(0), -1).squeeze(0)
						hook.remove()
					elif self.args.model == 'vit_b_16':
						inps,outs = [],[]
						def layer_hook(module, inp, out):
							inps.append(inp[0].data)
						hook = model[1].heads.register_forward_hook(layer_hook)
						_ = model(x_batch)
						batch_grads = inps[0].view(inps[0].size(0), -1).squeeze(0)
						hook.remove()
				
				if iex==0:
					full_cov = np.zeros(shape=(cur_examples, len(batch_grads)))
				full_cov[iex] = batch_grads.detach().cpu().numpy()

			### c. detect the backdoor data by the SVD decomposition
			total_p = self.args.percentile            
			full_mean = np.mean(full_cov, axis=0, keepdims=True)            
		
			centered_cov = full_cov - full_mean
			u,s,v = np.linalg.svd(centered_cov, full_matrices=False)
			logging.info(f'Top 7 Singular Values: {s[0:7]}')
			eigs = v[0:1]  
			p = total_p
			corrs = np.matmul(eigs, np.transpose(full_cov)) #shape num_top, num_active_indices
			scores = np.linalg.norm(corrs, axis=0) #shape num_active_indices
			logging.info(f'Length Scores: {len(scores)}' )
			p_score = np.percentile(scores, p)
			top_scores = np.where(scores>p_score)[0]
			logging.info(f'{top_scores}')
			

			removed_inds = np.copy(top_scores)
			re = [cur_indices[v] for i,v in enumerate(removed_inds)]
			re_all.extend(re)
			
		left_inds = np.delete(range(len(dataset)), re_all)
		### d. retrain the model with remaining data
		model = generate_cls_model(self.args.model,self.args.num_classes)
		if "," in self.device:
			model = torch.nn.DataParallel(
				model,
				device_ids=[int(i) for i in args.device[5:].split(",")]  # eg. "cuda:2,3,7" -> [2,3,7]
			)
			self.args.device = f'cuda:{model.device_ids[0]}'
			model.to(self.args.device)
		else:
			model.to(self.args.device)
		dataset.subset(left_inds)
		dataset.wrapped_dataset.getitem_all = True
		# dataset.subset(left_inds)
		dataset_left = dataset
		data_loader_sie = torch.utils.data.DataLoader(dataset_left, batch_size=self.args.batch_size, num_workers=self.args.num_workers, shuffle=True)
		
		optimizer, scheduler = argparser_opt_scheduler(model, self.args)
		# criterion = nn.CrossEntropyLoss()
		self.set_trainer(model)
		criterion = argparser_criterion(args)

		test_tran = get_transform(self.args.dataset, *([self.args.input_height,self.args.input_width]) , train = False)
		data_bd_testset = self.result['bd_test']
		data_bd_testset.wrap_img_transform = test_tran
		data_bd_loader = torch.utils.data.DataLoader(data_bd_testset, batch_size=self.args.batch_size, num_workers=self.args.num_workers,drop_last=False, shuffle=True,pin_memory=args.pin_memory)

		data_clean_testset = self.result['clean_test']
		data_clean_testset.wrap_img_transform = test_tran
		data_clean_loader = torch.utils.data.DataLoader(data_clean_testset, batch_size=self.args.batch_size, num_workers=self.args.num_workers,drop_last=False, shuffle=True,pin_memory=args.pin_memory)

		self.trainer.train_with_test_each_epoch_on_mix(
			data_loader_sie,
			data_clean_loader,
			data_bd_loader,
			args.epochs,
			criterion=criterion,
			optimizer=optimizer,
			scheduler=scheduler,
			device=self.args.device,
			frequency_save=args.frequency_save,
			save_folder_path=args.save_path,
			save_prefix='spectral',
			amp=args.amp,
			prefetch=args.prefetch,
			prefetch_transform_attr_name="ori_image_transform_in_loading", # since we use the preprocess_bd_dataset
			non_blocking=args.non_blocking,
		)

		result = {}
		result["dataset"] = dataset_left
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
	spectral_method = spectral()
	result = spectral_method.defense(spectral_method.args.result_file)