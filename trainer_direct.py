"""
basic trainer
"""
from builtins import isinstance
from importlib.machinery import DEBUG_BYTECODE_SUFFIXES
import time
import os
import torch.autograd
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import utils as utils
import numpy as np
import torch
from pytorchcv.models.resnet import ResUnit
# from pytorchcv.models.mobilenet import DwsConvBlock
# from pytorchcv.models.mobilenetv2 import LinearBottleneck
from models.models import Bottleneck, BasicBlock
from quantization_utils.quant_modules import *
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
try:
	import medmnist
	from medmnist import INFO, Evaluator
	MEDMNIST_AVAILABLE = True
except ImportError:
	MEDMNIST_AVAILABLE = False


__all__ = ["Trainer"]


class Trainer(object):
	"""
	trainer for training network, use SGD
	"""
	
	def __init__(self, model, model_teacher, lr_master_S, lr_master_G,
	             train_loader, test_loader, settings, args, logger, tensorboard_logger=None,
	             opt_type="SGD", optimizer_state=None, run_count=0):
		"""
		init trainer
		"""
		
		self.settings = settings
		self.args = args
		self.model = model
		self.model_teacher = model_teacher
		self.train_loader = train_loader
		self.test_loader = test_loader
		self.tensorboard_logger = tensorboard_logger
		# self.tensorboard_logger = SummaryWriter(self.settings.save_path) if dist.get_rank()==0 else None
		self.criterion = nn.CrossEntropyLoss().to(self.args.local_rank)
		self.bce_logits = nn.BCEWithLogitsLoss().to(self.args.local_rank)
		self.MSE_loss = nn.MSELoss().to(self.args.local_rank)
		self.KLloss = nn.KLDivLoss(reduction='batchmean').to(self.args.local_rank)
		self.lr_master_S = lr_master_S
		self.lr_master_G = lr_master_G
		self.logger = logger
		
		# Always use compute_singlecrop for accuracy calculation (no evaluator)
		self.use_evaluator = False
		self.task = 'multi-class'
		self.logger.info("Using utils.compute_singlecrop for all accuracy calculations")
		self.lr_master_G = lr_master_G
		self.opt_type = opt_type
		if opt_type == "SGD":
			self.optimizer_S = torch.optim.SGD(
				params=self.model.parameters(),
				lr=self.lr_master_S.lr,
				momentum=self.settings.momentum,
				weight_decay=self.settings.weightDecay,
				nesterov=True,
			)
		elif opt_type == "RMSProp":
			self.optimizer_S = torch.optim.RMSprop(
				params=self.model.parameters(),
				lr=self.lr_master_S.lr,
				eps=1.0,
				weight_decay=self.settings.weightDecay,
				momentum=self.settings.momentum,
				alpha=self.settings.momentum
			)
		elif opt_type == "Adam":
			self.optimizer_S = torch.optim.Adam(
				params=self.model.parameters(),
				lr=self.lr_master_S.lr,
				eps=1e-5,
				weight_decay=self.settings.weightDecay
			)
		else:
			assert False, "invalid type: %d" % opt_type
		if optimizer_state is not None:
			self.optimizer_S.load_state_dict(optimizer_state)

		self.run_count = run_count
		self.scalar_info = {}
		self.mean_list = []
		self.var_list = []
		self.teacher_running_mean = []
		self.teacher_running_var = []
		self.save_BN_mean = []
		self.save_BN_var = []
		self.activation_teacher = []
		self.activation = []
		self.handle_list = []
		

	def update_lr(self, epoch):
		"""
		update learning rate of optimizers
		:param epoch: current training epoch
		"""
		lr_S = self.lr_master_S.get_lr(epoch)
		lr_G = self.lr_master_G.get_lr(epoch)
		# update learning rate of model optimizer
		for param_group in self.optimizer_S.param_groups:
			param_group['lr'] = lr_S

	def loss_fn_kd(self, output, labels, teacher_outputs):
		"""
		Compute the knowledge-distillation (KD) loss given outputs, labels.
		"Hyperparameters": temperature and alpha

		NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
		and student expects the input tensor to be log probabilities! See Issue #2
		"""	
		alpha = self.settings.alpha
		T = self.settings.temperature
		a = F.log_softmax(output / T, dim=1)
		b = F.softmax(teacher_outputs / T, dim=1)
		c = (alpha * T * T)
		
		# Properly format labels for loss calculation
		if self.settings.train_dataset != self.settings.test_dataset:
			d = torch.tensor(0.0).to(self.args.local_rank)
		else:
			if len(labels.shape) > 1 and labels.shape[1] > 1:
				# Multi-label case
				labels_formatted = labels.to(torch.float32)
				# For multi-label, use different loss function
				d = self.bce_logits(output, labels_formatted)
			else:
				# Single-label case - squeeze if necessary
				labels_formatted = torch.squeeze(labels, 1).long() if len(labels.shape) > 1 else labels.long()
				d = self.criterion(output, labels_formatted)
		
		KD_loss = self.KLloss(a, b) * c
		return KD_loss, d

	def loss_fa(self):
		fa = torch.zeros(1).to(self.args.local_rank)
		for l in range(len(self.activation)):
			fa += (self.activation[l] - self.activation_teacher[l]).pow(2).mean()
		fa = self.settings.lam * fa
		return fa
	
	def forward(self, images, teacher_outputs, labels=None):
		"""
		forward propagation
		"""
		# forward and backward and optimize
		output = self.model(images)
		loss_KL, loss_CE = self.loss_fn_kd(output, labels, teacher_outputs)
		loss_FA = self.loss_fa()
		return output, loss_KL, loss_FA, loss_CE
	
	# def backward_G(self, loss_G):
	# 	"""
	# 	backward propagation
	# 	"""
	# 	self.optimizer_G.zero_grad()
	# 	loss_G.backward()
	# 	self.optimizer_G.step()

	def backward_S(self, loss_S):
		"""
		backward propagation
		"""
		self.optimizer_S.zero_grad()
		loss_S.backward()
		self.optimizer_S.step()

	def backward(self, loss):
		"""
		backward propagation
		"""
		self.optimizer_S.zero_grad()
		loss.backward()
		self.optimizer_S.step()

	def reduce_minmax(self):
		for m in self.model.module.modules():
			if isinstance(m, QuantAct):
				dist.all_reduce(m.x_min, op=dist.ReduceOp.SUM)
				dist.all_reduce(m.x_max, op=dist.ReduceOp.SUM)
				m.x_min = m.x_min / dist.get_world_size()
				m.x_max = m.x_max / dist.get_world_size()

	def spatial_attention(self, x):
		return F.normalize(x.pow(2).mean([1]).view(x.size(0), -1))

	def channel_attention(self, x):
		return F.normalize(x.pow(2).mean([2,3]).view(x.size(0), -1))

	def hook_activation_teacher(self, module, input, output):
		self.activation_teacher.append(self.channel_attention(output.clone()))

	def hook_activation(self, module, input, output):
		self.activation.append(self.channel_attention(output.clone()))

	def hook_fn_forward(self,module, input, output):
		input = input[0]
		mean = input.mean([0, 2, 3])
		# use biased var in train
		var = input.var([0, 2, 3], unbiased=False)

		self.mean_list.append(mean)
		self.var_list.append(var)
		self.teacher_running_mean.append(module.running_mean)
		self.teacher_running_var.append(module.running_var)
	
	def train(self, epoch, direct_dataload=None):
		"""
		training
		"""
		top1_error = utils.AverageMeter()
		top1_loss = utils.AverageMeter()
		top5_error = utils.AverageMeter()
		fp_acc = utils.AverageMeter()

		self.update_lr(epoch)

		self.model.train()
		self.model_teacher.eval()
		
		start_time = time.time()
		end_time = start_time
		
		self.logger.info(f"Starting training epoch {epoch + 1}/{self.settings.nEpochs}")
		print(f"Starting training epoch {epoch + 1}/{self.settings.nEpochs}")
		
		if epoch==0:
			#register BN hook
			for m in self.model_teacher.modules():
				if isinstance(m, nn.SyncBatchNorm):
					handle = m.register_forward_hook(self.hook_fn_forward)
					self.handle_list.append(handle)
			self.logger.info("Registered BN hooks for teacher model")
			print("Registered BN hooks for teacher model")


		if epoch == 4:
			#remove BN hook
			for handle in self.handle_list:
				handle.remove()
			self.reduce_minmax()
			self.logger.info("Removed BN hooks and reduced minmax values")
			print("Removed BN hooks and reduced minmax values")

			# Register hooks for ResNet layers (layer1, layer2, layer3, layer4)
			# Teacher model hooks
			if hasattr(self.model_teacher, 'layer1'):
				self.model_teacher.layer1.register_forward_hook(self.hook_activation_teacher)
				print("Registered teacher layer1 hook")
			if hasattr(self.model_teacher, 'layer2'):
				self.model_teacher.layer2.register_forward_hook(self.hook_activation_teacher)
				print("Registered teacher layer2 hook")
			if hasattr(self.model_teacher, 'layer3'):
				self.model_teacher.layer3.register_forward_hook(self.hook_activation_teacher)
				print("Registered teacher layer3 hook")
			if hasattr(self.model_teacher, 'layer4'):
				self.model_teacher.layer4.register_forward_hook(self.hook_activation_teacher)
				print("Registered teacher layer4 hook")
			
			# Student model hooks
			if hasattr(self.model.module, 'layer1'):
				self.model.module.layer1.register_forward_hook(self.hook_activation)
				print("Registered student layer1 hook")
			if hasattr(self.model.module, 'layer2'):
				self.model.module.layer2.register_forward_hook(self.hook_activation)
				print("Registered student layer2 hook")
			if hasattr(self.model.module, 'layer3'):
				self.model.module.layer3.register_forward_hook(self.hook_activation)
				print("Registered student layer3 hook")
			if hasattr(self.model.module, 'layer4'):
				self.model.module.layer4.register_forward_hook(self.hook_activation)
				print("Registered student layer4 hook")
			
			self.logger.info("Registered activation hooks for ResNet layers (layer1-4) for feature alignment")
			print("Registered activation hooks for ResNet layers (layer1-4) for feature alignment")

		for i, (images, labels) in enumerate(self.train_loader):

			start_time = time.time()
			data_time = start_time - end_time

			# Move data to device
			images, labels = images.to(self.args.local_rank), labels.to(self.args.local_rank)

			if epoch < 4:
				# During early epochs, use BN statistics matching
				self.mean_list.clear()
				self.var_list.clear()
				output_teacher_batch = self.model_teacher(images)
				
				# Train student model
				output = self.model(images)
				
				# Properly format labels for loss calculation
				if self.settings.train_dataset != self.settings.test_dataset:
					loss_one_hot = torch.tensor(0.0).to(self.args.local_rank)
				else:
					if len(labels.shape) > 1 and labels.shape[1] > 1:
						# Multi-label case
						labels_formatted = labels.to(torch.float32)
						loss_one_hot = self.criterion(output, labels_formatted)
					else:
						# Single-label case - squeeze if necessary
						labels_formatted = torch.squeeze(labels, 1).long() if len(labels.shape) > 1 else labels.long()
						loss_one_hot = self.criterion(output, labels_formatted)
				
				# BN statistic loss
				BNS_loss = torch.zeros(1).to(self.args.local_rank)
				for num in range(len(self.mean_list)):
					BNS_loss += self.MSE_loss(self.mean_list[num], self.teacher_running_mean[num]) + self.MSE_loss(
						self.var_list[num], self.teacher_running_var[num])

				if len(self.mean_list) > 0:
					BNS_loss = BNS_loss / len(self.mean_list)
				
				loss_S = loss_one_hot + 0.1 * BNS_loss
				# self.backward_S(loss_S)
			else:
				# After epoch 4, use feature alignment and knowledge distillation
				self.activation_teacher.clear()
				self.activation.clear()

				output_teacher_batch = self.model_teacher(images)
				output, loss_KL, loss_FA, loss_CE = self.forward(images, output_teacher_batch, labels)
				loss_S = loss_KL + loss_FA*0 + loss_CE*0

				self.activation_teacher.clear()
				self.activation.clear()

				loss_total = loss_S 

				self.backward_S(loss_total)

			# Use compute_singlecrop for accuracy calculation
			if self.settings.train_dataset != self.settings.test_dataset:
				# Log progress every 10 batches
				if (i + 1) % 10 == 0:
					if epoch < 4:
						log_msg = f"[Epoch {epoch + 1}/{self.settings.nEpochs}] [Batch {i + 1}/{len(self.train_loader)}] [Student loss: {loss_S.item():.6f}] [One-hot loss: {loss_one_hot.item():.6f}] [BNS_loss:{BNS_loss.item():.6f}]"
						self.logger.info(log_msg)
						print(log_msg)
					else:
						log_msg = f"[Epoch {epoch + 1}/{self.settings.nEpochs}] [Batch {i + 1}/{len(self.train_loader)}] [loss KL: {loss_KL.item():.6f}] [loss FA: {loss_FA.item():.6f}] [loss CE: {loss_CE.item():.6f}]"
						self.logger.info(log_msg)
						print(log_msg)
			else:
				single_error, single_loss, single5_error = utils.compute_singlecrop(
					outputs=output, labels=labels,
					loss=loss_S, top5_flag=True, mean_flag=True)
				
				top1_error.update(single_error, images.size(0))
				top1_loss.update(single_loss, images.size(0))
				top5_error.update(single5_error, images.size(0))
				
				end_time = time.time()
				
				# Calculate accuracy from error rate
				acc = 100.0 - single_error
				fp_acc.update(acc)
				
				# Log progress every 10 batches
				if (i + 1) % 10 == 0:
					if epoch < 4:
						log_msg = f"[Epoch {epoch + 1}/{self.settings.nEpochs}] [Batch {i + 1}/{len(self.train_loader)}] [acc: {fp_acc.avg:.4f}%] [Student loss: {loss_S.item():.6f}] [One-hot loss: {loss_one_hot.item():.6f}] [BNS_loss:{BNS_loss.item():.6f}]"
						self.logger.info(log_msg)
						print(log_msg)
					else:
						log_msg = f"[Epoch {epoch + 1}/{self.settings.nEpochs}] [Batch {i + 1}/{len(self.train_loader)}] [acc: {fp_acc.avg:.4f}%] [loss KL: {loss_KL.item():.6f}] [loss FA: {loss_FA.item():.6f}] [loss CE: {loss_CE.item():.6f}]"
						self.logger.info(log_msg)
						print(log_msg)
		
		# Log progress at the end of epoch
		if self.settings.train_dataset != self.settings.test_dataset:
			if epoch < 4:
				final_log_msg = f"[Epoch {epoch + 1}/{self.settings.nEpochs}] [Batches: {len(self.train_loader)}] [Student loss: {loss_S.item():.6f}] [One-hot loss: {loss_one_hot.item():.6f}] [BNS_loss:{BNS_loss.item():.6f}]"
				self.logger.info(final_log_msg)
				print(final_log_msg)
			else:
				final_log_msg = f"[Epoch {epoch + 1}/{self.settings.nEpochs}] [Batches: {len(self.train_loader)}] [loss KL: {loss_KL.item():.6f}] [loss FA: {loss_FA.item():.6f}]"
				self.logger.info(final_log_msg)
				print(final_log_msg)
		else:
			if epoch < 4:
				final_log_msg = f"[Epoch {epoch + 1}/{self.settings.nEpochs}] [Batches: {len(self.train_loader)}] [acc: {fp_acc.avg:.4f}%] [Student loss: {loss_S.item():.6f}] [One-hot loss: {loss_one_hot.item():.6f}] [BNS_loss:{BNS_loss.item():.6f}]"
				self.logger.info(final_log_msg)
				print(final_log_msg)
			else:
				final_log_msg = f"[Epoch {epoch + 1}/{self.settings.nEpochs}] [Batches: {len(self.train_loader)}] [acc: {fp_acc.avg:.4f}%] [loss KL: {loss_KL.item():.6f}] [loss FA: {loss_FA.item():.6f}]"
				self.logger.info(final_log_msg)
				print(final_log_msg)

			
		# self.scalar_info['accuracy every epoch'] = 100 * d_acc
		# self.scalar_info['training_top1error'] = top1_error.avg
		# self.scalar_info['training_top5error'] = top5_error.avg
		# self.scalar_info['training_loss'] = top1_loss.avg
		
		# if self.tensorboard_logger is not None:
		# 	for tag, value in list(self.scalar_info.items()):
		# 		self.tensorboard_logger.scalar_summary(tag, value, self.run_count)
		# 	self.scalar_info = {}

		return top1_error.avg, top1_loss.avg, top5_error.avg


	def test(self, epoch):
		"""
		testing
		"""
		top1_error = utils.AverageMeter()
		top1_loss = utils.AverageMeter()
		top5_error = utils.AverageMeter()
		
		self.model.eval()
		self.model_teacher.eval()
		
		iters = len(self.test_loader)
		start_time = time.time()
		end_time = start_time
		# g=[]
		with torch.no_grad():
			for i, (images, labels) in enumerate(self.test_loader):
				start_time = time.time()
				
				labels = labels.to(self.args.local_rank)
				images = images.to(self.args.local_rank)
				output = self.model(images)
				loss = torch.ones(1)
				self.mean_list.clear()
				self.var_list.clear()

				single_error, single_loss, single5_error = utils.compute_singlecrop(
					outputs=output, loss=loss,
					labels=labels, top5_flag=True, mean_flag=True)

				top1_error.update(single_error, images.size(0))
				top1_loss.update(single_loss, images.size(0))
				top5_error.update(single5_error, images.size(0))
				
				end_time = time.time()
		self.logger.info(
			"[Epoch %d/%d] [Batch %d/%d] [acc: %.4f%%]"
			% (epoch + 1, self.settings.nEpochs, i + 1, iters, (100.00-top1_error.avg))
		)
		print(
			"[Epoch %d/%d] [Batch %d/%d] [acc: %.4f%%]"
			% (epoch + 1, self.settings.nEpochs, i + 1, iters, (100.00-top1_error.avg))
		)
		
		# self.scalar_info['testing_top1error'] = top1_error.avg
		# self.scalar_info['testing_top5error'] = top5_error.avg
		# self.scalar_info['testing_loss'] = top1_loss.avg
		# if self.tensorboard_logger is not None:
		# 	for tag, value in self.scalar_info.items():
		# 		self.tensorboard_logger.scalar_summary(tag, value, self.run_count)
		# 	self.scalar_info = {}
		self.run_count += 1

		return top1_error.avg, top1_loss.avg, top5_error.avg


	def test_teacher(self, epoch):
		"""
		testing
		"""
		top1_error = utils.AverageMeter()
		top1_loss = utils.AverageMeter()
		top5_error = utils.AverageMeter()

		self.model_teacher.eval()

		iters = len(self.test_loader)
		start_time = time.time()
		end_time = start_time

		with torch.no_grad():
			for i, (images, labels) in enumerate(self.test_loader):
				start_time = time.time()
				data_time = start_time - end_time

				labels = labels.to(self.args.local_rank)

				if self.settings.tenCrop:
					image_size = images.size()
					images = images.view(
						image_size[0] * 10, image_size[1] / 10, image_size[2], image_size[3])
					images_tuple = images.split(image_size[0])
					output = None
					for img in images_tuple:
						if self.settings.nGPU == 1:
							img = img.to(self.args.local_rank)
						img_var = Variable(img, volatile=True)
						temp_output, _ = self.forward(img_var)
						if output is None:
							output = temp_output.data
						else:
							output = torch.cat((output, temp_output.data))
					single_error, single_loss, single5_error = utils.compute_tencrop(
						outputs=output, labels=labels)
				else:
					if self.settings.nGPU == 1:
						images = images.to(self.args.local_rank)
					self.activation_teacher.clear()
					output = self.model_teacher(images)

					loss = torch.ones(1)
					self.mean_list.clear()
					self.var_list.clear()

					single_error, single_loss, single5_error = utils.compute_singlecrop(
						outputs=output, loss=loss,
						labels=labels, top5_flag=True, mean_flag=True)
				#
				top1_error.update(single_error, images.size(0))
				top1_loss.update(single_loss, images.size(0))
				top5_error.update(single5_error, images.size(0))

				end_time = time.time()
				iter_time = end_time - start_time

		print(
				"Teacher network: [Epoch %d/%d] [Batch %d/%d] [acc: %.4f%%]"
				% (epoch + 1, self.settings.nEpochs, i + 1, iters, (100.00 - top1_error.avg))
		)

		self.run_count += 1

		return top1_error.avg, top1_loss.avg, top5_error.avg
