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
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
try:
	import medmnist
	from medmnist import INFO, Evaluator
	MEDMNIST_AVAILABLE = True
except ImportError:
	MEDMNIST_AVAILABLE = False


__all__ = ["Trainer"]


class NewBatchNorm2d(nn.Module):
	"""
	New BatchNorm layer that uses current input statistics for normalization
	but applies old beta and gamma parameters for scale and shift
	"""
	def __init__(self, original_bn):
		super(NewBatchNorm2d, self).__init__()
		self.num_features = original_bn.num_features
		self.eps = original_bn.eps
		self.momentum = original_bn.momentum
		
		# Store original parameters (old_beta, old_gamma)
		self.register_buffer('old_weight', original_bn.weight.clone().detach())
		self.register_buffer('old_bias', original_bn.bias.clone().detach())
		
		# Running statistics for new BN (will be updated during training)
		self.register_buffer('running_mean', torch.zeros(self.num_features))
		self.register_buffer('running_var', torch.ones(self.num_features))
		self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
		
		# Store reference to original BN for easy restoration
		self.original_bn = original_bn
	
	def forward(self, input):
		# Use in-place operations and avoid unnecessary tensor creation
		if self.training:
			# Calculate mean and variance from current batch more efficiently
			batch_mean = input.mean([0, 2, 3], keepdim=False)
			batch_var = input.var([0, 2, 3], unbiased=False, keepdim=False)
			
			# # Update running statistics with in-place operations
			# if self.momentum is None:
			# 	exponential_average_factor = 0.0
			# else:
			# 	exponential_average_factor = self.momentum
			
			# # In-place updates to save memory
			# with torch.no_grad():
			# 	self.running_mean.mul_(1 - exponential_average_factor).add_(batch_mean, alpha=exponential_average_factor)
			# 	self.running_var.mul_(1 - exponential_average_factor).add_(batch_var, alpha=exponential_average_factor)
			
			# Use batch statistics for normalization
			mean = batch_mean.detach()  # Detach to prevent requires_grad issues
			var = batch_var.detach()    # Detach to prevent requires_grad issues
		else:
			# Use running statistics for normalization during evaluation
			mean = self.running_mean.detach()  # Detach to prevent requires_grad issues
			var = self.running_var.detach()    # Detach to prevent requires_grad issues
		
		# Use F.batch_norm for efficient computation instead of manual operations
		# This is more memory efficient than manual tensor operations
		# Detach mean and var to avoid autograd issues with running statistics
		return F.batch_norm(
			input, 
			mean.detach(), 
			var.detach(), 
			self.old_weight, 
			self.old_bias, 
			training=False,  # We handle the statistics ourselves
			momentum=0.0,    # No momentum since we handle it ourselves
			eps=self.eps
		)


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
		
		# Store original BN layers for test time
		self.original_bn_layers = {}
		self.new_bn_layers = {}
		
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
	
	def backward_G(self, loss_G):
		"""
		backward propagation
		"""
		self.optimizer_G.zero_grad()
		loss_G.backward()
		self.optimizer_G.step()

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

	def freeze_batchnorm(self, model):
		"""
		freeze all BatchNorm parameters (weight, bias, running_mean, running_var)
		"""
		if isinstance(model, DDP):
			model = model.module
			
		for module in model.modules():
			if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)):
				# Set to eval mode to prevent running statistics update
				# module.train()
				# module.eval()
				
				# Freeze parameters
				if module.weight is not None:
					module.weight.requires_grad = False
				if module.bias is not None:
					module.bias.requires_grad = False
				
				# Optionally freeze running statistics (they won't update in eval mode anyway)
				module.track_running_stats = False
				print(f"Freezing BatchNorm module: {type(module).__name__}")
	
	def unfreeze_batchnorm(self, model):
		"""
		unfreeze all BatchNorm parameters
		"""
		if isinstance(model, DDP):
			model = model.module
			
		for module in model.modules():
			if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)):
				# Unfreeze parameters
				if module.weight is not None:
					module.weight.requires_grad = True
				if module.bias is not None:
					module.bias.requires_grad = True
				# Allow running statistics update
				module.track_running_stats = True
	
	def freeze_classifier(self, model):
		"""
		freeze the classifier (last layer) parameters
		"""
		if isinstance(model, DDP):
			model = model.module
		
		# Common classifier layer names in different architectures
		classifier_names = ['fc', 'classifier', 'linear', 'head']
		
		for name, module in model.named_modules():
			# Check if this is likely the final classifier layer
			if any(classifier_name in name.lower() for classifier_name in classifier_names):
				# Additional check: should be a Linear layer and likely the last one
				if isinstance(module, (nn.Linear, Quant_Linear)):
					# Freeze parameters
					if module.weight is not None:
						module.weight.requires_grad = False
						print(f"Freezing classifier weight: {name}")
					if module.bias is not None:
						module.bias.requires_grad = False
						print(f"Freezing classifier bias: {name}")
		
		# Alternative approach: freeze the last Linear layer
		last_linear = None
		last_linear_name = ""
		for name, module in model.named_modules():
			if isinstance(module, nn.Linear):
				last_linear = module
				last_linear_name = name
		
		if last_linear is not None and last_linear_name:
			if last_linear.weight is not None:
				last_linear.weight.requires_grad = False
				print(f"Freezing last linear layer weight: {last_linear_name}")
			if last_linear.bias is not None:
				last_linear.bias.requires_grad = False
				print(f"Freezing last linear layer bias: {last_linear_name}")
	
	def unfreeze_classifier(self, model):
		"""
		unfreeze the classifier (last layer) parameters
		"""
		if isinstance(model, DDP):
			model = model.module
		
		# Common classifier layer names in different architectures
		classifier_names = ['fc', 'classifier', 'linear', 'head']
		
		for name, module in model.named_modules():
			# Check if this is likely the final classifier layer
			if any(classifier_name in name.lower() for classifier_name in classifier_names):
				# Additional check: should be a Linear layer
				if isinstance(module, nn.Linear):
					# Unfreeze parameters
					if module.weight is not None:
						module.weight.requires_grad = True
						print(f"Unfreezing classifier weight: {name}")
					if module.bias is not None:
						module.bias.requires_grad = True
						print(f"Unfreezing classifier bias: {name}")
		
		# Alternative approach: unfreeze the last Linear layer
		last_linear = None
		last_linear_name = ""
		for name, module in model.named_modules():
			if isinstance(module, nn.Linear):
				last_linear = module
				last_linear_name = name
		
		if last_linear is not None and last_linear_name:
			if last_linear.weight is not None:
				last_linear.weight.requires_grad = True
				print(f"Unfreezing last linear layer weight: {last_linear_name}")
			if last_linear.bias is not None:
				last_linear.bias.requires_grad = True
				print(f"Unfreezing last linear layer bias: {last_linear_name}")

	def train(self, epoch, direct_dataload=None):
		"""
		training
		"""
		top1_error = utils.AverageMeter()
		top1_loss = utils.AverageMeter()
		top5_error = utils.AverageMeter()
		fp_acc = utils.AverageMeter()
		loss_train = utils.AverageMeter()  # Add average meter for loss_total

		self.update_lr(epoch)

		self.model.train()
		self.model_teacher.eval()

		# Switch to new BN mode for training
		self.switch_to_new_bn_mode()
		
		# Clear CUDA cache at the beginning of each epoch
		if torch.cuda.is_available():
			torch.cuda.empty_cache()
		
		start_time = time.time()
		end_time = start_time
		
		self.logger.info(f"Starting training epoch {epoch + 1}/{self.settings.nEpochs}")
		print(f"Starting training epoch {epoch + 1}/{self.settings.nEpochs}")
		
		# Initialize class counters for training predictions
		num_classes = getattr(self.settings, 'nClasses', 8)  # Default to 8 for tissuemnist
		student_pred_counts = [0] * num_classes
		teacher_pred_counts = [0] * num_classes
		true_train_counts = [0] * num_classes
		
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

			# self.freeze_batchnorm(self.model)
			# self.freeze_classifier(self.model)
			# self.logger.info(f"Epoch {epoch}: Freezed BatchNorm and classifier parameters")
			# print(f"Epoch {epoch}: Freezed BatchNorm and classifier parameters")

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
				
				# Count predictions for both models
				with torch.no_grad():
					_, student_predicted = torch.max(output.data, 1)
					_, teacher_predicted = torch.max(output_teacher_batch.data, 1)
					
					# Count true labels and predictions
					for j in range(labels.size(0)):
						# Handle true labels
						true_label = labels[j].item()
						if len(labels.shape) > 1 and labels.shape[1] > 1:
							true_label = torch.squeeze(labels[j], 0).long().item() if len(labels[j].shape) > 0 else labels[j].item()
						elif len(labels.shape) > 1:
							true_label = torch.squeeze(labels[j], 0).long().item()
						
						student_pred = student_predicted[j].item()
						teacher_pred = teacher_predicted[j].item()
						
						# Count if within valid range
						if 0 <= true_label < num_classes:
							true_train_counts[true_label] += 1
						if 0 <= student_pred < num_classes:
							student_pred_counts[student_pred] += 1
						if 0 <= teacher_pred < num_classes:
							teacher_pred_counts[teacher_pred] += 1
				
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
				
				# Update loss_total average for epoch < 4
				loss_train.update(loss_S.item(), images.size(0))
				
				# Backward pass for early epochs
				# self.backward_S(loss_S)
				
				# Clear intermediate variables to free memory
				del output_teacher_batch, output
				if 'labels_formatted' in locals():
					del labels_formatted

			else:
				# After epoch 4, use feature alignment and knowledge distillation
				# Unfreeze model parameters for normal training
				for param in self.model.parameters():
					param.requires_grad = True
				
				self.activation_teacher.clear()
				self.activation.clear()

				output_teacher_batch = self.model_teacher(images)
				output, loss_KL, loss_FA, loss_CE = self.forward(images, output_teacher_batch, labels)
				
				# Add entropy loss to prevent mode collapse
				loss_entropy = self.entropy_loss(output)
				loss_diversity = self.prediction_diversity_loss(output)
				
				# Combine all losses
				# You can adjust the weights (0.1 for entropy, 0.05 for diversity) based on your needs
				loss_S = loss_KL + loss_FA + loss_CE*0 + loss_entropy*0 + loss_diversity*0

				# Count predictions for both models
				with torch.no_grad():
					_, student_predicted = torch.max(output.data, 1)
					_, teacher_predicted = torch.max(output_teacher_batch.data, 1)
					
					# Count true labels and predictions
					for j in range(labels.size(0)):
						# Handle true labels
						true_label = labels[j].item()
						if len(labels.shape) > 1 and labels.shape[1] > 1:
							true_label = torch.squeeze(labels[j], 0).long().item() if len(labels[j].shape) > 0 else labels[j].item()
						elif len(labels.shape) > 1:
							true_label = torch.squeeze(labels[j], 0).long().item()
						
						student_pred = student_predicted[j].item()
						teacher_pred = teacher_predicted[j].item()
						
						# Count if within valid range
						if 0 <= true_label < num_classes:
							true_train_counts[true_label] += 1
						if 0 <= student_pred < num_classes:
							student_pred_counts[student_pred] += 1
						if 0 <= teacher_pred < num_classes:
							teacher_pred_counts[teacher_pred] += 1

				self.activation_teacher.clear()
				self.activation.clear()

				loss_total = loss_S 

				# Update loss_total average for epoch >= 4
				loss_train.update(loss_total.item(), images.size(0))

				self.backward_S(loss_total)
			
			# Clear intermediate variables to free memory
			del images, labels
			if 'output_teacher_batch' in locals():
				del output_teacher_batch
			if 'output' in locals():
				del output
			
			# Clear CUDA cache periodically to prevent memory accumulation
			if (i + 1) % 5 == 0 and torch.cuda.is_available():
				torch.cuda.empty_cache()

			# Use compute_singlecrop for accuracy calculation
			if self.settings.train_dataset != self.settings.test_dataset:
				# Log progress every 10 batches
				if (i + 1) % 10 == 0:
					if epoch < 4:
						log_msg = f"[Epoch {epoch + 1}/{self.settings.nEpochs}] [Batch {i + 1}/{len(self.train_loader)}] [Student loss: {loss_S.item():.6f}] [One-hot loss: {loss_one_hot.item():.6f}] [BNS_loss:{BNS_loss.item():.6f}]"
						self.logger.info(log_msg)
						print(log_msg)
					else:
						log_msg = f"[Epoch {epoch + 1}/{self.settings.nEpochs}] [Batch {i + 1}/{len(self.train_loader)}] [loss KL: {loss_KL.item():.6f}] [loss FA: {loss_FA.item():.6f}] [loss CE: {loss_CE.item():.6f}] [loss Ent: {loss_entropy.item():.6f}] [loss Div: {loss_diversity.item():.6f}]"
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
						log_msg = f"[Epoch {epoch + 1}/{self.settings.nEpochs}] [Batch {i + 1}/{len(self.train_loader)}] [acc: {fp_acc.avg:.4f}%] [loss KL: {loss_KL.item():.6f}] [loss FA: {loss_FA.item():.6f}] [loss CE: {loss_CE.item():.6f}] [loss Ent: {loss_entropy.item():.6f}] [loss Div: {loss_diversity.item():.6f}]"
						self.logger.info(log_msg)
						print(log_msg)
		
		# Log progress at the end of epoch
		print(f"\n=== Training Results for Epoch {epoch + 1} ===")
		print("True class distribution in training:")
		total_true_train = sum(true_train_counts)
		for class_idx in range(num_classes):
			count = true_train_counts[class_idx]
			percentage = (count / total_true_train) * 100 if total_true_train > 0 else 0
			print(f"  Class {class_idx}: {count} images ({percentage:.2f}%)")
		
		print("Student model predictions:")
		total_student_pred = sum(student_pred_counts)
		for class_idx in range(num_classes):
			count = student_pred_counts[class_idx]
			percentage = (count / total_student_pred) * 100 if total_student_pred > 0 else 0
			print(f"  Class {class_idx}: {count} images ({percentage:.2f}%)")
		
		print("Teacher model predictions:")
		total_teacher_pred = sum(teacher_pred_counts)
		for class_idx in range(num_classes):
			count = teacher_pred_counts[class_idx]
			percentage = (count / total_teacher_pred) * 100 if total_teacher_pred > 0 else 0
			print(f"  Class {class_idx}: {count} images ({percentage:.2f}%)")
		print(f"Total training samples: {total_true_train}")
		print()
		
		if self.settings.train_dataset != self.settings.test_dataset:
			if epoch < 4:
				final_log_msg = f"[Epoch {epoch + 1}/{self.settings.nEpochs}] [Batches: {len(self.train_loader)}] [Student loss: {loss_S.item():.6f}] [One-hot loss: {loss_one_hot.item():.6f}] [BNS_loss:{BNS_loss.item():.6f}]"
				self.logger.info(final_log_msg)
				print(final_log_msg)
			else:
				final_log_msg = f"[Epoch {epoch + 1}/{self.settings.nEpochs}] [Batches: {len(self.train_loader)}] [loss KL: {loss_KL.item():.6f}] [loss FA: {loss_FA.item():.6f}] [loss Ent: {loss_entropy.item():.6f}] [loss Div: {loss_diversity.item():.6f}]"
				self.logger.info(final_log_msg)
				print(final_log_msg)
		else:
			if epoch < 4:
				final_log_msg = f"[Epoch {epoch + 1}/{self.settings.nEpochs}] [Batches: {len(self.train_loader)}] [acc: {fp_acc.avg:.4f}%] [Student loss: {loss_S.item():.6f}] [One-hot loss: {loss_one_hot.item():.6f}] [BNS_loss:{BNS_loss.item():.6f}]"
				self.logger.info(final_log_msg)
				print(final_log_msg)
			else:
				final_log_msg = f"[Epoch {epoch + 1}/{self.settings.nEpochs}] [Batches: {len(self.train_loader)}] [acc: {fp_acc.avg:.4f}%] [loss KL: {loss_KL.item():.6f}] [loss FA: {loss_FA.item():.6f}] [loss Ent: {loss_entropy.item():.6f}] [loss Div: {loss_diversity.item():.6f}]"
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

		return top1_error.avg, loss_train.avg, top5_error.avg


	def test(self, epoch):
		"""
		testing
		"""
		# Switch to original BN mode for testing
		self.switch_to_original_bn_mode()
		
		top1_error = utils.AverageMeter()
		top1_loss = utils.AverageMeter()
		top5_error = utils.AverageMeter()
		
		self.model.eval()
		self.model_teacher.eval()
		
		iters = len(self.test_loader)
		start_time = time.time()
		end_time = start_time
		
		# Initialize class counters
		num_classes = getattr(self.settings, 'nClasses', 8)  # Default to 8 for tissuemnist
		true_class_counts = [0] * num_classes
		pred_class_counts = [0] * num_classes
		
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

				# Get predictions
				_, predicted = torch.max(output.data, 1)
				
				# Count true and predicted classes
				for j in range(labels.size(0)):
					true_label = labels[j].item()
					pred_label = predicted[j].item()
					
					# Handle different label formats
					if len(labels.shape) > 1 and labels.shape[1] > 1:
						# Multi-dimensional labels - take the first element or squeeze
						true_label = torch.squeeze(labels[j], 0).long().item() if len(labels[j].shape) > 0 else labels[j].item()
					elif len(labels.shape) > 1:
						# Squeeze if necessary
						true_label = torch.squeeze(labels[j], 0).long().item()
					
					# Ensure labels are within valid range
					if 0 <= true_label < num_classes:
						true_class_counts[true_label] += 1
					if 0 <= pred_label < num_classes:
						pred_class_counts[pred_label] += 1

				single_error, single_loss, single5_error = utils.compute_singlecrop(
					outputs=output, loss=loss,
					labels=labels, top5_flag=True, mean_flag=True)

				top1_error.update(single_error, images.size(0))
				top1_loss.update(single_loss, images.size(0))
				top5_error.update(single5_error, images.size(0))
				
				end_time = time.time()
		
		# Print class distribution
		print(f"\n=== Test Results for Epoch {epoch + 1} ===")
		print("True class distribution:")
		total_true = sum(true_class_counts)
		for class_idx in range(num_classes):
			count = true_class_counts[class_idx]
			percentage = (count / total_true) * 100 if total_true > 0 else 0
			print(f"  Class {class_idx}: {count} images ({percentage:.2f}%)")
		
		print("Predicted class distribution:")
		total_pred = sum(pred_class_counts)
		for class_idx in range(num_classes):
			count = pred_class_counts[class_idx]
			percentage = (count / total_pred) * 100 if total_pred > 0 else 0
			print(f"  Class {class_idx}: {count} images ({percentage:.2f}%)")
		print(f"Total test samples: {total_true}")
		print()
		
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
		# Switch to original BN mode for testing
		self.switch_to_original_bn_mode()
		
		top1_error = utils.AverageMeter()
		top1_loss = utils.AverageMeter()
		top5_error = utils.AverageMeter()

		self.model_teacher.eval()

		iters = len(self.test_loader)
		start_time = time.time()
		end_time = start_time
		
		# Initialize class counters for teacher model
		num_classes = getattr(self.settings, 'nClasses', 8)  # Default to 8 for tissuemnist
		true_class_counts = [0] * num_classes
		teacher_pred_counts = [0] * num_classes

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

					# Get predictions for class counting
					_, teacher_predicted = torch.max(output.data, 1)
					
					# Count true and predicted classes
					for j in range(labels.size(0)):
						true_label = labels[j].item()
						teacher_pred = teacher_predicted[j].item()
						
						# Handle different label formats
						if len(labels.shape) > 1 and labels.shape[1] > 1:
							# Multi-dimensional labels - take the first element or squeeze
							true_label = torch.squeeze(labels[j], 0).long().item() if len(labels[j].shape) > 0 else labels[j].item()
						elif len(labels.shape) > 1:
							# Squeeze if necessary
							true_label = torch.squeeze(labels[j], 0).long().item()
						
						# Ensure labels are within valid range
						if 0 <= true_label < num_classes:
							true_class_counts[true_label] += 1
						if 0 <= teacher_pred < num_classes:
							teacher_pred_counts[teacher_pred] += 1

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

		# Print teacher class distribution
		print(f"\n=== Teacher Test Results for Epoch {epoch + 1} ===")
		print("True class distribution:")
		total_true = sum(true_class_counts)
		for class_idx in range(num_classes):
			count = true_class_counts[class_idx]
			percentage = (count / total_true) * 100 if total_true > 0 else 0
			print(f"  Class {class_idx}: {count} images ({percentage:.2f}%)")
		
		print("Teacher model predictions:")
		total_teacher_pred = sum(teacher_pred_counts)
		for class_idx in range(num_classes):
			count = teacher_pred_counts[class_idx]
			percentage = (count / total_teacher_pred) * 100 if total_teacher_pred > 0 else 0
			print(f"  Class {class_idx}: {count} images ({percentage:.2f}%)")
		print(f"Total test samples: {total_true}")
		print()

		print(
				"Teacher network: [Epoch %d/%d] [Batch %d/%d] [acc: %.4f%%]"
				% (epoch + 1, self.settings.nEpochs, i + 1, iters, (100.00 - top1_error.avg))
		)

		self.run_count += 1


		return top1_error.avg, top1_loss.avg, top5_error.avg
	
	def entropy_loss(self, logits):
		"""
		Compute entropy loss to prevent mode collapse
		Higher entropy encourages the model to make diverse predictions
		"""
		# Convert logits to probabilities
		probs = F.softmax(logits, dim=1)
		
		# Compute entropy for each sample: -sum(p * log(p))
		log_probs = F.log_softmax(logits, dim=1)
		entropy_per_sample = -torch.sum(probs * log_probs, dim=1)
		
		# Return negative entropy (we want to maximize entropy, so minimize negative entropy)
		return -torch.mean(entropy_per_sample)
	
	def prediction_diversity_loss(self, logits):
		"""
		Compute prediction diversity loss to encourage diverse predictions across the batch
		This helps prevent mode collapse by encouraging different samples to have different predictions
		"""
		# Get prediction probabilities
		probs = F.softmax(logits, dim=1)
		
		# Compute mean prediction distribution across the batch
		mean_probs = torch.mean(probs, dim=0)
		
		# Compute entropy of the mean distribution
		# Higher entropy means more diverse predictions across the batch
		log_mean_probs = torch.log(mean_probs + 1e-8)  # Add small epsilon to avoid log(0)
		diversity_entropy = -torch.sum(mean_probs * log_mean_probs)
		
		# Return negative entropy (we want to maximize diversity entropy)
		return -diversity_entropy
	
	def replace_bn_with_new_bn(self, model, model_name="model"):
		"""
		Replace all BatchNorm layers with NewBatchNorm2d layers
		"""
		if isinstance(model, DDP):
			model = model.module
		
		# Clear cache before replacement
		if torch.cuda.is_available():
			torch.cuda.empty_cache()
		
		def replace_bn_recursive(module, name=""):
			for child_name, child_module in module.named_children():
				full_name = f"{name}.{child_name}" if name else child_name
				
				if isinstance(child_module, (nn.BatchNorm2d, nn.SyncBatchNorm)):
					# Store original BN layer
					original_key = f"{model_name}.{full_name}"
					if original_key not in self.original_bn_layers:
						self.original_bn_layers[original_key] = child_module
					
					# Create new BN layer only if not already created
					if original_key not in self.new_bn_layers:
						new_bn = NewBatchNorm2d(child_module)
						new_bn = new_bn.to(child_module.weight.device)
						self.new_bn_layers[original_key] = new_bn
					else:
						new_bn = self.new_bn_layers[original_key]
					
					# Replace the module
					setattr(module, child_name, new_bn)
					# print(f"Replaced {full_name} with NewBatchNorm2d in {model_name}")
				else:
					# Recursively replace in child modules
					replace_bn_recursive(child_module, full_name)
		
		replace_bn_recursive(model)
		
		# Clear cache after replacement
		if torch.cuda.is_available():
			torch.cuda.empty_cache()
	
	def restore_original_bn(self, model, model_name="model"):
		"""
		Restore original BatchNorm layers for testing
		"""
		if isinstance(model, DDP):
			model = model.module
		
		# Clear cache before restoration
		if torch.cuda.is_available():
			torch.cuda.empty_cache()
		
		def restore_bn_recursive(module, name=""):
			for child_name, child_module in module.named_children():
				full_name = f"{name}.{child_name}" if name else child_name
				
				if isinstance(child_module, NewBatchNorm2d):
					# Get original BN layer
					original_key = f"{model_name}.{full_name}"
					if original_key in self.original_bn_layers:
						original_bn = self.original_bn_layers[original_key]
						
						# Restore the original module
						setattr(module, child_name, original_bn)
						# print(f"Restored original BN for {full_name} in {model_name}")
				else:
					# Recursively restore in child modules
					restore_bn_recursive(child_module, full_name)
		
		restore_bn_recursive(model)
		
		# Clear cache after restoration
		if torch.cuda.is_available():
			torch.cuda.empty_cache()
	
	def switch_to_new_bn_mode(self):
		"""
		Switch both teacher and student models to use new BN layers for training
		"""
		print("Switching to new BN mode for training...")
		# Clear CUDA cache before switching
		if torch.cuda.is_available():
			torch.cuda.empty_cache()
		
		self.replace_bn_with_new_bn(self.model_teacher, "teacher")
		self.replace_bn_with_new_bn(self.model, "student")
	
	def switch_to_original_bn_mode(self):
		"""
		Switch both teacher and student models to use original BN layers for testing
		"""
		print("Switching to original BN mode for testing...")
		# Clear CUDA cache before switching
		if torch.cuda.is_available():
			torch.cuda.empty_cache()
		
		self.restore_original_bn(self.model_teacher, "teacher")
		self.restore_original_bn(self.model, "student")
