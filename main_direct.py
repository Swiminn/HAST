import argparse
import datetime
import logging
import os
import time
import traceback
import sys
import copy
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import Dataset
# option file should be modified according to your expriment
from options import Option
import torchvision.transforms as transforms
from dataloader import DataLoader
from trainer_direct import Trainer
import shutil
import utils as utils
from quantization_utils.quant_modules import *
from pytorchcv.model_provider import get_model as ptcv_get_model
from conditional_batchnorm import CategoricalConditionalBatchNorm2d
import pickle
from PIL import Image
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from models.models import ResNet18, ResNet50
from torchvision.models import resnet18
import wandb
# from regularizer import get_reg_criterions

# class Generator(nn.Module):
# 	def __init__(self, options=None, conf_path=None):
# 		super(Generator, self).__init__()
# 		self.settings = options or Option(conf_path)
# 		self.label_emb = nn.Embedding(self.settings.nClasses, self.settings.latent_dim)
# 		self.init_size = self.settings.img_size // 4
# 		self.l1 = nn.Sequential(nn.Linear(self.settings.latent_dim, 128 * self.init_size ** 2))

# 		self.conv_blocks0 = nn.Sequential(
# 			nn.BatchNorm2d(128),
# 		)

# 		self.conv_blocks1 = nn.Sequential(
# 			nn.Conv2d(128, 128, 3, stride=1, padding=1),
# 			nn.BatchNorm2d(128, 0.8),
# 			nn.LeakyReLU(0.2, inplace=True),
# 		)
# 		self.conv_blocks2 = nn.Sequential(
# 			nn.Conv2d(128, 64, 3, stride=1, padding=1),
# 			nn.BatchNorm2d(64, 0.8),
# 			nn.LeakyReLU(0.2, inplace=True),
# 			nn.Conv2d(64, self.settings.channels, 3, stride=1, padding=1),
# 			nn.Tanh(),
# 			nn.BatchNorm2d(self.settings.channels, affine=False)
# 		)

# 	def forward(self, z, labels):
# 		gen_input = torch.mul(self.label_emb(labels), z)
# 		out = self.l1(gen_input)
# 		out = out.view(out.shape[0], 128, self.init_size, self.init_size)
# 		img = self.conv_blocks0(out)
# 		img = nn.functional.interpolate(img, scale_factor=2)
# 		img = self.conv_blocks1(img)
# 		img = nn.functional.interpolate(img, scale_factor=2)
# 		img = self.conv_blocks2(img)
# 		return img

# class Generator_imagenet(nn.Module):
# 	def __init__(self, options=None, conf_path=None):
# 		self.settings = options or Option(conf_path)

# 		super(Generator_imagenet, self).__init__()

# 		self.init_size = self.settings.img_size // 4
# 		self.l1 = nn.Sequential(nn.Linear(self.settings.latent_dim, 128 * self.init_size ** 2))

# 		self.conv_blocks0_0 = CategoricalConditionalBatchNorm2d(1000, 128)

# 		self.conv_blocks1_0 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
# 		self.conv_blocks1_1 = CategoricalConditionalBatchNorm2d(1000, 128, 0.8)
# 		self.conv_blocks1_2 = nn.LeakyReLU(0.2, inplace=True)

# 		self.conv_blocks2_0 = nn.Conv2d(128, 64, 3, stride=1, padding=1)
# 		self.conv_blocks2_1 = CategoricalConditionalBatchNorm2d(1000, 64, 0.8)
# 		self.conv_blocks2_2 = nn.LeakyReLU(0.2, inplace=True)
# 		self.conv_blocks2_3 = nn.Conv2d(64, self.settings.channels, 3, stride=1, padding=1)
# 		self.conv_blocks2_4 = nn.Tanh()
# 		self.conv_blocks2_5 = nn.BatchNorm2d(self.settings.channels, affine=False)

# 	def forward(self, z, labels):
# 		out = self.l1(z)
# 		out = out.view(out.shape[0], 128, self.init_size, self.init_size)
# 		img = self.conv_blocks0_0(out, labels)
# 		img = nn.functional.interpolate(img, scale_factor=2)
# 		img = self.conv_blocks1_0(img)
# 		img = self.conv_blocks1_1(img, labels)
# 		img = self.conv_blocks1_2(img)
# 		img = nn.functional.interpolate(img, scale_factor=2)
# 		img = self.conv_blocks2_0(img)
# 		img = self.conv_blocks2_1(img, labels)
# 		img = self.conv_blocks2_2(img)
# 		img = self.conv_blocks2_3(img)
# 		img = self.conv_blocks2_4(img)
# 		img = self.conv_blocks2_5(img)
# 		return img

# class direct_dataset(Dataset):
# 	def __init__(self, settings, logger, dataset):
# 		self.settings = settings
# 		self.logger = logger
# 		normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
# 										 std=[0.229, 0.224, 0.225])

# 		if dataset in ["cifar10", "cifar100"]:
# 			self.train_transform = transforms.Compose([
# 				transforms.RandomResizedCrop(size=32, scale=(0.5, 1.0)),
# 				transforms.RandomHorizontalFlip(),
# 			])
# 		elif dataset in ["tissuemnist_28"]:
# 			self.train_transform = transforms.Compose([
# 				transforms.RandomResizedCrop(size=28, scale=(0.5, 1.0)),
# 				transforms.RandomHorizontalFlip(),
# 			])
# 		else:
# 			self.train_transform = transforms.Compose([
# 				transforms.RandomResizedCrop(size=224, scale=(0.5, 1.0)),
# 				transforms.RandomHorizontalFlip(),
# 			])

# 		self.tmp_data = None
# 		self.tmp_label = None
# 		for i in range(1,5):
# 			# data!
# 			path = self.settings.generateDataPath +str(i)+".pickle"
# 			self.logger.info(path)
# 			with open(path, "rb") as fp:  # Pickling
# 				gaussian_data = pickle.load(fp)
# 			# import IPython
# 			# IPython.embed()
# 			if self.tmp_data is None:
# 				self.tmp_data = np.concatenate(gaussian_data, axis=0)
# 			else:
# 				self.tmp_data = np.concatenate((self.tmp_data, np.concatenate(gaussian_data, axis=0)))

# 			# label!info
# 			path = self.settings.generateLabelPath + str(i) + ".pickle"
# 			self.logger.info(path)
# 			with open(path, "rb") as fp:  # Pickling
# 				labels_list = pickle.load(fp)
# 			if self.tmp_label is None:
# 				self.tmp_label = np.concatenate(labels_list, axis=0)
# 			else:
# 				self.tmp_label = np.concatenate((self.tmp_label, np.concatenate(labels_list, axis=0)))

# 		assert len(self.tmp_label) == len(self.tmp_data)
# 		print(self.tmp_data.shape, self.tmp_label.shape)
# 		print('direct datset image number', len(self.tmp_label))


# 	def __getitem__(self, index):
# 		img = self.tmp_data[index]
# 		label = self.tmp_label[index]
# 		img = self.train_transform(torch.from_numpy(img))
# 		return img, label

# 	def __len__(self):
# 		return len(self.tmp_label)

class ExperimentDesign:
	def __init__(self, options=None, args=None, logger=None):
		self.settings = options
		self.args = args
		self.logger = logger

		self.train_loader = None
		self.test_loader = None
		self.model = None
		self.model_teacher = None
		self.optimizer_state = None
		self.trainer = None
		self.start_epoch = 0

		self.prepare()
	
	def set_logger(self):
		# logger = logging.getLogger('baseline')
		if dist.get_rank()==0:
			file_formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
			file_handler = logging.FileHandler(os.path.join(self.settings.save_path, "train_test.log"))
			file_handler.setFormatter(file_formatter)
			self.logger.addHandler(file_handler)
		self.logger.setLevel(logging.INFO if self.args.local_rank in [-1, 0] else logging.WARN)
		return self.logger

	def prepare(self):
		torch.cuda.set_device(self.args.local_rank)
		dist.init_process_group(backend='nccl')
		if dist.get_rank() == 0:
			self.settings.set_save_path()
			shutil.copyfile(self.args.conf_path, os.path.join(self.settings.save_path, os.path.basename(self.args.conf_path)))
			shutil.copyfile('./main_direct.py', os.path.join(self.settings.save_path, 'main_direct.py'))
			shutil.copyfile('./trainer_direct.py', os.path.join(self.settings.save_path, 'trainer_direct.py'))
		self.logger = self.set_logger()
		self.settings.paramscheck(self.logger)
		self._set_gpu()
		self._set_dataloader()
		self._set_model()
		self._replace()
		# self.logger.info(self.model_teacher)
		# self.logger.info(self.model)
		print(self.model)
		self._set_trainer()
	
	def _set_gpu(self):
		torch.manual_seed(self.settings.manualSeed)
		torch.cuda.manual_seed(self.settings.manualSeed)
		cudnn.benchmark = True

	def _set_dataloader(self):
		# create data loader
		# Get train_dataset and test_dataset from settings
		train_dataset = getattr(self.settings, 'train_dataset')
		test_dataset = getattr(self.settings, 'test_dataset')
		
		# Get train_dataPath and test_dataPath from settings
		train_data_path = getattr(self.settings, 'train_dataPath')
		test_data_path = getattr(self.settings, 'test_dataPath')
		
		data_loader = DataLoader(train_dataset=train_dataset,
		                         test_dataset=test_dataset,
		                         batch_size=self.settings.batchSize,
		                         train_data_path=train_data_path,
		                         test_data_path=test_data_path,
		                         n_threads=self.settings.nThreads,
		                         ten_crop=self.settings.tenCrop,
		                         logger=self.logger)
		
		self.train_loader, self.test_loader = data_loader.getloader()
		self.n_channels = data_loader.n_channels
		self.n_classes = data_loader.n_classes

	def _set_model(self):
		if self.settings.test_dataset in ["cifar100", "cifar10"]:
			self.model = ptcv_get_model(self.settings.model_name, pretrained=True)
			self.model_teacher = ptcv_get_model(self.settings.model_name, pretrained=True)
			self.model_teacher.eval()

		elif self.settings.test_dataset in ["imagenet"]:
			self.model = ptcv_get_model(self.settings.model_name, pretrained=True)
			self.model_teacher = ptcv_get_model(self.settings.model_name, pretrained=True)
			self.model_teacher.eval()
		elif self.settings.test_dataset in ["dermamnist_224"]:
			print("loading dermamnist_224 model")
			self.model = resnet18(pretrained=False, num_classes=self.n_classes)
			self.model.load_state_dict(torch.load("/home/suyoung/Vscode/HAST/models/checkpoints/weights_dermamnist/resnet18_224_1.pth")['net'], strict=True)
			self.model_teacher = resnet18(pretrained=False, num_classes=self.n_classes)
			self.model_teacher.load_state_dict(torch.load("/home/suyoung/Vscode/HAST/models/checkpoints/weights_dermamnist/resnet18_224_1.pth")['net'], strict=True)
			self.model_teacher.eval()
		elif self.settings.test_dataset in ["tissuemnist_224"]:
			self.model = resnet18(pretrained=False, num_classes=self.n_classes)
			# self.model = ResNet18(in_channels=self.n_channels, num_classes=self.n_classes)
			self.model.load_state_dict(torch.load("/home/suyoung/Vscode/HAST/models/checkpoints/weights_tissuemnist/resnet18_224_1.pth")['net'], strict=True)
			self.model_teacher = resnet18(pretrained=False, num_classes=self.n_classes)
			# self.model_teacher = ResNet18(in_channels=self.n_channels, num_classes=self.n_classes)
			self.model_teacher.load_state_dict(torch.load("/home/suyoung/Vscode/HAST/models/checkpoints/weights_tissuemnist/resnet18_224_1.pth")['net'], strict=True)
			self.model_teacher.eval()
		elif self.settings.test_dataset in ["dermamnist_28", "tissuemnist_28"]:
			# For 28x28 datasets, use a smaller model suitable for smaller images
			self.model = ptcv_get_model(self.settings.model_name, pretrained=True)
			self.model_teacher = ptcv_get_model(self.settings.model_name, pretrained=True)
			self.model_teacher.eval()
			# You might need to adjust the model architecture for different number of classes
			if hasattr(self.model, 'output') and hasattr(self.model.output, 'in_features'):
				self.model.output = nn.Linear(self.model.output.in_features, self.n_classes)
				self.model_teacher.output = nn.Linear(self.model_teacher.output.in_features, self.n_classes)
			elif hasattr(self.model, 'fc') and hasattr(self.model.fc, 'in_features'):
				self.model.fc = nn.Linear(self.model.fc.in_features, self.n_classes)
				self.model_teacher.fc = nn.Linear(self.model_teacher.fc.in_features, self.n_classes)
		else:
			assert False, "unsupport data set: " + self.settings.dataset
		
		self.model_teacher = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model_teacher)
		self.model_teacher = self.model_teacher.to(self.args.local_rank)

	def _set_trainer(self):
		# set lr master
		lr_master_S = utils.LRPolicy(self.settings.lr_S,
		                           self.settings.nEpochs,
		                           self.settings.lrPolicy_S)
		lr_master_G = utils.LRPolicy(self.settings.lr_G,
									 self.settings.nEpochs,
									 self.settings.lrPolicy_G)

		params_dict_S = {
			'step': self.settings.step_S,
			'decay_rate': self.settings.decayRate_S
		}

		params_dict_G = {
			'step': self.settings.step_G,
			'decay_rate': self.settings.decayRate_G
		}
		
		lr_master_S.set_params(params_dict=params_dict_S)
		lr_master_G.set_params(params_dict=params_dict_G)

		# set trainer
		self.trainer = Trainer(
			model=self.model,
			model_teacher=self.model_teacher,
			train_loader=self.train_loader,
			test_loader=self.test_loader,
			lr_master_S=lr_master_S,
			lr_master_G=lr_master_G,
			settings=self.settings,
			args = self.args,
			logger=self.logger,
			opt_type=self.settings.opt_type,
			optimizer_state=self.optimizer_state,
			run_count=self.start_epoch)

	def quantize_model(self,model):
		"""
		Recursively quantize a pretrained single-precision model to int8 quantized model
		model: pretrained single-precision model
		"""
		
		weight_bit = self.settings.qw
		act_bit = self.settings.qa
		# Print model structure for debugging
		# print("Original model structure:")
		# print(model)
		# quantize convolutional and linear layers
		if type(model) == nn.Conv2d:
			quant_mod = Quant_Conv2d(weight_bit=weight_bit)
			quant_mod.set_param(model)
			return quant_mod
		elif type(model) == nn.Linear:
			quant_mod = Quant_Linear(weight_bit=weight_bit)
			quant_mod.set_param(model)
			return quant_mod
		
		# quantize all the activation
		elif type(model) == nn.ReLU or type(model) == nn.ReLU6:
			return nn.Sequential(*[model, QuantAct(activation_bit=act_bit)])
		
		# recursively use the quantized module to replace the single-precision module
		elif type(model) == nn.Sequential:
			mods = []
			for n, m in model.named_children():
				mods.append(self.quantize_model(m))
			return nn.Sequential(*mods)
		else:
			q_model = copy.deepcopy(model)
			for attr in dir(model):
				mod = getattr(model, attr)
				if isinstance(mod, nn.Module) and 'norm' not in attr:
					setattr(q_model, attr, self.quantize_model(mod))
			return q_model
	
	def _replace(self):
		self.model = self.quantize_model(self.model)
		self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
		self.model = DDP(self.model.to(self.args.local_rank), device_ids=[self.args.local_rank], output_device=self.args.local_rank, broadcast_buffers=False, find_unused_parameters=True)
	
	def freeze_model(self,model):
		"""
		freeze the activation range
		"""
		if type(model) == QuantAct:
			model.fix()
		elif type(model) == nn.Sequential:
			for n, m in model.named_children():
				self.freeze_model(m)
		else:
			for attr in dir(model):
				mod = getattr(model, attr)
				if isinstance(mod, nn.Module) and 'norm' not in attr:
					self.freeze_model(mod)
			return model
	
	def unfreeze_model(self,model):
		"""
		unfreeze the activation range
		"""
		if type(model) == QuantAct:
			model.unfix()
		elif type(model) == nn.Sequential:
			for n, m in model.named_children():
				self.unfreeze_model(m)
		else:
			for attr in dir(model):
				mod = getattr(model, attr)
				if isinstance(mod, nn.Module) and 'norm' not in attr:
					self.unfreeze_model(mod)
			return model
	

	def run(self):
		best_top1 = 100
		best_top5 = 100
		self.best_epoch = 0
		self.start_epoch = 0
		start_time = time.time()

		# Initialize wandb for experiment tracking
		if dist.get_rank() == 0:  # Only initialize on main process
			wandb.init(
				project="HAST-quantization",
				config={
					"model": self.settings.model_name,
					"dataset": self.settings.test_dataset,
					"train_dataset": self.settings.train_dataset,
					"batch_size": self.settings.batchSize,
					"learning_rate": self.settings.lr_S,
					"epochs": self.settings.nEpochs,
					"weight_bits": self.settings.qw,
					"activation_bits": self.settings.qa,
					"temperature": self.settings.temperature,
					"alpha": self.settings.alpha,
				},
				name=f"{self.settings.test_dataset}_{self.settings.model_name}_qw{self.settings.qw}_qa{self.settings.qa}"
			)

		# dataset = direct_dataset(self.settings, self.logger, self.settings.dataset)

		# direct_dataload = torch.utils.data.DataLoader(dataset,
		# 											   batch_size=min(self.settings.batchSize, len(dataset)),
		# 											   sampler = DistributedSampler(dataset))

		try:
			for epoch in range(self.start_epoch, self.settings.nEpochs):
				self.epoch = epoch
				
				if epoch < 4:
					self.unfreeze_model(self.model)

				train_error, train_loss, train5_error = self.trainer.train(epoch=epoch, direct_dataload=self.train_loader)

				self.freeze_model(self.model)

				# Test with evaluator-based evaluation
				test_error, test_loss, test5_error = self.trainer.test(epoch=epoch)
				
				# Test teacher model performance
				teacher_test_error, teacher_test_loss, teacher_test5_error = self.trainer.test_teacher(epoch=epoch)
				
				# Calculate accuracy from test error
				test_acc = (100 - test_error)
				teacher_test_acc = (100 - teacher_test_error)
				
				self.logger.info(f"[Epoch {epoch + 1}] Teacher Test Results - Accuracy: {teacher_test_acc:.4f}%")
				self.logger.info(f"[Epoch {epoch + 1}] Student Test Results - Accuracy: {test_acc:.4f}%")
				
				print(f"[Epoch {epoch + 1}] Teacher Test - Accuracy: {teacher_test_acc:.4f}%")
				print(f"[Epoch {epoch + 1}] Student Test - Accuracy: {test_acc:.4f}%")
				
				# Log metrics to wandb (only on main process)
				if dist.get_rank() == 0:
					wandb.log({
						"epoch": epoch + 1,
						"train/loss": train_loss,
						"train/error": train_error,
						"train/accuracy": 100 - train_error,
						"test/loss": test_loss,
						"test/error": test_error,
						"test/accuracy": test_acc,
						"teacher/test_loss": teacher_test_loss,
						"teacher/test_error": teacher_test_error,
						"teacher/test_accuracy": teacher_test_acc,
						"best/accuracy": 100 - best_top1,
						"best/epoch": self.best_epoch,
					})

				if best_top1 >= test_error:
					best_top1 = test_error
					best_top5 = test5_error
					self.best_epoch = epoch + 1
					# self.logger.info(
					# 	'Save model! The path is ' + os.path.join(self.settings.save_path, "model.pth"))
					# if dist.get_rank() == 0:
					# 	torch.save(self.model.state_dict(), os.path.join(self.settings.save_path, "model.pth"))
				
				self.logger.info("#==>Best Result is: Top1 Accuracy: {:f}, Top5 Accuracy: {:f}".format(100 - best_top1,
																									   100 - best_top5))
				print("#==>Best Result is: Top1 Accuracy: {:f}, Top5 Accuracy: {:f}".format(100 - best_top1,
																							  100 - best_top5))
				print("Best epoch: ", self.best_epoch)

		except BaseException as e:
			self.logger.error("Training is terminating due to exception: {}".format(str(e)))
			traceback.print_exc()
		
		end_time = time.time()
		time_interval = end_time - start_time
		t_string = "Running Time is: " + str(datetime.timedelta(seconds=time_interval)) + "\n"
		self.logger.info(t_string)

		# Log final results to wandb
		if dist.get_rank() == 0:
			wandb.log({
				"final/best_accuracy": 100 - best_top1,
				"final/best_epoch": self.best_epoch,
				"final/training_time": time_interval,
			})
			wandb.finish()

		return best_top1, best_top5


def main():
	logger = logging.getLogger()
	parser = argparse.ArgumentParser(description='Baseline')
	parser.add_argument('--conf_path', type=str, metavar='conf_path',
	                    help='input the path of config file')
	parser.add_argument("--local_rank", type=int, default=0)
	args = parser.parse_args()

	option = Option(args.conf_path)
	option.manualSeed = 1

	experiment = ExperimentDesign(option, args, logger)
	experiment.run()


if __name__ == '__main__':
	main()
