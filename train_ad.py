
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"


import copy
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
#from data_loader import get_cifar
from data_loader_custom import get_cifar, get_PAMAP2_data, get_PAMAP2_data3, get_PAMAP2_data4
from model_factory import create_cnn_model, is_resnet

from sklearn.metrics import classification_report, confusion_matrix

import time
import datetime
from torchsummary import summary


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def str2bool(v):
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	else:
		return False
	
	
def parse_arguments():
	parser = argparse.ArgumentParser(description='TA Knowledge Distillation Code')
	parser.add_argument('--epochs', default=200, type=int,  help='number of total epochs to run')
	parser.add_argument('--dataset', default='cifar100', type=str, help='dataset. can be either cifar10 or cifar100')
	parser.add_argument('--batch_size', default=128, type=int, help='batch_size')
	parser.add_argument('--learning-rate', default=0.1, type=float, help='initial learning rate')
	parser.add_argument('--momentum', default=0.9, type=float,  help='SGD momentum')
	parser.add_argument('--weight-decay', default=1e-4, type=float, help='SGD weight decay (default: 1e-4)')
	parser.add_argument('--teacher', default='', type=str, help='teacher student name')
	parser.add_argument('--student', '--model', default='resnet8', type=str, help='teacher student name')
	parser.add_argument('--student-checkpoint', default='', type=str, help='optinal pretrained checkpoint for student')
	parser.add_argument('--teacher-checkpoint', default='', type=str, help='optinal pretrained checkpoint for teacher')
	parser.add_argument('--teacher2', default='', type=str, help='teacher student name')
	parser.add_argument('--teacher-checkpoint2', default='', type=str, help='optinal pretrained checkpoint for teacher')
	parser.add_argument('--cuda', default=False, type=str2bool, help='whether or not use cuda(train on GPU)')
	parser.add_argument('--dataset-dir', default='./data', type=str,  help='dataset directory')
	parser.add_argument('--trial', default=0, type=str,  help='trial memo number')
	parser.add_argument('--sbj', default=0, type=int,  help='sbj number')
	parser.add_argument('--seed', default=1234, type=int,  help='seed number')
	parser.add_argument('--save_weight', default=0, type=int,  help='save_default:0 save_flag:1')
	args = parser.parse_args()
	return args


def load_checkpoint(model, checkpoint_path):
	"""
	Loads weights from checkpoint
	:param model: a pytorch nn student
	:param str checkpoint_path: address/path of a file
	:return: pytorch nn student with weights loaded from checkpoint
	"""
	model_ckp = torch.load(checkpoint_path)
	model.load_state_dict(model_ckp['model_state_dict'])
	return model



class TrainManager(object):
	def __init__(self, student, teacher=None, teacher2=None, train_loader=None, test_loader=None, train_loader1=None, test_loader1=None, train_config={}):
		self.student = student
		self.teacher = teacher
		self.teacher2 = teacher2
		self.have_teacher = bool(self.teacher)
		self.device = train_config['device']
		self.name = train_config['name']
		self.optimizer = optim.SGD(self.student.parameters(),
								   lr=train_config['learning_rate'],
								   momentum=train_config['momentum'],
								   weight_decay=train_config['weight_decay'])
		if self.have_teacher:
			self.teacher.eval()
			self.teacher.train(mode=False)
			self.teacher2.eval()
			self.teacher2.train(mode=False)
			
		self.train_loader = train_loader #signal
		self.test_loader = test_loader

		self.train_loader1 = train_loader1 #image
		self.test_loader1 = test_loader1

		self.config = train_config
	
	def train(self):
		lambda_ = 0.99
		T = 4
		epochs = self.config['epochs']
		trial_id = self.config['trial_id']
		
		max_val_acc = 0
		iteration = 0
		best_acc = 0
		criterion = nn.CrossEntropyLoss()
		save_flag = args.save_weight

		for epoch in range(epochs):
			start_time = time.time()
			
			self.student.train()
			lr = self.adjust_learning_rate(self.optimizer, epoch) #loss plan
			loss = 0

			CE_loss = 0.0
			KD_loss = 0.0
			T_loss = 0.0
			count_iter = 0

			CE_loss1 = 0.0
			KD_loss1 = 0.0
			T_loss1 = 0.0

			CE_loss2 = 0.0
			KD_loss2 = 0.0
			T_loss2 = 0.0

			c_iter = 0.0

			for batch_idx, (data, signals, target) in enumerate(self.train_loader):
				iteration += 1
				data = data.to(self.device)
				signals = signals.to(self.device)
				target = target.to(self.device)
				self.optimizer.zero_grad()
				output = self.student(signals)
				
                                # Standard Learning Loss ( Classification Loss)
				stds = torch.std(output, dim=-1, keepdim=True)
				loss_SL = criterion(output / stds, target) #softmax entropy
				loss = loss_SL
				
				if self.have_teacher:
					teacher_outputs2 = self.teacher2(signals)
					teacher_outputs = self.teacher(data)

					stdt = torch.std(teacher_outputs, dim=-1, keepdim=True)
					loss_KD = nn.KLDivLoss()(F.log_softmax(output * 2.0/ stds, dim=1), F.softmax(teacher_outputs * 2.0/ stdt, dim=1))


					stdt2 = torch.std(teacher_outputs2, dim=-1, keepdim=True)
					loss_KD2 = nn.KLDivLoss()(F.log_softmax(output * 2.0/ stds, dim=1), F.softmax(teacher_outputs2 * 2.0/ stdt2, dim=1))

					KD_loss1 += loss_KD * target.size(0)
					KD_loss2 += loss_KD2 * target.size(0)
					loss = (1 - lambda_) * loss_SL + 0.7 * lambda_ * T * T * loss_KD + 0.3 * lambda_ * T * T * loss_KD2

				loss.backward()
				self.optimizer.step()

				T_loss1 += loss*target.size(0)
				CE_loss1 += loss_SL*target.size(0)


				count_iter += target.size(0)
				c_iter += 1

			end_time = time.time()
			epoch_mins, epoch_secs = epoch_time(start_time, end_time)
			current_time = datetime.datetime.now()
			print(f'current_time: {current_time}')
			best_buf = "%.4f" % (best_acc)
			if self.have_teacher:
				ls = T_loss1 / count_iter
				l_KD = KD_loss1 / count_iter
				l_CE = CE_loss1 / count_iter
				l_KD2 = KD_loss2 / count_iter
				print(f'"epoch {epoch}/{epochs} | Epoch Time: {epoch_mins}m {epoch_secs}s | lr: {lr:.7f} | ' \
          			f'loss {ls:.9f} loss_CE {l_CE:.9f} loss_KD {l_KD:.9f} loss_KD2 {l_KD2:.9f} | best_acc {best_buf}')

			else:
				ls = T_loss1 / count_iter
				print(f'"epoch {epoch}/{epochs} | Epoch Time: {epoch_mins}m {epoch_secs}s | lr: {lr:.7f} | ' \
          			f'loss {ls:.9f} | best_acc {best_buf}')

			val_acc = self.validate(step=epoch)
			if val_acc > best_acc:
				best_acc = val_acc
				buf = "%.4f" % (val_acc)
				if epoch >= 0 and save_flag > 0:
					self.save(epoch, name='./test_model/{}_{}_ep{}_val{}_best.pth.tar'.format(self.name, trial_id, epoch, buf))
			if epoch % 50 == 0 or epoch == 20: 
				buf = "%.4f" % (val_acc)
				if epoch >= 30 and save_flag > 0:
					self.save(epoch, name='./test_model/{}_{}_ep{}_val{}_current.pth.tar'.format(self.name, trial_id, epoch, buf))
			if epoch == epochs - 1 and save_flag > 0:
				buf = "%.4f" % (val_acc)
				self.save(epoch, name='./test_model/{}_{}_ep{}_val{}_final.pth.tar'.format(self.name, trial_id, epoch, buf))
		
		return best_acc
	
	def validate(self, step=0):
		self.student.eval()
		with torch.no_grad():
			correct = 0
			total = 0
			acc = 0
			T = 4
			loss_KD = 0.0

			KD_loss = 0.0
			loss_SL = 0.0
			loss_ = 0.0

			KD_loss2 = 0.0

			for batch_idx1, (images, signals, labels) in enumerate(self.test_loader):
				images = images.to(self.device)
				signals = signals.to(self.device)
				labels = labels.to(self.device)

				outputs = self.student(signals)

				loss_SL = nn.CrossEntropyLoss()(outputs, labels) #softmax entropy
				loss_ += loss_SL* labels.size(0)

				if self.have_teacher:
					teacher_outputs = self.teacher(images)
					teacher_outputs2 = self.teacher2(signals)

					loss_KD = nn.KLDivLoss()(F.log_softmax(outputs / T, dim=1), F.softmax(teacher_outputs / T, dim=1))
					loss_KD2 = nn.KLDivLoss()(F.log_softmax(outputs / T, dim=1), F.softmax(teacher_outputs2 / T, dim=1))
					KD_loss += loss_KD * labels.size(0)
					KD_loss2 += loss_KD2 * labels.size(0)

				_, predicted = torch.max(outputs.data, 1)
				total += labels.size(0)
				correct += (predicted == labels).sum().item()

			acc = 100 * correct / total

			buf = "%.4f" % (acc)

			if self.have_teacher:
				KD_L = KD_loss / total
				KD_L2 = KD_loss2 / total
				SF_L = loss_ / total			
				print(f'( "metric": "{self.name}_val_accuracy", "value": {buf}, "SF_L": {SF_L:.9f}, "KD_L": {KD_L:.9f}, "KD_L2": {KD_L2:.9f} )')
			else:
				SF_L = loss_ / total
				print(f'( "metric": "{self.name}_val_accuracy", "value": {buf}, "SF_L": {SF_L:.9f})')
			return acc
	
	def save(self, epoch, name=None):
		trial_id = self.config['trial_id']
		if name is None:
			torch.save({
				'epoch': epoch,
				'model_state_dict': self.student.state_dict(),
				'optimizer_state_dict': self.optimizer.state_dict(),
			}, '{}_{}_epoch{}.pth.tar'.format(self.name, trial_id, epoch))
		else:
			torch.save({
				'model_state_dict': self.student.state_dict(),
				'optimizer_state_dict': self.optimizer.state_dict(),
				'epoch': epoch,
			}, name)
	
	def adjust_learning_rate(self, optimizer, epoch):
		epochs = self.config['epochs']
		models_are_plane = self.config['is_plane']

		same_lr = 0

		if same_lr:
			lr = 0.01
		else:

			if epoch < int(epochs/18.0):
				lr = 0.05
			elif epoch < int(epochs/3.0):
				lr = 0.1 * 0.1
			elif epoch < int(epochs*2/3.0):
				lr = 0.1 * 0.01
			else:
				lr = 0.1 * 0.001

		

		
		# update optimizer's learning rate
		for param_group in optimizer.param_groups:
			param_group['lr'] = lr
		return lr

	def adjust_learning_rate2(self, optimizer, epoch):
		epochs = self.config['epochs']
		models_are_plane = self.config['is_plane']
		
		# depending on dataset
		if models_are_plane:
			lr = 0.01
		else:
			if epoch < int(epoch/2.0):
				lr = 0.1
			elif epoch < int(epochs*3/4.0):
				lr = 0.1 * 0.1
			else:
				lr = 0.1 * 0.01

		
		# update optimizer's learning rate
		for param_group in optimizer.param_groups:
			param_group['lr'] = lr
		return lr



def get_all_preds(model, loader):
    all_preds = torch.tensor([])
    for batch in loader:
        images, labels = batch

        preds = model(images)
        all_preds = torch.cat(
            (all_preds, preds)
            ,dim=0
        )
    return all_preds


if __name__ == "__main__":
	# Parsing arguments and prepare settings for training
	args = parse_arguments()
	print(args)

	SEED = args.seed
	torch.manual_seed(SEED)
	torch.cuda.manual_seed(SEED)
	trial_id = args.trial
	dataset = args.dataset

	if dataset == 'pamap':
		num_classes = 12
		test_id = args.sbj
	else:
		num_classes = 10

	teacher_model = None
	student_model = create_cnn_model(args.student, dataset, use_cuda=args.cuda, num_cls=num_classes)

	if args.student_checkpoint:
		print("---------- Loading Student -------")
		student_model = load_checkpoint(student_model, args.student_checkpoint)

	train_config = {
		'epochs': args.epochs,
		'learning_rate': args.learning_rate,
		'momentum': args.momentum,
		'weight_decay': args.weight_decay,
		'device': 'cuda' if args.cuda else 'cpu',
		'is_plane': not is_resnet(args.student),
		'trial_id': trial_id,
	}
	
	
	# Train Teacher if provided a teacher, otherwise it's a normal training using only cross entropy loss
	# This is for training single models(NOKD in paper) for baselines models (or training the first teacher)
	if args.teacher:
		teacher_model = create_cnn_model(args.teacher, dataset, use_cuda=args.cuda, num_cls=num_classes)
		teacher_model2 = create_cnn_model(args.teacher2, dataset, use_cuda=args.cuda, num_cls=num_classes)
		if args.teacher_checkpoint:
			print("---------- Loading Teacher -------")
			teacher_model = load_checkpoint(teacher_model, args.teacher_checkpoint)
			teacher_model2 = load_checkpoint(teacher_model2, args.teacher_checkpoint2)
		else:
			print("---------- Training Teacher -------")
			train_loader, test_loader = get_PAMAP2_data3(test_id=test_id, batch_size=args.batch_size)
			teacher_train_config = copy.deepcopy(train_config)
			teacher_name = 'teacher_{}_{}_best.pth.tar'.format(args.teacher, trial_id)
			teacher_train_config['name'] = args.teacher
			teacher_trainer = TrainManager(teacher_model, teacher=None, train_loader=train_loader, test_loader=test_loader, train_config=teacher_train_config)
			teacher_trainer.train()
			teacher_model = load_checkpoint(teacher_model, os.path.join('./', teacher_name))
			
	# Student training
	print("---------- Training Student -------")
	student_train_config = copy.deepcopy(train_config)
	train_loader, test_loader = get_PAMAP2_data4(test_id=test_id, batch_size=args.batch_size) #signal+image load
	#train_loader2, test_loader2 = get_PAMAP2_data(test_id,SEED) #signal load

	images, signals, labels = next(iter(train_loader))
	print('input shape: ', signals[0].shape, labels.shape)
	summary(student_model, signals[0].shape)


	student_train_config['name'] = args.student
	student_trainer = TrainManager(student_model, teacher=teacher_model, teacher2=teacher_model2, train_loader=train_loader, test_loader=test_loader, train_config=student_train_config)
	best_student_acc = student_trainer.train()

