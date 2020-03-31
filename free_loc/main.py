import argparse
import os
import shutil
import time
import sys
sys.path.insert(0, 'faster_rcnn')
import sklearn
import sklearn.metrics

import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from datasets.factory import get_imdb
from custom import *

model_names = sorted(name for name in models.__dict__
					 if name.islower() and not name.startswith("__")
					 and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--arch', default='localizer_alexnet')
parser.add_argument(
	'-j',
	'--workers',
	default=4,
	type=int,
	metavar='N',
	help='number of data loading workers (default: 4)')
parser.add_argument(
	'--epochs',
	default=30,
	type=int,
	metavar='N',
	help='number of total epochs to run')
parser.add_argument(
	'--start-epoch',
	default=0,
	type=int,
	metavar='N',
	help='manual epoch number (useful on restarts)')
parser.add_argument(
	'-b',
	'--batch-size',
	default=256,
	type=int,
	metavar='N',
	help='mini-batch size (default: 256)')
parser.add_argument(
	'--lr',
	'--learning-rate',
	default=0.1,
	type=float,
	metavar='LR',
	help='initial learning rate')
parser.add_argument(
	'--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument(
	'--weight-decay',
	'--wd',
	default=1e-4,
	type=float,
	metavar='W',
	help='weight decay (default: 1e-4)')
parser.add_argument(
	'--print-freq',
	'-p',
	default=10,
	type=int,
	metavar='N',
	help='print frequency (default: 10)')
parser.add_argument(
	'--eval-freq',
	default=10,
	type=int,
	metavar='N',
	help='print frequency (default: 10)')
parser.add_argument(
	'--resume',
	default='',
	type=str,
	metavar='PATH',
	help='path to latest checkpoint (default: none)')
parser.add_argument(
	'-e',
	'--evaluate',
	dest='evaluate',
	action='store_true',
	help='evaluate model on validation set')
parser.add_argument(
	'--pretrained',
	dest='pretrained',
	action='store_true',
	help='use pre-trained model')
parser.add_argument(
	'--world-size',
	default=1,
	type=int,
	help='number of distributed processes')
parser.add_argument(
	'--dist-url',
	default='tcp://224.66.41.62:23456',
	type=str,
	help='url used to set up distributed training')
parser.add_argument(
	'--dist-backend', default='gloo', type=str, help='distributed backend')
parser.add_argument('--vis', action='store_true')

args = None
best_prec1 = 0
global_step = 0
idx_to_class = {}

# Taken from - https://github.com/pytorch/pytorch/issues/5059
# def worker_init_fn(worker_id):                                                          
# 	np.random.seed(np.random.get_state()[1][0] + worker_id)

# Set random seed
SEED = 42
np.random.seed(42)
torch.manual_seed(42)

def main():
	global args, best_prec1, global_step, idx_to_class
	args = parser.parse_args()
	args.distributed = args.world_size > 1

	# create model
	print("=> creating model '{}'".format(args.arch))
	if args.arch == 'localizer_alexnet':
		model = localizer_alexnet(pretrained=args.pretrained)
	elif args.arch == 'localizer_alexnet_robust':
		model = localizer_alexnet_robust(pretrained=args.pretrained)
	print(model)

	model.features = torch.nn.DataParallel(model.features)
	model.cuda()

	# TODO:
	# define loss function (criterion) and optimizer
	criterion = nn.BCEWithLogitsLoss()
	# criterion = nn.BCELoss()
	optimizer = torch.optim.SGD(
								model.parameters(), 
								lr=args.lr, 
								momentum=args.momentum, 
								weight_decay=args.weight_decay
							)
	# NOTE:Done

	# optionally resume from a checkpoint
	if args.resume:
		if os.path.isfile(args.resume):
			print("=> loading checkpoint '{}'".format(args.resume))
			checkpoint = torch.load(args.resume)
			args.start_epoch = checkpoint['epoch']
			best_prec1 = checkpoint['best_prec1']
			model.load_state_dict(checkpoint['state_dict'])
			optimizer.load_state_dict(checkpoint['optimizer'])
			print("=> loaded checkpoint '{}' (epoch {})".format(
				args.resume, checkpoint['epoch']))
		else:
			print("=> no checkpoint found at '{}'".format(args.resume))

	cudnn.benchmark = True

	# Data loading code
	# TODO: Write code for IMDBDataset in custom.py
	trainval_imdb = get_imdb('voc_2007_trainval')
	# NOTE:Done
	class_to_idx = trainval_imdb._class_to_ind
	idx_to_class = {v:k for k,v in class_to_idx.items()}

	test_imdb = get_imdb('voc_2007_test')

	normalize = transforms.Normalize(
		mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	train_dataset = IMDBDataset(
		trainval_imdb,
		transforms.Compose([
			transforms.Resize((512, 512)),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			normalize,
		]))
	train_sampler = None
	train_loader = torch.utils.data.DataLoader(
		train_dataset,
		batch_size=args.batch_size,
		shuffle=(train_sampler is None),
		num_workers=args.workers,
		pin_memory=True,
		sampler=train_sampler,
		# worker_init_fn=worker_init_fn # Add to have same outputs
		)

	val_loader = torch.utils.data.DataLoader(
		IMDBDataset(
			test_imdb,
			transforms.Compose([
				transforms.Resize((384, 384)),
				transforms.ToTensor(),
				normalize,
			])),
		batch_size=args.batch_size,
		shuffle=False,
		num_workers=args.workers,
		pin_memory=True)

	if args.evaluate:
		validate(val_loader, model, criterion, 0, loggers=None)
		return

	# TODO: Create loggers for visdom and tboard
	# TODO: You can pass the logger objects to train(), make appropriate
	# modifications to train()
	if args.vis:
		from tensorboardX import SummaryWriter
		import visdom
		tb_logger = SummaryWriter(logdir='lr_{}_b_{}'.format(args.lr, args.batch_size))
		vis_logger = visdom.Visdom(server='0.0.0.0',port='8080')
		loggers = (tb_logger, vis_logger)
		#NOTE:Done
		#NOTE:Done
	else:
		loggers = None

	for epoch in range(args.start_epoch, args.epochs):
		adjust_learning_rate(optimizer, epoch)

		# train for one epoch
		train(train_loader, model, criterion, optimizer, epoch, loggers=loggers)

		# evaluate on validation set
		if epoch % args.eval_freq == 0 or epoch == args.epochs - 1:
			m1, m2 = validate(val_loader, model, criterion, epoch, loggers=loggers)
			score = m1 * m2
			# remember best prec@1 and save checkpoint
			is_best = score > best_prec1
			best_prec1 = max(score, best_prec1)
			save_checkpoint({
				'epoch': epoch + 1,
				'arch': args.arch,
				'state_dict': model.state_dict(),
				'best_prec1': best_prec1,
				'optimizer': optimizer.state_dict(),
			}, is_best)
			print("Saved Checkpoint")


def process_image(img):
	mean = np.array([0.485, 0.456, 0.406])
	std = np.array([0.229, 0.224, 0.225])
	img = img*std[:, None, None] + mean[:, None, None]
	return img

#TODO: You can add input arguments if you wish
#NOTE:Done
def train(train_loader, model, criterion, optimizer, epoch, loggers=None):
	global global_step, idx_to_class

	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses = AverageMeter()
	avg_m1 = AverageMeter()
	avg_m2 = AverageMeter()

	# switch to train mode
	model.train()

	end = time.time()
	for i, (input, target) in enumerate(train_loader):
		# measure data loading time
		data_time.update(time.time() - end)
		
		target = target.type(torch.FloatTensor).cuda(async=True)
		input_var = input
		target_var = target

		# TODO: Get output from model
		# TODO: Perform any necessary functions on the output
		# TODO: Compute loss using ``criterion``
		score_logits = model(input_var)
		pool_layer = torch.nn.functional.adaptive_max_pool2d(score_logits, output_size=(1,1))
		imoutput = torch.squeeze(pool_layer)
		loss = criterion(imoutput, target_var)
		#NOTE:Done
		#NOTE:Done
		#NOTE:Done

		# measure metrics and record loss
		m1 = metric1(imoutput.data, target)
		m2 = metric2(imoutput.data, target)
		losses.update(loss.item(), input.size(0))
		avg_m1.update(m1[0], input.size(0))
		avg_m2.update(m2[0], input.size(0))

		# TODO:
		# compute gradient and do SGD step
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		#NOTE:Done

		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()

		if i % args.print_freq == 0:
			print('Epoch: [{0}][{1}/{2}]\t'
				  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
				  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
				  'Metric1 {avg_m1.val:.3f} ({avg_m1.avg:.3f})\t'
				  'Metric2 {avg_m2.val:.3f} ({avg_m2.avg:.3f})'.format(
					  epoch,
					  i,
					  len(train_loader),
					  batch_time=batch_time,
					  data_time=data_time,
					  loss=losses,
					  avg_m1=avg_m1,
					  avg_m2=avg_m2))

		#TODO: Visualize things as mentioned in handout
		#TODO: Visualize at appropriate intervals
		if args.vis:
			assert loggers is not None
			tb_logger, vis_logger = loggers
			# Plot the training loss
			tb_logger.add_scalar('train/loss', loss.item(), global_step)
			tb_logger.add_scalar('train/mAP', m1[0], global_step)
			tb_logger.add_scalar('train/mean_recall', m2[0], global_step)
			# Plot the heatmaps and the images
			if i == 0 or i == len(train_loader)//2:
				# Get the image 
				img = process_image(input[0].cpu().numpy())
				gt_classes = np.where(target[0].cpu().numpy() == 1)[0]
				# import traceback as tb; import code; tb.print_stack(); namespace = globals().copy();namespace.update(locals());code.interact(local=namespace)
				gt_heatmaps = torch.sigmoid(score_logits).detach().cpu().numpy()[0, gt_classes][:,None,:,:]
				# Tensorboard log
				img_step = 0 if i == 0 else 1
				tb_logger.add_image('train/images_{}'.format(img_step), img, global_step=epoch, dataformats='CHW')
				tb_logger.add_images('train/heatmaps_{}'.format(img_step), gt_heatmaps, global_step=epoch, dataformats='NCHW')
				# Visdom logging
				img_tag = 'train_imgs_epoch_{}_iter_{}_batchindex_{}'.format(epoch, global_step, i)
				heatmap_base_tag = 'train_heatmaps_epoch_{}_iter_{}_batchindex_{}'.format(epoch, global_step, i)
				vis_logger.image(np.array(img*255.0, dtype=np.uint8), opts={'title':img_tag})
				for idx, heatmap in enumerate(gt_heatmaps):
					heatmap_tag = heatmap_base_tag + idx_to_class[gt_classes[idx]]
					vis_logger.heatmap(heatmap.squeeze(), opts={'title':heatmap_tag})
		global_step += 1
		#NOTE:Done
		#NOTE:Done
		# End of train()


def validate(val_loader, model, criterion, epoch, loggers=None):
	global global_step, idx_to_class

	batch_time = AverageMeter()
	losses = AverageMeter()
	avg_m1 = AverageMeter()
	avg_m2 = AverageMeter()

	# switch to evaluate mode
	model.eval()

	end = time.time()
	for i, (input, target) in enumerate(val_loader):
		target = target.type(torch.FloatTensor).cuda(async=True)
		input_var = input
		target_var = target

		# TODO: Get output from model
		# TODO: Perform any necessary functions on the output
		# TODO: Compute loss using ``criterion``
		score_logits = model(input_var)
		pool_layer = torch.nn.functional.adaptive_max_pool2d(score_logits, output_size=1)
		imoutput = torch.squeeze(pool_layer)
		loss = criterion(imoutput, target_var)
		#NOTE:Done
		#NOTE:Done
		#NOTE:Done


		# measure metrics and record loss
		m1 = metric1(imoutput.data, target)
		m2 = metric2(imoutput.data, target)
		losses.update(loss.item(), input.size(0))
		avg_m1.update(m1[0], input.size(0))
		avg_m2.update(m2[0], input.size(0))

		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()

		if i % args.print_freq == 0:
			print('Test: [{0}/{1}]\t'
				  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
				  'Metric1 {avg_m1.val:.3f} ({avg_m1.avg:.3f})\t'
				  'Metric2 {avg_m2.val:.3f} ({avg_m2.avg:.3f})'.format(
					  i,
					  len(val_loader),
					  batch_time=batch_time,
					  loss=losses,
					  avg_m1=avg_m1,
					  avg_m2=avg_m2))

		#TODO: Visualize things as mentioned in handout
		#TODO: Visualize at appropriate intervals
		if args.vis:
			assert loggers is not None
			tb_logger, vis_logger = loggers
			# Plot the training loss
			tb_logger.add_scalar('val/loss', loss.item(), global_step)
			# Plot the heatmaps and the images
			if i == 0 or i == len(val_loader)//2:
				# Get the image 
				img = process_image(input[0].cpu().numpy())
				gt_classes = np.where(target[0].cpu().numpy() == 1)[0]
				# import traceback as tb; import code; tb.print_stack(); namespace = globals().copy();namespace.update(locals());code.interact(local=namespace)
				gt_heatmaps = torch.sigmoid(score_logits).detach().cpu().numpy()[0, gt_classes][:,None,:,:]
				# Tensorboard log
				img_step = 0 if i == 0 else 1
				tb_logger.add_image('val/images_{}'.format(img_step), img, global_step=epoch, dataformats='CHW')
				tb_logger.add_images('val/heatmaps_{}'.format(img_step), gt_heatmaps, global_step=epoch, dataformats='NCHW')
				# Visdom log
				img_tag = 'val_imgs_epoch_{}_iter_{}_batchindex_{}'.format(epoch, global_step, i)
				heatmap_base_tag = 'val_heatmaps_epoch_{}_iter_{}_batchindex_{}'.format(epoch, global_step, i)
				vis_logger.image(np.array(img*255.0, dtype=np.uint8), opts={'title':img_tag})
				for idx, heatmap in enumerate(gt_heatmaps):
					heatmap_tag = heatmap_base_tag + idx_to_class[gt_classes[idx]]
					vis_logger.heatmap(heatmap.squeeze(), opts={'title':heatmap_tag})
		#NOTE:Done
		#NOTE:Done
	if args.vis:
		tb_logger.add_scalar('val/mAP', avg_m1.avg(), epoch)
		tb_logger.add_scalar('val/mean_recall', avg_m2.avg(), epoch)
	print(' * Metric1 {avg_m1.avg:.3f} Metric2 {avg_m2.avg:.3f}'.format(
		avg_m1=avg_m1, avg_m2=avg_m2))

	return avg_m1.avg, avg_m2.avg


# TODO: You can make changes to this function if you wish (not necessary)
# NOTE:Done
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
	dir_name = 'lr_{}_b_{}'.format(args.lr, args.batch_size)
	torch.save(state, filename)
	if is_best:
		shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
	"""Computes and stores the average and current value"""

	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
	"""Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
	lr = args.lr * (0.1**(epoch // 30))
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr


def metric1(output, target):
	# TODO: Ignore for now - proceed till instructed
	prob_scores = torch.sigmoid(output)
	# Filter the columns that don't have any true instance
	cls_counts = torch.sum(target, dim=0)
	valid_classes = cls_counts > 0
	filtered_scores = prob_scores[:,valid_classes]
	filtered_targets = target[:,valid_classes]
	mAP = sklearn.metrics.average_precision_score(filtered_targets, filtered_scores)
	#NOTE:Done

	# ---------------------------- mAP keeping classes with no instance as 0/1 ----------------------------
	# prob_scores = torch.sigmoid(output)
	# APs = np.zeros(output.shape[1])	
	# # import traceback as tb; import code; tb.print_stack(); namespace = globals().copy();namespace.update(locals());code.interact(local=namespace)
	# for idx, cls_index in enumerate(range(output.shape[1])):
	# 	cls_outputs, cls_targets = prob_scores[:,cls_index], target[:,cls_index]
	# 	if torch.sum(cls_targets) == 0:
	# 		APs[idx] = 0
	# 	else:
	# 		APs[idx] = sklearn.metrics.average_precision_score(cls_targets, cls_outputs, average=None)

	# 	# if np.isnan(APs[idx]):
	# 	# 	import traceback as tb; import code; tb.print_stack(); namespace = globals().copy();namespace.update(locals());code.interact(local=namespace)
	# mAP = np.mean(APs)
	# ------------------------------------------------------------------------------------------------------
	return [mAP]


def metric2(output, target):
	#TODO: Ignore for now - proceed till instructed
	THRESH = 0.3
	prob_scores = torch.sigmoid(output)
	# Filter the columns that don't have any true instance
	cls_counts = torch.sum(target, dim=0)
	valid_classes = cls_counts > 0
	filtered_scores = prob_scores[:,valid_classes]
	filtered_targets = target[:,valid_classes]
	preds = np.zeros(prob_scores.shape, dtype=np.int)
	preds[prob_scores >= THRESH] = 1
	recalls = sklearn.metrics.recall_score(target, preds)
	mRecall = np.mean(recalls)
	#NOTE:Done
	return [mRecall]


if __name__ == '__main__':
	main()
