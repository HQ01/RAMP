from __future__ import print_function
from argparse import ArgumentParser
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets,transforms
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import myio as myio
from network import LeNet
import utilities as utilities
import time
import cPickle as pickle
import os
#train function

def get_data_loader(args):
    print('getting dataset...{}.....\n'.format(args.dataset))
    train_loader,test_loader = utilities.get_dataset(args.dataset)(args)
    return train_loader,test_loader

def get_criterion(args,criterion = nn.CrossEntropyLoss()):
	if args.cuda:
		criterion.cuda()
	return criterion

def get_optimizer(args,model,optim_type = 'SGD'):
    if optim_type == 'SGD':
        optimizer = optim.SGD(
                params = model.parameters(),
                lr = args.lr,
                momentum = args.momentum,
                weight_decay = args.weight_decay
                )
    return optimizer
def get_model(args):
    print('getting models.....{}.......\n'.format(args.model))
    if args.resume:
        print('\n+++++++++++++++Resume training from checkpoint+++++++++++\n')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found'
        checkpoint = torch.load('./checkpoint/ckpt.pth')
        model = checkpoint['model']
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
    else:
        print('\n++++++++++++building model++++++++++++++\n')
        model = utilities.get_net(args.model)
        best_acc = 0
        start_epoch = 1
    if args.cuda:
    	print('CHANGE MODEL TO CUDA...')
        model.cuda()
        model = torch.nn.DataParallel(model,device_ids = range(torch.cuda.device_count()))
        cudnn.benchmark = True
    return model,best_acc,start_epoch

def acc_metrics():
    pass


def train(args,train_loader,model,criterion,optimizer,epoch,progress,train_time):
	batch_time = utilities.AverageMeter()
	data_time = utilities.AverageMeter()

	model.train()
	correct = 0

	end = time.time()

	for i,(data,label) in enumerate(train_loader):
		data_time.update(time.time()-end)
		if args.cuda:
			data,label = data.cuda(),label.cuda()
		data, label = Variable(data),Variable(label) 
		optimizer.zero_grad()
		output = model(data)
		loss = criterion(output,label)
		loss.backward()
		optimizer.step()
		pred = output.data.max(1)[1]
		correct += pred.eq(label.data).cpu().sum()

		batch_time.update(time.time() - end)
		end = time.time()

        if i % args.log_interval == 0:
            print('#Training Epoch:{} [{}/{} ({:.0f}%)]\tLoss:{:.6f}'.format(epoch,i*len(data),len(train_loader.dataset),100.*i/len(train_loader),loss.data[0]))
	train_time.update(batch_time.get_sum())
	train_acc = 100. * correct / len(train_loader.dataset)
	progress['train'].append((
		epoch, loss.data[0],train_acc,batch_time.get_sum(), batch_time.get_avg(),
							data_time.get_sum(),data_time.get_avg()
							))



def test(args,test_loader,model,criterion,epoch,progress,best_acc,test_time):
	model.eval()
	test_loss = 0
	correct = 0

	end = time.time()

	for data, label in test_loader:
		if args.cuda:
			data,label = data.cuda(),label.cuda()
		data, label = Variable(data,volatile=True),Variable(label)
		output = model(data)
		#I don't understand this line...
		test_loss += criterion(output,label).data[0]
		pred = output.data.max(1)[1]
		correct += pred.eq(label.data).cpu().sum()
	
	test_time.update(time.time() - end)
	test_loss /= len(test_loader.dataset)
	test_acc = 100. * correct/len(test_loader.dataset)
		#print('CORRET is',correct)
	print('\n #Epoch{}//avg.loss{:4f},accuracy:{}/{}, ==> {}%\n'.format(epoch,test_loss,correct,len(test_loader.dataset),test_acc))
	progress['test'].append((epoch,test_loss,test_acc))


	#Save checkpoint

	if test_acc > best_acc:
		print('Saving checkpoint..')

		state = {
		'model':model.module if args.cuda else model,
		'acc':test_acc,
		'epoch': epoch + 1
		}
		if not os.path.isdir('checkpoint'):
			os.mkdir('checkpoint')
		torch.save(state, './checkpoint/ckpt.pth')
		best_acc = test_acc


if __name__ == '__main__':

	parser = ArgumentParser(description = 'PyTorch Self-designed framework  example')
	parser.add_argument('--no-cuda',type = bool,default = False,metavar = 'T/F',help = 'enable CUDA acceleration')
	parser.add_argument('--batch_size',type = int, default = 128, metavar = 'N',
		help = 'input batch size for training, default 64')
	parser.add_argument('--test_batch_size',type = int,default = 1000, metavar = 'N',
		help = 'test batch size, default is 1000')
	parser.add_argument('--epochs',type = int, default = 10, metavar = 'N',
		help = '# epochs of train')
	parser.add_argument('--lr',type = float, default = 0.01, metavar = 'LR',
		help = 'learning rate')
	parser.add_argument('--momentum',type = float,default = 0.9, metavar = 'M',
		help = 'SGD momentum, default is 0.9')
	parser.add_argument('--weight-decay',type = float, default = 1e-4,metavar='W',
        help = 'weight decay rate default is 1e-4')
	parser.add_argument('--workers',type = int, default = 2, metavar = 'N',
        help = '# of data loading workers, default is 2')
	parser.add_argument('--resume',action = 'store_true',default = False,help = 'resume')
	parser.add_argument('--seed',type = int, default = 1, metavar = 'S',
		help = 'random seed, default is 1')
	parser.add_argument('--log-interval',type = int, default = 10,metavar = 'N',
		help = 'how many batches to wait before logging')
	parser.add_argument('--model',type = str, default = 'LeNet',metavar= 'MODEL',
        help = 'model architecture(default is LeNet,you can choose ResNetNum)')
	parser.add_argument('--dataset',type = str,default = 'Mnist',metavar = 'DATA',
        help = 'dataset,default is Mnist')
	parser.add_argument('--verbose',type = bool,metavar = 'T/F',default = True,
        help = 'print detailed training history, default is True')
	parser.add_argument('--lr_clipping',type = bool, metavar = 'T/F',default = False,
		help = 'clip learning rate according to the milestone schedule, default is False')
	parser.add_argument('--milestones',type = str, default = '0',metavar = 'M',
		help = "milestones to adjust learning rate, ints split by '-'")
	args = parser.parse_args()
	args.cuda = (not args.no_cuda) and (torch.cuda.is_available())

	torch.manual_seed(args.seed)
	if args.cuda:
		torch.cuda.manual_seed(args.seed)

	#Load training and testing data
	train_loader,test_loader = get_data_loader(args)
	#Load model
	model, best_acc,start_epoch = get_model(args)
	criterion = get_criterion(args)
	optimizer = get_optimizer(args,model)
	lr = args.lr
	if args.lr_clipping:
		milestones = utilities.parse_milestones(args.milestones)

	progress = {}
	progress['train'] = []
	progress['test'] = []
	train_time = utilities.AverageMeter()
	test_time = utilities.AverageMeter()


	for epoch in range(start_epoch,start_epoch+args.epochs):
		if args.lr_clipping:
			utilities.adjust_learning_rate(optimizer,lr,epoch,milestones)
	#train function
		train(args,train_loader,model,criterion,optimizer,epoch,progress,train_time)
		test(args,test_loader,model,criterion,epoch,progress,best_acc,test_time)
	progress['train_time'] = (train_time.get_avg(),train_time.get_sum())
	progress['test_time'] = (test_time.get_avg()/len(test_loader.dataset),test_time.get_avg())


	current_time = utilities.get_current_time()

	pickle.dump(progress,open('./'+args.model+('-resume' if args.resume else '')+ '_progress_'+current_time+'.pkl','wb'))