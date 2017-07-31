from torchvision import datasets,transforms
import torch
import torch.nn as nn
import torch.nn.functional as F

def FetchMnist(args):#cuda = True,batch_size = 32):
#A wrapper for fetching Mnist data, train_loader/test_loader, already normalized
	if args.cuda:
		kwargs = {'num_workers':1,'pin_memory':True}
	else:
		kwargs = {}
	train_loader = torch.utils.data.DataLoader(
		datasets.MNIST('../data',train=True,download = True,transform = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.1307,),(0.3081,))
			])),
		batch_size = args.batch_size,
		shuffle = True,
		**kwargs)
	test_loader = torch.utils.data.DataLoader(
		datasets.MNIST('../data',train=False, transform = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.1307,),(0.3081,))
			])),
		batch_size = args.batch_size, shuffle = True, **kwargs)
	return train_loader, test_loader
		
def FetchCifar10(args):#cuda = True,batch_size = 32,num_workers = 1):
#A wrapper for fetching Mnist data, train_loader/test_loader, already normalized
	if args.cuda:
		kwargs = {'num_workers':args.workers,'pin_memory':True}
	else:
		kwargs = {}
    #train_loader
	train_loader = torch.utils.data.DataLoader(
		datasets.CIFAR10('../data',train=True,download = True,transform = transforms.Compose([
        	transforms.RandomCrop(32,padding = 4),
        	transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize(mean = (0.4914,0.4822,0.4465),std = (0.2023,0.1994,0.2010))
			])),
		batch_size = args.batch_size,
		shuffle = True,
		**kwargs)
	#test_loader
	test_loader = torch.utils.data.DataLoader(
		datasets.CIFAR10('../data',train=False, transform = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize(mean = (0.4914,0.4822,0.4465),std = (0.2023,0.1994,0.2010))
			])),
		batch_size = args.test_batch_size, 
		shuffle = True,
		**kwargs)
	return train_loader,test_loader
	