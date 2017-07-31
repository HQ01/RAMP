import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from network import *
from myio import *
import datetime
net_zoo = {
        'LeNet':LeNet(),
        'VGG11':VGG(network_name = 'VGG11'),
        'VGG13':VGG(network_name = 'VGG13'),
        'VGG16':VGG(network_name = 'VGG16'),
        'VGG19':VGG(network_name = 'VGG19'),
        'ResNet20':ResNet(depth = 20,network_name = 'ResNet20'),
        'ResNet32':ResNet(depth = 32, network_name = 'ResNet32'),
        'ResNet44':ResNet(depth = 44,network_name = 'ResNet44'),
        'ResNet56':ResNet(depth = 56, network_name = 'ResNet56'),
        'ResNet110':ResNet(depth = 110, network_name = 'ResNet110'),
        'ResNet164':ResNet(depth = 164,network_name = 'ResNet164'),
        'ResNet1001':ResNet(depth = 1001,network_name = 'ResNet1001')
        }

dataset_dict = {
        'Mnist':FetchMnist,
        'Cifar10':FetchCifar10
        }

def get_net(net):
    assert net in net_zoo.keys(), 'Network not found'
    print 'get_net returning ...',net_zoo[net]
    return net_zoo[net]

def get_dataset(dataset):
    assert dataset in dataset_dict,'Dataset not found'
    return dataset_dict[dataset]



class AverageMeter(object):
    """
    Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self,val,n = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def get_sum(self):
        return self.sum

    def get_avg(self):
        return self.avg
def parse_milestones(text):
    try:
        milestones = [int(e) for e in str.split('-')]
    except:
        print('Error: invalid milestones input')

    return milestones

def adjust_learning_rate(optimizer,lr,epoch,milestones):
    if epoch in milestones:
        lr = lr * 0.1
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

def get_current_time():
    now = datetime.datetime.now()

    return now.strftime('%Y-%m-%d_%H-%M-%S')


