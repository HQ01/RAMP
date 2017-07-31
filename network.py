import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import math
from argparse import ArgumentParser



class LeNet(nn.Module):
	def __init__(self):
		super(LeNet,self).__init__()
		self.conv1 = nn.Conv2d(1,10,kernel_size=5)
		self.conv2 = nn.Conv2d(10,20,kernel_size =5)
		self.conv2_drop = nn.Dropout2d()
		self.fc1 = nn.Linear(320,50)
		#self.fc1_drop = nn.Dropout()
		self.fc2 = nn.Linear(50,10)
	def forward(self,x):
		x = F.relu(F.max_pool2d(self.conv1(x),2))
		x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)),2))
		x = x.view(x.size(0),-1)
		x = F.dropout(self.fc1(x),training = self.training)
		x = self.fc2(x)
		return F.log_softmax(x)





































vgg_config = {
        'VGG11':{
            'basis':[64,128,256,512,512],
            'multiple':[1,1,2,2,2],
            'pool_type':'max'
                },
        'VGG13':{
                'basis':[64,128,256,512,512],
                'multiple':[2,2,2,2,2],
                'pool_type':'max'
                }, 
        'VGG16':{
                'basis':[64,128,256,512,512],
                'multiple':[2,2,3,3,3],
                'pool_type':'max'
                },
        'VGG19':{
                'basis':[64,128,256,512,512],
                'multiple':[2,2,4,4,4],
                'pool_type':'max'
                }
            }

class VGG(nn.Module):
    def __init__(self,config=vgg_config,network_name='VGG11'):
        super(VGG,self).__init__()
        self.mainBody = self._builder(config[network_name])
        # Here we call network after the first affine layer as tail network
        self.fc1 = nn.Linear(512,512)
        self.fc2 = nn.Linear(512,512)
        self.fc3 = nn.Linear(512,10)
       # self.inshape = 3

    def forward(self,x):
        x = self.mainBody(x)
        x = self._flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def _flatten(self,x):
        return x.view(x.size(0),-1)

    def _builder(self,config):
        # unzip arguments, pass down to block construction, common to all networks.

        basis,multiple,pool_type = config['basis'],config['multiple'],config['pool_type']
        layers = self._block(zip(basis,multiple),pool_type)
        return nn.Sequential(*layers)

    def _block(self,config,pool_type):
        #build up sequential of layer by specifying each block's #conv, #block. Each block end with a specific type of pooling
        result = []
        inshape = 3
        for channel,multiple in config:
            for i in range(multiple):
                result+=[self._conv(inshape,channel)]
                inshape = channel
            if pool_type == 'max':
                result+=[nn.MaxPool2d(kernel_size = 2, stride = 2)]
        return result

    def _conv(self,in_channels,out_channels):
        #wrapper for conv series,note the usage of in_channels
        return nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size = 3,padding = 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True)
            )



















ResNet_depth = [20,32,44,56,110,164,1001]
"""
ResNet_config = {
'ResNet20':20,
'ResNet32':32,
'ResNet44':44,
'ResNet56':56,
'ResNet110':110,
'ResNet164':164,
'ResNet1001':1001
}
"""

class ResBlock(nn.Module):
    step = 1
    def __init__(self,in_channels,out_channels,first = False):
        super(ResBlock,self).__init__()
        if first:
            self.conv1 = ResBlock._3x3conv(in_channels,out_channels*2,stride = 2)
            self.conv2 = ResBlock._3x3conv(out_channels*2,out_channels)
            self.shortcut = ResBlock._skip(in_channels,out_channels)
            self.first = first
        else:
            self.conv1 = ResBlock._3x3conv(in_channels,out_channels)
            self.conv2 = ResBlock._3x3conv(out_channels,out_channels)
            self.first = first

    @staticmethod
    def _3x3conv(in_channels,out_channels,stride = 1):
        return nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size = 3, padding = 1,stride = stride),
            nn.BatchNorm2d(out_channels),
            )
    def forward(self,x):
        identity = x
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        if self.first:
            x = x + self.shortcut(identity)
        else:
            x = x + identity
        x = F.relu(x)
        return x
    @staticmethod
    def _skip(in_channels,out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size = 1,stride = 2,bias = False),
            nn.BatchNorm2d(out_channels)
            )

class ResBottleneck(nn.Module):
    step = 4
    def __init__(self,in_channels,out_channels,first = False):
        super(ResBottleneck,self).__init__()
        if first:
            self.conv1 = ResBottleneck._1x1conv(in_channels,out_channels)
            self.conv2 = ResBottleneck._3x3conv(out_channels,out_channels,stride = 2)
            self.conv3 = ResBottleneck._1x1conv(out_channels,out_channels*self.step)
            self.shortcut = self._skip(in_channels,out_channels)
            self.first = first
        else:
            self.conv1 = ResBottleneck._1x1conv(in_channels,out_channels)
            self.conv2 = ResBottleneck._3x3conv(out_channels,out_channels)
            self.conv3 = ResBottleneck._1x1conv(out_channels,out_channels*self.step)
            self.first = first

    def forward(self,x):
        identity = x
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        if self.first:
            shortcut = self.shortcut(identity)
        else:
            x = x + identity
        x = F.relu(x)
        return x

    @staticmethod
    def _3x3conv(in_channels,out_channels,stride = 1):
        return nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size = 3,padding = 1,stride = stride),
            nn.BatchNorm2d(out_channels),
            #nn.ReLu()
            )
    @staticmethod
    def _1x1conv(in_channels,out_channels,stride = 1):
        return nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size = 1,stride = stride),
            nn.BatchNorm2d(out_channels)
            )

    def _skip(self,in_channels,out_channels,stride = 1):
        return nn.Sequential(
            nn.Conv2d(in_channels,out_channels*self.step,kernel_size = 1,stride = 2,bias = False),
            nn.BatchNorm2d(out_channels*self.step)
            )


class ResNet(nn.Module):
    def __init__(self,depth = 32 ,num_labels = 10,network_name = 'ResNet'):
        assert depth in ResNet_depth,'depth not found'
        super(ResNet,self).__init__()
        self.block = (ResBlock,1) if depth < 110 else (ResBottleneck,4)
        block_size = (depth - 2) // 6 if depth < 110 else (depth - 2) // 9
        filters = [16,16,32,64]
        self.conv1 = ResNet._3x3conv(3,filters[0])
        self.in_channels = filters[0]
        self.block1 = self._builder(self.block,block_size,filters[1])
        self.block2 = self._builder(self.block,block_size,filters[2])
        self.block3 = self._builder(self.block,block_size,filters[3])
        self.fc1 = nn.Linear(self.in_channels,num_labels)

        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0,math.sqrt(2./n))
            elif isinstance(m,nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = F.avg_pool2d(x,x.size()[2:])
        #kernel_size = input.size()[2:], try this later
        x = ResNet._flatten(x)
        x = self.fc1(x)

        return x
        

    def _builder(self,block,block_size,out_channels):
        block,step = block
        result = []
        result.append(block(self.in_channels,out_channels,first = True))
        self.in_channels = out_channels
        for num_block in range(block_size-1):
            result.append(block(step*self.in_channels,out_channels))
            self.in_channels = out_channels*step
        return nn.Sequential(*result)

    @staticmethod
    def _3x3conv(in_channels,out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size = 3, padding = 1),
            nn.BatchNorm2d(out_channels),
            )
    @staticmethod
    def _flatten(x):
        return x.view(x.size(0),-1)


        

