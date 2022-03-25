import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules import flatten

def conv_block(input_channels, num_channels):
    '''
    卷积块

    参数介绍：
    input_channels  输入通道数\n
    num_channels    输出通道数
    '''
    return nn.Sequential(
        nn.BatchNorm2d(input_channels), 
        nn.ReLU(),
        nn.Conv2d(input_channels, num_channels * 4, kernel_size=1),
        nn.ReLU(),
        nn.Conv2d(num_channels * 4, num_channels, kernel_size=3, padding=1))

def transition_block(input_channels, num_channels):
    '''
    过渡层

    降低通道数
    
    参数介绍：
        input_channels  输入通道数\n
        num_channels    输出通道数 
    '''
    return nn.Sequential(
        nn.BatchNorm2d(input_channels), 
        nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=1),
        nn.AvgPool2d(kernel_size=2, stride=2))

class DenseBlock(nn.Module):
    def __init__(self, num_convs, input_channels, num_channels):
        '''
        参数介绍
            num_convs       卷积层数量\n
            input_channels  输入通道数\n
            num_channels    卷积块通道数（增长率）\n
        '''
        super(DenseBlock, self).__init__()
        layer = []
        for i in range(num_convs):
            layer.append(conv_block(num_channels * i + input_channels, num_channels))
        self.net = nn.Sequential(*layer)
    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            # 连接通道维度上每个块的输⼊和输出
            X = torch.cat((X, Y), dim=1)
        return X

class DenseNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DenseNet, self).__init__()

        # 网络初始部分
        self.beginning = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(), 
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(), 
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        
        # 稠密部分
        num_channels, growth_rate = 64, 32
        num_convs_in_dense_blocks = [6, 12, 32, 32]
        blks = []
        for i, num_convs in enumerate(num_convs_in_dense_blocks):
            blks.append(DenseBlock(num_convs, num_channels, growth_rate))
            # 上⼀个稠密块的输出通道数
            num_channels += num_convs * growth_rate
            # 在稠密块之间添加⼀个转换层，使通道数量减半
            if i != len(num_convs_in_dense_blocks) - 1:
                blks.append(transition_block(num_channels, num_channels // 2))
                num_channels = num_channels // 2
        self.denseBlks = nn.Sequential(*blks)

        self.bn = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU()
        self.adaptivePool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(num_channels, out_channels)
    
    def forward(self, x):
        x = self.beginning(x)
        x = self.denseBlks(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.adaptivePool(x)
        x = self.flatten(x)
        x = self.fc(x)

        return x