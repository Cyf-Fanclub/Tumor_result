from torch import nn
import torch.nn.functional as F

class Residual(nn.Module):
    def __init__(self, input_channels, hide_channels, num_channels, downsampling=False, strides=1):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, hide_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(hide_channels),
            nn.ReLU(),

            nn.Conv2d(hide_channels, hide_channels, kernel_size=3, padding=1, stride=strides),
            nn.BatchNorm2d(hide_channels),
            nn.ReLU(),

            nn.Conv2d(hide_channels, num_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(num_channels))
        if downsampling:
            self.identity = nn.Sequential(
                nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides),
                nn.BatchNorm2d(num_channels)
            )
        elif input_channels != num_channels:
            self.identity = nn.Sequential(
                nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=1),
                nn.BatchNorm2d(num_channels))
        else:
            self.identity = None
    def forward(self, X):
        Y = self.conv(X)
        if self.identity:
            X = self.identity(X)
        Y += X
        return F.relu(Y)

def resnet_block(input_channels, hide_channels, num_channels, num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, hide_channels, num_channels, downsampling=True, strides=2))
        else:
            blk.append(Residual(input_channels, hide_channels, num_channels))
        input_channels = num_channels
    return blk

class ResNet(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(ResNet, self).__init__()

        self.b1 = nn.Sequential(
                    nn.Conv2d(input_channels, 64, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(), 
                    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(), 
                    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(), 
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.b2 = nn.Sequential(*resnet_block(64, 64, 256, 3, first_block=True))
        self.b3 = nn.Sequential(*resnet_block(256, 128, 512, 4))
        self.b4 = nn.Sequential(*resnet_block(512, 256, 1024, 23))
        self.b5 = nn.Sequential(*resnet_block(1024, 512, 2048, 3))

        self.fc = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=1, stride=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(1024, output_channels)
        )
    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        return self.fc(x)