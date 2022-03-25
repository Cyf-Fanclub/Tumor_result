from torch import nn

class StackingNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(StackingNet, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(in_channels, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, out_channels)
        )
    
    def forward(self, x):
        return self.fc(x)