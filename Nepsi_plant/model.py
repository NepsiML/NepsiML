import torch
import torch.nn as nn
import torchvision


class SkipConnection(nn.Module):
    def __init__(self, input_channels:int,
                        hidden_channels:int,
                        output_channels:int):
        super(SkipConnection, self).__init__()
        self.c1 = nn.Conv2d(in_channels=input_channels,
                            out_channels=hidden_channels,
                            kernel_size=3,
                            stride=1,
                            padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.c2 = nn.Conv2d(in_channels=hidden_channels,
                            out_channels=output_channels,
                            kernel_size=3,
                            stride=1,
                            padding=1)
        self.bn2 = nn.BatchNorm2d(output_channels)
        self.conv_skip = nn.Conv2d(in_channels=input_channels,
                            out_channels=output_channels,
                            kernel_size=1,
                            stride=2,
                            padding=0)
        
        self.dropout = nn.Dropout(p=0.4)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2)
        

    def forward(self, x):
        # save the input
        inp = self.conv_skip(x)
        # forward pass
        x = self.bn1(self.relu(self.c1(x)))
        x = self.dropout(x)
        x = self.maxpool(x)
        # forward pass
        x = self.bn2(self.c2(x))
        return x + inp