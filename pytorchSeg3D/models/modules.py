import torch
import torch.nn as nn

class Conv_Block(nn.Module):

    def __init__(self, kernel_size, channel_list):
        super(Conv_Block, self).__init__()
        self.conv = nn.Conv3d(channel_list[0], channel_list[1], kernel_size, 1, (kernel_size-1)//2)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm3d(channel_list[1])

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.bn(x)

        return x


class Double_Conv(nn.Module):

    def __init__(self, kernel_size, channel_list):
        super(Double_Conv, self).__init__()
        self.conv1 = nn.Conv3d(channel_list[0], channel_list[1], kernel_size, 1, (kernel_size - 1) // 2)
        self.conv2 = nn.Conv3d(channel_list[1], channel_list[2], kernel_size, 1, (kernel_size - 1) // 2)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.bn1 = nn.BatchNorm3d(channel_list[1])
        self.bn2 = nn.BatchNorm3d(channel_list[2])

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.bn1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.bn2(x)

        return x




