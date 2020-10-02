import torch
import torch.nn as nn
import torch.functional as F


class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(conv_block, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.relu(self.batchnorm(self.conv(x)))

class Inception_block(nn.Module):
    def __init__(
        self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool
    ):
        super(Inception_block, self).__init__()
        self.branch1 = conv_block(in_channels, out_1x1, kernel_size=(1, 1))

        self.branch2 = nn.Sequential(
            conv_block(in_channels, red_3x3, kernel_size=(1, 1)),
            conv_block(red_3x3, out_3x3, kernel_size=(3, 3), padding=(1, 1)),
        )

        self.branch3 = nn.Sequential(
            conv_block(in_channels, red_5x5, kernel_size=(1, 1)),
            conv_block(red_5x5, out_5x5, kernel_size=(5, 5), padding=(2, 2)),
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            conv_block(in_channels, out_1x1pool, kernel_size=(1, 1)),
        )

    def forward(self, x):
        return torch.cat(
            [self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], 1
        )        
class GoogLeNet(nn.Module):
    def __init__(self):
        super(GoogLeNet, self).__init__()
        

        # Write in_channels, etc, all explicit in self.conv1, rest will write to
        # make everything as compact as possible, kernel_size=3 instead of (3,3)
        self.conv1 = conv_block(
            in_channels=3,
            out_channels=64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
        )

        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = conv_block(64, 192, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # In this order: in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool
        self.inception3a = Inception_block(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception_block(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)
        
        self.avgpool = nn.AvgPool2d(kernel_size=3,stride=1)
        self.flat = nn.Flatten()
        self.dense = nn.Linear(69120,128)

        # self.inception4a = Inception_block(480, 192, 96, 208, 16, 48, 64)
        # self.inception4b = Inception_block(512, 160, 112, 224, 24, 64, 64)
        # self.inception4c = Inception_block(512, 128, 128, 256, 24, 64, 64)
        # self.inception4d = Inception_block(512, 112, 144, 288, 32, 64, 64)
        # self.inception4e = Inception_block(528, 256, 160, 320, 32, 128, 128)
        # self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # self.inception5a = Inception_block(832, 256, 160, 320, 32, 128, 128)
        # self.inception5b = Inception_block(832, 384, 192, 384, 48, 128, 128)

    def forward(self, x):
        x = self.conv1(x)
        # print(x.shape)
        x = self.maxpool1(x)
        # print(x.shape)
        x = self.conv2(x)
        # print(x.shape)
        # x = self.conv3(x)
        # print(x.shape)
        x = self.maxpool2(x)
        # print(x.shape)

        x = self.inception3a(x)
        # print(x.shape)
        x = self.inception3b(x)
        # print(x.shape)
        x = self.maxpool3(x)
        # print(x.shape)
        x = self.avgpool(x)
        # print(x.shape)
        x = self.flat(x)
        # print(x.shape)
        x = self.dense(x)
        # print(x.shape)
        norm = x.norm(p=2, dim=1, keepdim=True)
        x = x.div(norm.expand_as(x))
        # x = F.normalize(x, p=2, dim=1)
        # print(x.shape)
        return x