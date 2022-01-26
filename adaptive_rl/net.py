import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        init.xavier_uniform_(m.weight)

class ResidualBlock(nn.Module):
    def __init__(self, filters):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(filters, filters, 3, stride = 1, padding = 1)
        #self.batchnorm1 = nn.BatchNorm2d(filters)

        self.conv2 = nn.Conv2d(filters, filters, 3, stride = 1, padding = 1)
        #self.batchnorm2 = nn.BatchNorm2d(filters)

    def forward(self, input):
        x = self.conv1(input)
        #x = self.batchnorm1(x)
        x = F.relu(x)
        x = self.conv2(x)
        #x = self.batchnorm2(x)
        x = x + input
        y = F.relu(x)
        return y

class Net(nn.Module):
    def __init__(self, input_frames, blocks, filters, size):
        super(Net, self).__init__()
        self.blocks = blocks
        self.size = size
        self.features_count = size * 32

        self.conv1 = nn.Conv2d(input_frames, filters, 3, stride=1, padding=1)
        self.residual_blocks = nn.ModuleList([ResidualBlock(filters) for i in range(blocks)])

        self.conv_policy = nn.Conv2d(filters, 32, 3, stride=1, padding=1)
        self.fc1_p = nn.Linear(self.features_count, 256)
        self.fc2_p = nn.Linear(256, size)

        self.conv_value = nn.Conv2d(filters, 32, 3, stride=1, padding=1)
        self.fc1_v = nn.Linear(self.features_count, 128)
        self.fc2_v = nn.Linear(128, 3)

        self.conv_moves_left = nn.Conv2d(filters, 32, 3, stride=1, padding=1)
        self.fc1_m = nn.Linear(self.features_count, 256)
        self.fc2_m = nn.Linear(256, size)

        self.apply(weights_init_xavier)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        for i in range(self.blocks):
            x = self.residual_blocks[i](x)

        x_p = F.relu(self.conv_policy(x))
        x_p = x_p.view(-1, self.features_count)
        x_p = F.relu(self.fc1_p(x_p))
        y_p = self.fc2_p(x_p)

        x_v = F.relu(self.conv_value(x))
        x_v = x_v.view(-1, self.features_count)
        x_v = F.relu(self.fc1_v(x_v))
        y_v = self.fc2_v(x_v)

        x_m = F.relu(self.conv_moves_left(x))
        x_m = x_m.view(-1, self.features_count)
        x_m = F.relu(self.fc1_m(x_m))
        y_m = self.fc2_m(x_m)

        return y_p, y_v, y_m
