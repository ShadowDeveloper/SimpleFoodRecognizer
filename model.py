import torch
from torch import nn

class CNN(nn.Module):
    def __init__(self, batch_size):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, (4, 4), 4) # 3x256x256 -> 64x64x64 using 4x4 kernel and 4 stride
        self.maxpool1 = nn.MaxPool2d(2, 2) # 64x64x64 -> 64x32x32
        self.conv2 = nn.Conv2d(64, 16, 2, 2) # 64x32x32 -> 16x16x16 using 2x2 kernel and 2 stride
        # normally there would be another maxpool here but the feature map is already small enough and I don't want to risk losing more detail. TODO: revist this
        self.flat = lambda x: x.view(x.size(0), -1) # 16x16x16 -> 4096 (flatten 3 dims into 1)
        self.fc1 = nn.Linear(4096, 256) # 128x4096 (flattened conv2, see line 24) -> 256
        self.fc2 = nn.Linear(256, 101) # 256 -> 101
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.2)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = self.relu(self.conv2(x))
        x = self.flat(x)
        x = self.relu(self.fc1(x))
        x = self.drop(x)
        x = self.relu(self.fc2(x))
        return x
