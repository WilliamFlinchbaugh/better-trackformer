import torch
import torch.nn.functional as F
from torch import nn

class CameraState(nn.Module):
    def __init__(self, input_size=[3, 766, 1332], state_len=512, abs=False):
        super().__init__()
        
        self.abs = abs
        
        self.conv1 = nn.Conv2d(in_channels=input_size[0], out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2)
        flattened_size = int((input_size[1]/4) * (input_size[2]/4) * 128)
        self.fc = nn.Linear(flattened_size, state_len)

    def forward(self, curr_frame, prev_frame):
        diff = curr_frame - prev_frame
        if self.abs:
            diff = torch.abs(diff)
        
        x = F.relu(self.conv1(diff))
        x = self.pool(x)
        x = F.relu(self.conv2(diff))
        x = self.pool(x)
        x = F.relu(self.conv3(diff))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x