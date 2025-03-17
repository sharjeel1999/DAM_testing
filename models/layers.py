import torch
import torch.nn as nn

class Linear_projection(nn.Module):
    def __init__(self, in_channels, out_channels):
        self.l1 = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        return self.l1(x)