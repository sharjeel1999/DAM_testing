import torch
import torch.nn as nn

class Linear_projection(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Linear_projection, self).__init__()

        self.l1 = nn.Linear(in_channels, out_channels, bias = False)
        # self.ln1 = nn.LayerNorm(out_channels, elementwise_affine = False, bias = None)

        self.__initialize__()

    def __initialize__(self):
        nn.init.normal_(self.l1.weight, mean = 0.0, std = 0.2)
        # nn.init.ones_(self.ln1.weight)
        # nn.init.zeros_(self.ln1.bias)

    def forward(self, x):
        # return self.ln1(self.l1(x))
        return self.l1(x)