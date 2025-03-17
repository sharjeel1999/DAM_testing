import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch.optim as optim

from models.Hopfield_core import Hopfield_Core
from utils import perturb_pattern, Thresh, Combined_loss

class Spherical_memory(Hopfield_Core):
    def __init__(self,
                 args,
                 weight_folder,
                 visual_folder,
                 ):
        
        super(Spherical_memory, self).__init__(args, weight_folder, visual_folder)

        self.training_epochs = args.training_epochs
        self.pattern_size = args.pattern_size
        self.mem_size = args.mem_size
        self.mem_dim = args.mem_dim

        self.alpha = 1

        self.weights = nn.Parameter(torch.rand((self.mem_size, self.mem_dim)))

        self.in_proj = nn.Linear(self.pattern_size, self.mem_dim)
        self.W_proj = nn.Linear(self.mem_dim, self.mem_dim)
        self.WT_proj = nn.Linear(self.mem_dim, self.mem_dim)
        self.output_proj = nn.Linear(self.mem_dim, self.pattern_size)

    def association(self, pattern):
        pattern = self.query_proj(pattern)
        W = self.W_proj(self.weights)
        WT = self.WT_proj(self.weights)

        if len(pattern.shape) == 1:
            pattern = F.normalize(pattern, p = 2, dim = 0)
        else:
            pattern = F.normalize(pattern, p = 2, dim = 1)
        
        inter_mul = torch.matmul(pattern, WT.t()) / (self.mem_dim ** 0.5)


    def train(self, pattern_loader):
        print('write')
        for e in range(self.training_epochs):
            for pattern_dict in pattern_loader:
                pattern = torch.squeeze(pattern_dict['image']).float()
                perturbed_pattern = perturb_pattern(pattern.clone(), self.args.perturb_percent, self.args.crop_percent, self.args.corrupt_type)

    def recall(self):
        print('write')
