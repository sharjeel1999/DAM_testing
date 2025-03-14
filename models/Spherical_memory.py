import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch.optim as optim

from models.Hopfield_core import Hopfield_Core

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

    def train(self):
        print('write')

    def recall(self):
        print('write')
