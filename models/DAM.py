import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch.optim as optim
from typing import Optional, Tuple, Union

from models.Hopfield_core import Hopfield_Core
from utils import perturb_pattern, Thresh

import matplotlib.pyplot as plt

class Continous_DAM(Hopfield_Core):
    def __init__(self,
                 args,
                 weight_folder,
                 visual_folder,
                 ):
        
        super(Continous_DAM, self).__init__(args, weight_folder, visual_folder)

        self.training_epochs = args.training_epochs
        self.pattern_size = args.pattern_size
        self.mem_size = args.mem_size
        self.mem_dim = args.mem_dim

        self.weights = nn.Parameter(torch.rand((self.mem_size, self.mem_dim)))

        self.query_proj = nn.Linear(self.pattern_size, self.mem_dim)
        self.key_proj = nn.Linear(self.mem_dim, self.mem_dim)
        self.value_proj = nn.Linear(self.mem_dim, self.mem_dim)
        self.output_proj = nn.Linear(self.mem_dim, self.pattern_size)

        self.beta = 8
        self.parameters = [self.weights, 
                        self.query_proj.weight, self.query_proj.bias,
                        self.key_proj.weight, self.key_proj.bias,
                        self.value_proj.weight, self.value_proj.bias,
                        self.output_proj.weight, self.output_proj.bias]
        

        self.optimizer = optim.Adam(self.parameters, 0.001)
        self.loss_function = nn.MSELoss() #nn.HuberLoss(delta=1.0, reduction='mean')
        
    def association_forward(self, pattern):
        q = self.query_proj(pattern)
        k = self.key_proj(self.weights)
        v = self.value_proj(self.weights)

        print('only matmul shapes: ', torch.matmul(q, k.t()).shape)
        attn_weights = F.softmax(self.beta*torch.matmul(q, k.t()) / (self.mem_dim ** 0.5), dim = 1)
        print('attn weights shape: ', attn_weights.shape)
        attn_output = torch.matmul(attn_weights, v)
        print('matmul output: ', attn_output.shape)

        output = self.output_proj(attn_output)
        return output
    

    def train(self, pattern_loader):
        print('implement train')

        for e in range(self.training_epochs):
            for pattern_dict in pattern_loader:
                pattern = torch.squeeze(pattern_dict['image']).float()
                perturbed_pattern = perturb_pattern(pattern, self.args.perturb_percent, self.args.crop_percent, self.args.corrupt_type)

                associated_output = self.association_forward(perturbed_pattern)
                
                loss = self.loss_function(associated_output, pattern)
                hamming = self.calculate_similarity(perturbed_pattern, pattern)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        self.save_weights()

    def recall(self, pattern_loader, steps=5):
        print('implement recall')

        self.load_weights()

        for m, pattern_dict in enumerate(pattern_loader):
            print('index: ', m)
            pattern = torch.squeeze(pattern_dict['image']).float()
            perturbed_pattern = torch.squeeze(pattern_dict['perturbed']).float()
        
            perturbed_hamming = self.calculate_similarity(perturbed_pattern, pattern)
            print(f'Perturbed Hamming Score: {perturbed_hamming}')

            print(f'Recovering pattern for {steps} steps.')
            for s in range(steps):
                perturbed_pattern = perturbed_pattern.unsqueeze(dim = 0)
                perturbed_pattern = self.association_forward(perturbed_pattern)
                perturbed_pattern = torch.squeeze(perturbed_pattern)
                
                hamming = self.calculate_similarity(perturbed_pattern, pattern)
                print(f'Step: {s}, Hamming Score: {hamming}')

            self.save_files(pattern, perturbed_pattern, m)

