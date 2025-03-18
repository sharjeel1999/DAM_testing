import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch.optim as optim

from models.Hopfield_core import Hopfield_Core
from models.layers import Linear_projection
from utils import perturb_pattern, Thresh, Combined_loss

import matplotlib.pyplot as plt
import numpy as np

class Continous_DAM(Hopfield_Core):
    def __init__(self,
                 args,
                 weight_folder,
                 visual_folder,
                 ):
        
        super(Continous_DAM, self).__init__(args, weight_folder, visual_folder)

        self.args = args
        self.training_epochs = args.training_epochs
        self.pattern_size = args.pattern_size
        self.mem_size = args.mem_size
        self.mem_dim = args.mem_dim

        self.weights = nn.Parameter(torch.rand((self.mem_size, self.mem_dim)))
        nn.init.normal_(self.weights, mean = 0.0, std = 0.6)


        self.query_proj = Linear_projection(self.pattern_size, self.mem_dim)
        self.key_proj = Linear_projection(self.mem_dim, self.mem_dim)
        self.value_proj = Linear_projection(self.mem_dim, self.mem_dim)
        self.output_proj = Linear_projection(self.mem_dim, self.pattern_size)

        self.beta = 8

        # self.optimizer = optim.Adam(self.parameters(), 0.001)
        self.optimizer = optim.SGD(self.parameters(), 0.01)
        self.loss_function = nn.MSELoss()
        
    def association_forward(self, pattern):
        q = self.query_proj(pattern)
        k = self.key_proj(self.weights)
        v = self.value_proj(self.weights)

        attn_weights = F.softmax(self.beta*torch.matmul(q, k.t()) / (self.mem_dim ** 0.5), dim = 1)
        attn_output = torch.matmul(attn_weights, v)

        output = self.output_proj(attn_output)
        return output
    

    def train(self, pattern_loader):
        print('implement train')

        for e in range(self.training_epochs):
            for pattern_dict in pattern_loader:
                pattern = torch.squeeze(pattern_dict['image']).float()
                perturbed_pattern = perturb_pattern(pattern.clone(), self.args.perturb_percent, self.args.crop_percent, self.args.corrupt_type)

                associated_output = self.association_forward(perturbed_pattern.to(self.args.device))
                # print('pattern/associated shapes: ', pattern.shape, associated_output.shape)
                loss = self.loss_function(associated_output, pattern.to(self.args.device))
                # loss = Combined_loss(associated_output, pattern.to(self.args.device))
                
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
            p_in = perturbed_pattern.clone()
            # p = torch.reshape(pattern, (64, 64)).cpu().numpy()
            # pp = torch.reshape(perturbed_pattern, (64, 64)).cpu().numpy()
            # plt.imshow(p)
            # plt.show()
            # plt.imshow(pp)
            # plt.show()
        
            perturbed_hamming = self.calculate_similarity(perturbed_pattern, pattern)
            print(f'Perturbed Hamming Score: {perturbed_hamming}')

            print(f'Recovering pattern for {steps} steps.')
            for s in range(steps):
                perturbed_pattern = perturbed_pattern.unsqueeze(dim = 0)
                perturbed_pattern = self.association_forward(perturbed_pattern.to(self.args.device))
                perturbed_pattern = torch.squeeze(perturbed_pattern)
                
                sim_score = self.calculate_similarity(perturbed_pattern.detach().cpu().numpy(), pattern)
                if self.args.pattern_type == 'binary':
                    print(f"Step: {s}, Hamming Score: {sim_score['hamming']}")
                else:
                    print(f"Step: {s}, MSE: {sim_score['MSE']} (Lower Better)")
                    print(f"Step: {s}, PSNR: {sim_score['PSNR']} (Higher Better)")
                    print(f"Step: {s}, SSIM: {sim_score['SSIM']} (Higher Better)")

            self.save_files(pattern, perturbed_pattern, p_in, m)

