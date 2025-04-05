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

from matplotlib.lines import Line2D


class DAM_layer(nn.Module):
    def __init__(self, mem_size, mem_dim, beta):
        super(DAM_layer, self).__init__()
        self.mem_size = mem_size
        self.mem_dim = mem_dim
        self.beta = beta

        self.weights = nn.Parameter(torch.rand((self.mem_size, self.mem_dim)))
        nn.init.normal_(self.weights, mean = 0.0, std = 0.6)

        self.key_proj = Linear_projection(self.mem_dim, self.mem_dim)
        self.value_proj = Linear_projection(self.mem_dim, self.mem_dim)

    def forward(self, q):
        k = self.key_proj(self.weights)
        v = self.value_proj(self.weights)

        attn_weights = F.softmax(self.beta*torch.matmul(q, k.t()) / (self.mem_dim ** 0.5), dim = 1)
        attn_output = torch.matmul(attn_weights, v)

        return attn_output


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

        # self.weights = nn.Parameter(torch.rand((self.mem_size, self.mem_dim)))
        # nn.init.normal_(self.weights, mean = 0.0, std = 0.6)

        self.beta = 8

        self.query_proj = Linear_projection(self.pattern_size, self.mem_dim)
        self.output_proj = Linear_projection(self.mem_dim, self.pattern_size)

        self.layer1 = DAM_layer(self.mem_size, self.mem_dim, self.beta)
        self.layer2 = DAM_layer(self.mem_size, self.mem_dim, self.beta)
        # self.layer3 = DAM_layer(self.mem_size, self.mem_dim, self.beta)
        # self.layer4 = DAM_layer(self.mem_size, self.mem_dim, self.beta)
        # self.layer5 = DAM_layer(self.mem_size, self.mem_dim, self.beta)


        self.optimizer = optim.Adam(self.parameters(), 0.001)
        # self.optimizer = optim.SGD(self.parameters(), 0.01)
        self.loss_function = nn.MSELoss()
        
    def association_forward(self, pattern):
        q = self.query_proj(pattern)

        attn_output = self.layer1(q)
        attn_output = self.layer2(attn_output)
        # attn_output = self.layer3(attn_output)
        # attn_output = self.layer4(attn_output)
        # attn_output = self.layer5(attn_output)

        output = self.output_proj(attn_output)
        return output
    

    def train(self, pattern_loader):
        print('implement train')

        for e in range(self.training_epochs):
            for pattern_dict in pattern_loader:
                pattern = torch.squeeze(pattern_dict['image']).float()
                perturbed_pattern = torch.squeeze(pattern_dict['perturbed']).float()
                # perturbed_pattern = perturb_pattern(pattern.clone(), self.args.perturb_percent, self.args.crop_percent, self.args.corrupt_type)
                # print('--- input/perturbed shapes: ', pattern.shape, perturbed_pattern.shape)

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
        mean_MSE = []
        mean_PSNR = []
        mean_SSIM = []

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
        
            perturbed_hamming = self.calculate_similarity(perturbed_pattern[500:], pattern[500:])
            print(f'Perturbed Hamming Score: {perturbed_hamming}')

            print(f'Recovering pattern for {steps} steps.')
            for s in range(steps):
                perturbed_pattern = perturbed_pattern.unsqueeze(dim = 0)
                perturbed_pattern = self.association_forward(perturbed_pattern.to(self.args.device))
                perturbed_pattern = torch.squeeze(perturbed_pattern)
                

                sim_score = self.calculate_similarity(perturbed_pattern.detach().cpu().numpy()[500:], pattern[500:])
                if self.args.pattern_type == 'binary':
                    print(f"Step: {s}, Hamming Score: {sim_score['hamming']}")
                else:
                    print(f"Step: {s}, MSE: {sim_score['MSE']} (Lower Better)")
                    print(f"Step: {s}, PSNR: {sim_score['PSNR']} (Higher Better)")
                    print(f"Step: {s}, SSIM: {sim_score['SSIM']} (Higher Better)")
                    
            self.save_files(pattern[500:], perturbed_pattern[500:], p_in[500:], m)
            mean_MSE.append(sim_score['MSE'])
            mean_PSNR.append(sim_score['PSNR'])
            mean_SSIM.append(sim_score['SSIM'])

        print('-- Mean MSE: ', np.mean(mean_MSE))
        print('-- Mean PSNR: ', np.mean(mean_PSNR))
        print('-- Mean SSIM: ', np.mean(mean_SSIM))
