import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch.optim as optim

from models.Hopfield_core import Hopfield_Core
from models.layers import Linear_projection
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

        self.in_proj = Linear_projection(self.pattern_size, self.mem_dim)
        self.W_proj = Linear_projection(self.mem_dim, self.mem_dim)
        self.WT_proj = Linear_projection(self.mem_dim, self.mem_dim)
        self.output_proj = Linear_projection(self.mem_dim, self.pattern_size)

        # self.optimizer = optim.Adam(self.parameters(), 0.001)
        self.optimizer = optim.SGD(self.parameters(), 0.01)
        self.loss_function = nn.MSELoss()

    def association_forward(self, pattern):
        pattern = self.in_proj(pattern)
        W = self.W_proj(self.weights)
        WT = self.WT_proj(self.weights)

        if len(pattern.shape) == 1:
            pattern = F.normalize(pattern, p = 2, dim = 0)
        else:
            pattern = F.normalize(pattern, p = 2, dim = 1)
        
        inter_mul = F.softmax(torch.matmul(pattern, WT.t()) / (self.mem_dim ** 0.5), dim = 1)
        attn_out = torch.matmul(inter_mul, W)

        output = self.output_proj(attn_out)
        return output

    def train(self, pattern_loader):
        for e in range(self.training_epochs):
            for pattern_dict in pattern_loader:
                pattern = torch.squeeze(pattern_dict['image']).float()
                perturbed_pattern = perturb_pattern(pattern.clone(), self.args.perturb_percent, self.args.crop_percent, self.args.corrupt_type)

                associated_output = self.association_forward(perturbed_pattern.to(self.args.device))
                loss = self.loss_function(associated_output, pattern.to(self.args.device))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        self.save_weights()


    def recall(self, pattern_loader, steps=5):
        self.load_weights()

        for m, pattern_dict in enumerate(pattern_loader):
            print('index: ', m)
            pattern = torch.squeeze(pattern_dict['image']).float()
            perturbed_pattern = torch.squeeze(pattern_dict['perturbed']).float()
            p_in = perturbed_pattern.clone()
        
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
