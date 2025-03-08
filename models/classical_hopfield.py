import os
import cv2
import numpy as np

import torch

from models.Hopfield_core import Hopfield_Core

class Classical_HN(Hopfield_Core):
    def __init__(self, args, weight_folder, visual_folder):

        super(Classical_HN, self).__init__(args, weight_folder, visual_folder)
        self.args = args
        self.num_neurons = args.pattern_size
        self.weight_folder = weight_folder
        self.visual_folder = visual_folder

        self.parameters = torch.zeros((self.num_neurons, self.num_neurons))

    def train(self, pattern_loader):
        for pattern_dict in pattern_loader:
            pattern = torch.squeeze(pattern_dict['image'])
            print('input pattern shape: ', pattern.shape, pattern_dict['image'].shape)
            pattern = torch.tensor(pattern, dtype = torch.float32)
            # print('input pattern shape: ', pattern.shape)
            pattern_T = torch.transpose(pattern, 0, 1)
            self.parameters += torch.matmul(pattern_T, pattern) #torch.outer(pattern, pattern)

        # Remove self-connections
        for i in range(self.num_neurons):
            self.parameters[i, i] = 0

        self.save_weights()

    def recall(self, pattern_loader, steps=5):
        self.load_weights()

        for m, pattern_dict in enumerate(pattern_loader):
            print('index: ', m)
            pattern = torch.squeeze(pattern_dict['image'])
            perturbed_pattern = torch.squeeze(pattern_dict['perturbed'])
            print('Pattern shapes: ', pattern.shape, perturbed_pattern.shape)
        
            pattern = torch.tensor(pattern, dtype=torch.float32)
            perturbed_pattern = torch.tensor(perturbed_pattern, dtype=torch.float32)
            copied_pattern = pattern.clone()

            print(f'Recovering pattern for {steps} steps.')
            for s in range(steps):
                for i in range(self.num_neurons):
                    weighted_sum = torch.matmul(self.parameters[i], perturbed_pattern)
                    perturbed_pattern[i] = 1.0 if weighted_sum >= 0 else -1.0
                hamming = self.calculate_similarity(perturbed_pattern, pattern)
                print(f'Step: {s}, Hamming Score: {hamming}')

            self.save_files(pattern, perturbed_pattern, m)
        # return perturbed_pattern, pattern

