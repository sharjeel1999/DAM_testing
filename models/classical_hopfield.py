import os

import torch
from utils import hamming_score

class Classical_HN:
    def __init__(self, args, use_weights = False):
        self.args = args
        self.num_neurons = args.pattern_size
        self.weight_folder = 'All_saves\\classical_hopfield\\weights.pth'

        self.weights = torch.zeros((self.num_neurons, self.num_neurons))

    def train(self, pattern_loader):
        for pattern_dict in pattern_loader:
            pattern = torch.squeeze(pattern_dict['image'])
            print('input pattern shape: ', pattern.shape, pattern_dict['image'].shape)
            pattern = torch.tensor(pattern, dtype = torch.float32)
            # print('input pattern shape: ', pattern.shape)
            pattern_T = torch.transpose(pattern, 0, 1)
            self.weights += torch.matmul(pattern_T, pattern) #torch.outer(pattern, pattern)

        # Remove self-connections
        for i in range(self.num_neurons):
            self.weights[i, i] = 0

        self.save_weights()

    def calculate_similarity(self, generated, original):
        return hamming_score(generated, original)
    
    def save_weights(self):
        torch.save(self.weights, os.path.join(os.getcwd(), self.weight_folder))

    def load_weights(self):
        self.weights = torch.load(os.path.join(os.getcwd(), self.weight_folder))

    def recall(self, pattern_loader, steps=5):
        self.load_weights()

        for pattern_dict in pattern_loader:
            pattern = torch.squeeze(pattern_dict['image'])
            perturbed_pattern = torch.squeeze(pattern_dict['perturbed'])
            print('Pattern shapes: ', pattern.shape, perturbed_pattern.shape)
        
        pattern = torch.tensor(pattern, dtype=torch.float32)
        perturbed_pattern = torch.tensor(perturbed_pattern, dtype=torch.float32)
        copied_pattern = pattern.clone()

        print(f'Recovering pattern for {steps} steps.')
        for s in range(steps):
            for i in range(self.num_neurons):
                weighted_sum = torch.matmul(self.weights[i], perturbed_pattern)
                perturbed_pattern[i] = 1.0 if weighted_sum >= 0 else -1.0
            hamming = self.calculate_similarity(perturbed_pattern, pattern)
            print(f'Step: {s}, Hamming Score: {hamming}')
        # return perturbed_pattern, pattern

