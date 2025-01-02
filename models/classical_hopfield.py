import torch
from utils import hamming_score

class Classical_HN:
    def __init__(self, args):
        self.num_neurons = args.pattern_size
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

    def calculate_similarity(self, generated, original):
        return hamming_score(generated, original)

    def recall(self, pattern, steps=5):
        pattern = torch.tensor(pattern, dtype=torch.float32)
        copied_pattern = pattern.clone()

        print(f'Recovering pattern for {steps} steps.')
        for s in range(steps):
            for i in range(self.num_neurons):
                weighted_sum = torch.matmul(self.weights[i], pattern)
                pattern[i] = 1.0 if weighted_sum >= 0 else -1.0
            hamming = self.calculate_similarity(pattern, copied_pattern)
            print(f'Step: {s}, Hamming Score: {hamming}')
        return pattern

