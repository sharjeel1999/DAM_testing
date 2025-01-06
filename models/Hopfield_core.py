import torch
import cv2
import os

from utils import hamming_score

class Hopfield_Core():
    def __init__(self, args, weight_folder, save_folder):
        self.args = args
        self.weight_folder = weight_folder
        self.save_folder = save_folder

    def calculate_similarity(self, generated, original):
        return hamming_score(generated, original)
    
    def save_weights(self):
        torch.save(self.weights, os.path.join(os.getcwd(), self.weight_folder))

    def load_weights(self):
        self.weights = torch.load(os.path.join(os.getcwd(), self.weight_folder))

    def save_files(self, pattern, perturbed, i):
        pattern = pattern.detach().cpu().numpy()
        perturbed = perturbed.detach().cpu().numpy()
        pattern = pattern.reshape(self.args.input_shape, self.args.input_shape)
        perturbed = perturbed.reshape(self.args.input_shape, self.args.input_shape)

        pattern = np.where(pattern == 1, 100, 0)
        perturbed = np.where(perturbed == 1, 100, 0)
        print('uniques: ', np.unique(pattern), np.unique(perturbed))

        name_original = str(i) + '_Original.png'
        name_recovered = str(i) + '_Recovered.png'
        cv2.imwrite(os.path.join(self.visual_folder, name_original), pattern)
        cv2.imwrite(os.path.join(self.visual_folder, name_recovered), perturbed)

    
