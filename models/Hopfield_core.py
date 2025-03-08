import torch
import cv2
import os
import numpy as np

from utils import hamming_score

class Hopfield_Core():
    def __init__(self, args, weight_folder, visual_folder):
        self.args = args
        self.weight_folder = weight_folder
        self.visual_folder = visual_folder

    def calculate_similarity(self, generated, original):
        return hamming_score(generated, original)
    
    def save_weights(self):
        torch.save(self.parameters, os.path.join(os.getcwd(), self.weight_folder))

    def load_weights(self):
        self.parameters = torch.load(os.path.join(os.getcwd(), self.weight_folder))

    def save_files(self, pattern, perturbed, i):
        pattern = pattern.detach().cpu().numpy()
        perturbed = perturbed.detach().cpu().numpy()
        pattern = pattern.reshape(self.args.input_shape, self.args.input_shape)
        perturbed = perturbed.reshape(self.args.input_shape, self.args.input_shape)

        if self.args.save_files == 'binary':
            pattern = np.where(pattern < 0, 0, 100)
            perturbed = np.where(perturbed < 0, 0, 100)
        print('uniques: ', np.unique(pattern), np.unique(perturbed))

        name_original = str(i) + '_Original.png'
        name_recovered = str(i) + '_Recovered.png'
        cv2.imwrite(os.path.join(self.visual_folder, name_original), pattern)
        cv2.imwrite(os.path.join(self.visual_folder, name_recovered), perturbed)

    
