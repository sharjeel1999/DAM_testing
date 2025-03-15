import torch
import torch.nn as nn

import cv2
import os
import numpy as np

from utils import hamming_score, PSNR, SSIM

class Hopfield_Core(nn.Module):
    def __init__(self, args, weight_folder, visual_folder):
        super(Hopfield_Core, self).__init__()

        self.args = args
        self.weight_folder = weight_folder
        self.visual_folder = visual_folder

    def calculate_similarity(self, generated, original):
        original = torch.tensor(original, dtype = torch.float32)
        generated = torch.tensor(generated, dtype = torch.float32)

        out = {}
        if self.args.pattern_type == 'binary':
            out['hamming'] = hamming_score(generated, original)
        else:
            out['MSE'] = torch.mean((generated - original) ** 2)
            out['PSNR'] = PSNR(generated, original)
            out['SSIM'] = SSIM(generated, original, self.args.input_shape)
        return out
    
    def save_weights(self):
        torch.save(self.parameters, os.path.join(os.getcwd(), self.weight_folder))

    def load_weights(self):
        self.parameters = torch.load(os.path.join(os.getcwd(), self.weight_folder))

    def save_files(self, pattern, perturbed, p_in, i):
        pattern = pattern.detach().cpu().numpy()
        perturbed = perturbed.detach().cpu().numpy()
        p_in = p_in.detach().cpu().numpy()
        pattern = pattern.reshape(self.args.input_shape, self.args.input_shape)
        perturbed = perturbed.reshape(self.args.input_shape, self.args.input_shape)
        p_in = p_in.reshape(self.args.input_shape, self.args.input_shape)

        if self.args.pattern_type == 'binary':
            pattern = np.where(pattern < 0, 0, 100)
            perturbed = np.where(perturbed < 0, 0, 100)
        # print('uniques: ', np.unique(pattern), np.unique(perturbed))

        name_original = str(i) + '_Original.png'
        name_recovered = str(i) + '_Recovered.png'
        name_input = str(i) + '_Input.png'
        cv2.imwrite(os.path.join(self.visual_folder, name_original), pattern)
        cv2.imwrite(os.path.join(self.visual_folder, name_recovered), perturbed)
        cv2.imwrite(os.path.join(self.visual_folder, name_input), p_in)

    
