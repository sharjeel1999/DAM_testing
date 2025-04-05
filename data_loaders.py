import numpy as np
import cv2
import os
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from utils import perturb_pattern, Thresh, hadamard_matrix


class Image_dataset(Dataset):
    def __init__(self, args, corrupt_flag = False):
        self.args = args
        self.folder_path = os.path.join(os.getcwd(), args.folder_path)
        self.num_images = args.num_images
        self.limit_in = 64

        # Corruption
        self.corrupt_flag = corrupt_flag
        self.perturb_percent = args.perturb_percent
        self.crop_percent = args.crop_percent
        self.corrupt_type = args.corrupt_type

        self.transform = transforms.Compose([
            transforms.Resize(size=(96, 96)),
            transforms.CenterCrop(size=(64, 64)),
            # transforms.ToTensor()              
        ])

        self.emb_size = 512
        self.H = np.array(hadamard_matrix(self.emb_size))

        self.collect_to_array()


    def create_embedding(self, i):
        emb = self.H[i]
        return emb

    def collect_to_array(self):
        self.image_array = []

        for i, name in enumerate(os.listdir(self.folder_path)):
            if i < self.num_images:
                image_path = os.path.join(self.folder_path, name)
                image = cv2.imread(image_path, 0)
                emb = self.create_embedding(i)
                self.image_array.append([image, emb])

    def __len__(self):
        return len(self.image_array)
    
    def __getitem__(self, index):
        inputs = {}
        if self.corrupt_flag == True:
            image, emb = self.image_array[index]

            if image.shape[0] > self.limit_in:
                image = Image.fromarray(image)
                image = np.array(self.transform(image))
            
            if self.args.pattern_type == 'binary':
                image = Thresh(np.array([image.flatten()-0.5]))
            else:
                image = np.array([image.flatten()])

            perturbed_image = perturb_pattern(image.copy(), self.args.perturb_percent, self.args.crop_percent, self.args.corrupt_type)
            
            emb = np.expand_dims(np.array(emb), axis = 0)
            # print('--- emb shape: ', emb.shape)
            inputs['image'] = np.concatenate((emb, image), axis = 1)
            inputs['perturbed'] = np.concatenate((emb, perturbed_image), axis = 1)
            return inputs
        
        else:
            image, emb = self.image_array[index]

            if image.shape[0] > self.limit_in:
                image = Image.fromarray(image)
                image = np.array(self.transform(image))

            if self.args.pattern_type == 'binary':
                image = Thresh(np.array([image.flatten()-0.5]))
            else:
                image = np.array([image.flatten()])

            emb = np.expand_dims(np.array(emb), axis = 0)
            inputs['image'] = np.concatenate((emb, image), axis = 1)
            return inputs
