import numpy as np
import cv2
import os
from PIL import Image

import torch
from torch.utils.data import Dataset

from utils import perturb_pattern, Thresh

class Image_dataset(Dataset):
    def __init__(self, args, corrupt_flag = False):
        self.folder_path = os.path.join(os.getcwd(), args.folder_path)
        self.num_images = args.num_images

        # Corruption
        self.corrupt_flag = corrupt_flag
        self.perturb_percent = args.perturb_percent
        self.crop_percent = args.crop_percent
        self.corrupt_type = args.corrupt_type

        self.collect_to_array()

    def collect_to_array(self):
        self.image_array = []

        for i, name in enumerate(os.listdir(self.folder_path)):
            if i < self.num_images:
                image_path = os.path.join(self.folder_path, name)
                image = cv2.imread(image_path, 0)
                self.image_array.append(image)

    def __len__(self):
        return len(self.image_array)
    
    def __getitem__(self, index):
        inputs = {}
        if self.corrupt_flag == True:
            image = self.image_array[index]
            image = Thresh(np.array([image.flatten()-0.5]))
            perturbed_image = perturb_pattern(image)

            inputs['image'] = image
            inputs['perturbed'] = perturbed_image
            return inputs
        
        else:
            image = self.image_array[index]
            image = Thresh(np.array([image.flatten()-0.5]))
            
            inputs['image'] = image
            return inputs
