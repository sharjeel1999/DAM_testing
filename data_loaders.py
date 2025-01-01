import numpy as np
import cv2
import os
from PIL import Image

import torch
from torch.utils.data import Dataset


class Full_dataset(Dataset):
    def __init__(self, folder_path, num_images):
        self.folder_path = folder_path
        self.num_images = num_images

        self.collect_to_array()

    def collect_to_array(self):
        self.image_array = []

        for i, name in enumerate(os.listdir(self.folder_path)):
            if i <= self.num_images:
                image_path = os.path.join(self.folder_path, name)
                image = cv2.imread(image_path)
                self.image_array.append(image)

    def __len__(self):
        return len(self.image_array)
    
    def __getitem__(self, index):
        image = self.image_array[index]

        return image