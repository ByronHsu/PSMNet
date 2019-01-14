import os
import torch
import torch.utils.data as data
import torch
import torchvision.transforms as transforms
import random
from PIL import Image
import preprocess 
import uslistfile as lt
import numpy as np

def default_loader(path):
    return Image.open(path).convert('RGB')

class myImageFloder(data.Dataset):
    def __init__(self, left, right, left_disparity, training, loader=default_loader):
        self.left = left
        self.right = right
        self.loader = loader
        self.training = training

    def __getitem__(self, index):
        left  = self.left[index]
        right = self.right[index]

        left_img = self.loader(left)
        right_img = self.loader(right)
        
        left_img_flip = np.fliplr(left_img)
        right_img_flip = np.fliplr(right_img)
        
        if self.training:  
           w, h = left_img.size
           processed = preprocess.get_transform(augment=False)  
           left_img   = processed(left_img)
           right_img  = processed(right_img)
           return left_img, right_img

    def __len__(self):
        return len(self.left)
