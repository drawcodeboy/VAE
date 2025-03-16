import torch
from torch.utils.data import dataset

import os
import cv2
import numpy as np

class CelebA_Dataset():
    def __init__(self,
                 root='data/celeba/img_align_celeba/',
                 mode='train'):
        self.root = root
        self.mode = mode
        
        self.data_li = []
        
        self._check()
        
        if mode == 'train':
            self.data_li = self.data_li[:5000]
        else:
            self.data_li = self.data_li[5000:5500]
        
    def __len__(self):
        return len(self.data_li)
    
    def __getitem__(self, idx):
        img = cv2.imread(self.data_li[idx], cv2.IMREAD_COLOR)

        # Get Center (128, 128, 3)
        h, w, c = img.shape
        h_start = (h - 128) // 2
        w_start = (w - 128) // 2
        img = img[h_start:h_start+128, w_start:w_start+128, :]
        
        img = np.array(img).reshape(3, 128, 128) / 255.
        img = torch.tensor(img, dtype=torch.float32)
        
        label = 1
        
        return img, label
    
    def _check(self):
        print(len(os.listdir(self.root)))
        for filename in os.listdir(self.root):
            self.data_li.append(os.path.join(self.root, filename))