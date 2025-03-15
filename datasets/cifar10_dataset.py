import torch
from torch.utils.data import dataset
from torchvision.datasets import CIFAR10
import numpy as np

class CIFAR10_Dataset():
    def __init__(self,
                 root="data/", # data 하위에 MNIST 디렉터리 생김
                 download=True, # root로 지정한 위치에 없으면 다운, 있으면 패스
                 mode='train'): 
        
        if mode not in ['train', 'test']:
            raise AssertionError(f"mode 확인: [{mode}]으로 입력 됨")
        
        self.data = CIFAR10(root=root,
                            download=download)
        
        self.data=list(self.data)
        
        train_size = 10000
        test_size = 500
        
        if mode == 'train':
            self.data = self.data[:train_size]
        elif mode == 'test':
            self.data = self.data[train_size:train_size+test_size]
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image, label = self.data[idx]
        
        image = np.array(image.getdata()).astype(np.float32)
        image /= 255.
        image = torch.tensor(image, dtype=torch.float32).reshape(3, 32, 32)
        
        label = torch.tensor(label, dtype=torch.int64)
        
        return image, label

if __name__ == '__main__':
    ds = CIFAR10_Dataset()
    image, label = ds[1000]
    
    import matplotlib.pyplot as plt
    image = image.detach().cpu().numpy().reshape(32, 32, 3)
    
    plt.imshow(image)
    plt.savefig('test2.png')