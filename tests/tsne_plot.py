import sys, os
sys.path.append(os.getcwd())

from models.VAE.vae import VAE
from datasets.mnist_dataset import MNIST_Dataset

import torch
from torch.utils.data import DataLoader

from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    device = 'cuda:2'
    
    ds = MNIST_Dataset(mode='test')
    dl = DataLoader(ds, batch_size=32)
    
    model = VAE(dims=[1, 32, 64],
                latent=16,
                img_size=(1, 28, 28)).to(device)

    ckpt = torch.load("saved/vae.mnist.100epochs.pth", weights_only=False)
    model.load_state_dict(ckpt['model'])
    
    mu_li = []
    label_li = []
    
    for batch_idx, (x, label) in enumerate(dl, start=1):
        x = x.to(device)
        
        x_prime, mu, std = model(x)
        
        mu_li.append(mu)
        label_li.append(label)
    
    mu_data = torch.cat(mu_li, dim=0).cpu().detach().numpy()
    label_data = torch.cat(label_li, dim=0).cpu().detach().numpy()
    
    tsne_model = TSNE(n_components=2)
    x_embedded = tsne_model.fit_transform(mu_data)
    
    palette = sns.color_palette("bright", 10)
    print(label_data.shape)
    sns.scatterplot(x=x_embedded[:,0], y=x_embedded[:,1], hue=label_data, legend='full', palette=palette)

    plt.title(r"t-SNE: MNIST Test set, $\mu$ latent vector ($dim=16,\:n=1000$)")
    plt.tight_layout()
    plt.savefig('./assets/t-SNE_MNIST.jpg', dpi=500)

if __name__ == '__main__':
    main()