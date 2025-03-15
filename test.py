import yaml
import argparse

from datasets import load_dataset
from models import load_model
from utils import *

import torch
import torch.distributions as dist
import matplotlib.pyplot as plt
from einops import rearrange

def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    
    parser.add_argument('--config', type=str, default='vae.mnist')
    
    return parser

def main(cfg):
    print(f"=================[{cfg['expr']}]=================")
    
    # Device Setting
    device = None
    if cfg['device'] != 'cpu' and torch.cuda.is_available():
        device = cfg['device']
    else:
        device = 'cpu'
    
    # Load Model
    model_cfg = cfg['model']
    model = load_model(**model_cfg).to(device)
    ckpt = torch.load(os.path.join(cfg['save_path'], cfg['load_weights']),
                      weights_only=False)
    model.load_state_dict(ckpt['model'])
    print(f"Load Model {model_cfg['name']}")
    
    # Random Sampling from Gaussian
    gaussian = dist.Normal(loc=torch.zeros((1, 64, 7, 7)),
                           scale=torch.ones((1, 64, 7, 7)))
    random_vector = gaussian.sample().to(device)
    
    # Generation
    x_prime = model.decoder(random_vector)
    
    # Reshape & Visualization
    x_prime = rearrange(x_prime, '1 c h w -> h w c').detach().cpu().numpy() * 255.
    
    plt.imshow(x_prime, cmap='gray')
    plt.savefig('test.png')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    args = parser.parse_args()
    
    with open(f'configs/test.{args.config}.yaml') as f:
        cfg = yaml.full_load(f)
        
    main(cfg)