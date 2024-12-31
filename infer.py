import yaml
import argparse

from datasets import load_dataset
from models import load_model
from utils import *

import torch
import torch.distributions as dist
import matplotlib.pyplot as plt
import cv2

def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    
    # --num 
    # (1) if --method=='reconstruct' -> Dataset sample num 
    # (2) if --method=='generate' -> Figure save num 
    parser.add_argument('--num', type=int, default=0) 
    parser.add_argument('--method', type=str, default='generate')
    
    return parser

def main(cfg, args):
    print(f"=================[{cfg['expr']}]=================")
    
    # Device Setting
    device = None
    if cfg['device'] != 'cpu' and torch.cuda.is_available():
        device = cfg['device']
    else:
        device = 'cpu'
    
    # Load Dataset
    if(args.method == 'reconstruct'):
        data_cfg = cfg['data']
        test_ds = load_dataset(dataset=cfg['data']['dataset'],
                                mode=cfg['data']['mode'])
        print(f"Load Dataset {data_cfg['dataset']}")
    
    # Load Model
    model_cfg = cfg['model']
    model = load_model(model=model_cfg['name'],
                       latent_size=model_cfg['latent_size'],
                       x_size=tuple(model_cfg['x_size'])).to(device)
    ckpt = torch.load(os.path.join(cfg['save_path'], cfg['load_weights']),
                      weights_only=False)
    model.load_state_dict(ckpt['model'])
    print(f"Load Model {model_cfg['name']}")
    
    if(args.method == 'reconstruct'):
        # Load Sample
        x, _ = test_ds[args.num]
        x = x.unsqueeze(0) # (C, W, H) -> (B, C, W, H)
        
        # Infer
        x_prime, _, __ = model(x)
        
        # Reshape & Visualization
        x = x.reshape(28, 28, 1).detach().cpu().numpy() * 255.
        x_prime = x_prime.reshape(28, 28, 1).detach().cpu().numpy() * 255.
        
        cv2.imwrite('figures/original.jpg', x)
        cv2.imwrite(f'figures/reconstruction_{int(cfg['load_weights'][10:13]):03d}.jpg', x_prime)
    
    elif(args.method == 'generate'):
        # Random Sampling from Gaussian
        gaussian = dist.Normal(loc=torch.zeros(model_cfg['latent_size']),
                               scale=torch.ones(model_cfg['latent_size']))
        random_vector = gaussian.sample()
        
        # Generation
        x_prime = model.decoder(random_vector)
        
        # Reshape & Visualization
        x_prime = x_prime.reshape(28, 28, 1).detach().cpu().numpy() * 255.
        
        cv2.imwrite(f'figures/generation_{int(cfg['load_weights'][10:13]):03d}_{args.num:02d}.jpg', x_prime)
    
if __name__ == '__main__':
    with open('config.yaml') as f:
        cfg = yaml.full_load(f)
        
    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    args = parser.parse_args()
    
    main(cfg, args)