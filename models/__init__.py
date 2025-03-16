from .ELBO.elbo import ELBO
from .VAE.vae import VAE

def load_model(**cfg):
    if cfg['name'] == 'VAE':
        return VAE(dims=cfg['dims'],
                   latent=cfg['latent'],
                   img_size=cfg['img_size'])

def load_loss_fn(**cfg):
    if cfg['name'] == 'ELBO':
        return ELBO(alpha=cfg['alpha'])