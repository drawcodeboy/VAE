from .ELBO.elbo import ELBO
from .VAE.vae import VAE

def load_model(model='VAE',
               latent_size=200,
               x_size=(1, 28, 28)):
    
    if model == 'VAE':
        return VAE(latent_size=latent_size,
                   x_size=x_size)

def load_loss_fn(loss_fn='ELBO',
                 latent_size=200):
    if loss_fn == 'ELBO':
        return ELBO(latent_size=latent_size)