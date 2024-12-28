from .ELBO.elbo import ELBO
from .VAE.vae import VAE

def load_model(model='VAE',
               latent_size=10,
               enc_conv_in_channels=1,
               enc_conv_out_channels=16,
               x_size=(1, 28, 28)):
    
    if model == 'VAE':
        return VAE(latent_size=latent_size,
                   enc_conv_in_channels=enc_conv_in_channels,
                   enc_conv_out_channels=enc_conv_out_channels,
                   x_size=x_size)

def load_loss_fn(loss_fn='ELBO',
                 latent_size=10):
    if loss_fn == 'ELBO':
        return ELBO(latent_size=latent_size)