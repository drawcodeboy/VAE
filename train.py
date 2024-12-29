import yaml
from datasets import load_dataset
from models import load_model, load_loss_fn

def main(cfg):
    print(cfg['expr'])
    
    device = None
    if cfg['device'] != 'cpu' and torch.cuda.is_available():
        device = cfg['device']
    else:
        device = 'cpu'
    
    # Load Dataset
    data_cfg = cfg['data']
    train_ds = load_dataset(dataset=cfg['data']['dataset'],
                            mode=cfg['data']['mode'])
    print(f"Load Dataset {data_cfg['dataset']}")
    
    # Load Model
    model_cfg = cfg['model']
    model = load_model(model=model_cfg['name'],
                       latent_size=model_cfg['latent_size'],
                       enc_conv_in_channels=model_cfg['enc_conv_in_channels'],
                       enc_conv_out_channels=model_cfg['enc_conv_out_channels'],
                       x_size=tuple(model_cfg['x_size'])).to(device)
    print(f"Load Model {model_cfg['name']}")
    
    # Load Loss function
    loss_fn = load_loss_fn(loss_fn=cfg['loss_fn'],
                           latent_size=model_cfg['latent_size'])
    print(f"Load Loss function {cfg['loss_fn']}")
    
    
    
if __name__ == '__main__':
    with open('config.yaml') as f:
        cfg = yaml.full_load(f)
    
    main(cfg)