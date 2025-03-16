# VAE Implementation with PyTorch

## Description
* Variational Autoencoder

## Preview
### (1) Reconstruction: Original > 10 > 20 > 30 > 100 (Epochs)
<p align="center">
    <img src="./figures/original.jpg" width="15%" height="15%">
    <img src="./figures/reconstruction_010.jpg" width="15%" height="15%">
    <img src="./figures/reconstruction_020.jpg" width="15%" height="15%">
    <img src="./figures/reconstruction_030.jpg" width="15%" height="15%">
    <img src="./figures/reconstruction_100.jpg" width="15%" height="15%">
</p>

### (2) Generation from random vector 

$$
p_{\theta}(x\mid z), \\ \\ z \sim N(0, 1)
$$

<p align="center">
    <img src="./figures/generation_200_01.jpg" width="20%" height="20%">
    <img src="./figures/generation_200_02.jpg" width="20%" height="20%">
    <img src="./figures/generation_200_03.jpg" width="20%" height="20%">
    <img src="./figures/generation_200_04.jpg" width="20%" height="20%">
</p>

## Installation
```
# Docker (Optional)
docker pull ubuntu:22.04
docker run -itd --gpus=all --shm-size=16G --name=vae ubuntu:22.04 /bin/bash

apt-get update
apt-get install sudo
sudo apt-get install git
sudo apt-get install gdown # For CelebA Dataset
sudo apt-get install unzip
sudo apt-get install -y libgl1-mesa-glx # OpenCV
sudo apt-get install -y libglib2.0-0 # OpenCV

# Download
git clone https://github.com/drawcodeboy/VAE.git
cd VAE

# Download CelebA
gdown --id 1m8-EBPgi5MRubrm6iQjafK2QMHDBMSfJ --output data/
unzip data/celeba.zip -d data/
unzip data/celeba/img_align_celeba.zip -d data/celeba/

# Install Python
sudo apt-get install python3
sudo apt-get install python3-venv

# Virtual Environment
python3 venv -m .venv
source .venv/bin/activate

# Install packages
pip install -r requirements.txt

# !!when you train or inference, you should check config.yaml!!

# training
python train.py
```

## References
1. <i><b>VAE</b></i>: <a href="https://arxiv.org/pdf/1312.6114">Kingma, D. P. (2013). Auto-encoding variational bayes. <i>arXiv preprint arXiv:1312.6114.</i></a>
