# VAE Implementation with PyTorch

### (1) Reconstruction: Original > 10 > 20 > 30 > 50> 100 (Epochs)
<p align="center">
    <img src="./figures/original.jpg" width="15%" height="15%">
    <img src="./figures/reconstruction_010.jpg" width="15%" height="15%">
    <img src="./figures/reconstruction_020.jpg" width="15%" height="15%">
    <img src="./figures/reconstruction_030.jpg" width="15%" height="15%">
    <img src="./figures/reconstruction_050.jpg" width="15%" height="15%">
    <img src="./figures/reconstruction_100.jpg" width="15%" height="15%">
</p>

### (2) Generation from random vector 
* $x \sim N(0, 1)$

<p align="center">
    <img src="./figures/generation_200_01.jpg" width="20%" height="20%">
    <img src="./figures/generation_200_02.jpg" width="20%" height="20%">
    <img src="./figures/generation_200_03.jpg" width="20%" height="20%">
    <img src="./figures/generation_200_04.jpg" width="20%" height="20%">
</p>

## Setting
```
# Download
git clone https://github.com/drawcodeboy/VAE.git

# Virtual Environment
python venv -m .venv
.venv\Scripts\activate # Windows command

# Install packages
pip install -r requirements.txt

# !!when you train or inference, you should check config.yaml!!

# training
python train.py

# Inference (1) : Reconstruction
python infer.py --method=reconstruct --num=12 # Dataset sample num 

# Inference (2) : Generate
python infer.py --method=generate --num=1 # Figure save num
```

## References
1. <a href="https://arxiv.org/pdf/1312.6114">Kingma, D. P. (2013). Auto-encoding variational bayes. <i>arXiv preprint arXiv:1312.6114.</i></a>