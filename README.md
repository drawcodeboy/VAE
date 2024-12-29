# VAE Implementation with PyTorch

<p align="center">
    <img src="./figures/original.jpg" width="15%" height="15%">
    <img src="./figures/generation_010.jpg" width="15%" height="15%">
    <img src="./figures/generation_020.jpg" width="15%" height="15%">
    <img src="./figures/generation_030.jpg" width="15%" height="15%">
    <img src="./figures/generation_050.jpg" width="15%" height="15%">
    <img src="./figures/generation_100.jpg" width="15%" height="15%">
</p>

* Original > 10 Epochs > 20 Epochs > 30 Epochs > 50 Epochs > 100 Epochs

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

# Inference
python infer.py --num=12 # 12 is the sample at the top of this document.
```

## References
1. <a href="https://arxiv.org/pdf/1312.6114">Kingma, D. P. (2013). Auto-encoding variational bayes. <i>arXiv preprint arXiv:1312.6114.</i></a>