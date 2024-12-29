import os
import torch
import numpy as np
import re

def save_model_ckpt(model_name,
                    dataset,
                    current_epoch,
                    model,
                    save_path):
    
    model_name, dataset = model_name.lower(), dataset.lower()
    
    ckpt = {}
    ckpt['model'] = model.state_dict()
    ckpt['epochs'] = current_epoch
    
    # 이전 Epoch를 제거하기 위함
    prev_pths = os.listdir(save_path)
    
    for prev_pth in prev_pths:
        prev_parsed = prev_pth.split('.')
        if prev_parsed[-1] != "pth": # 확장자 파일이 아니면 건너뛰기
            continue
        
        prev_model = prev_parsed[0]
        prev_dataset = prev_parsed[1]
        prev_epoch = int("".join(re.findall(r'\d+', prev_parsed[2])))
        
        if prev_model == model_name and prev_dataset == dataset:
            if prev_epoch < current_epoch:
                # 현재 epoch 수보다 적으면 제거
                os.remove(os.path.join(save_path, prev_pth))
    
    state_dict_name = f'{model_name}.{dataset}.{current_epoch:03d}epochs.pth'
    
    try:
        torch.save(ckpt, os.path.join(save_path, state_dict_name))
        print(f"Save Model @epoch: {current_epoch}")
    except:
        print(f"Can\'t Save Model @epoch: {current_epoch}")


def save_loss_ckpt(model_name,
                   dataset,
                   train_loss,
                   save_path):
    
    model_name, dataset = model_name.lower(), dataset.lower()
    
    try:
        np.save(os.path.join(save_path, f'train_loss.{model_name}.{dataset}.npy'), np.array(train_loss))
        print('Save Train Loss')
    except:
        print('Can\'t Save Train Loss')