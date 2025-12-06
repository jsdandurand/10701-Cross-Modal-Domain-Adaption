import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Optional
import os
import json
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans


def load_hyperparameters(config_path: Optional[str]) -> Dict:
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"Loaded config from {config_path}")
        return config
    else:
        return {
            'lr': 5e-5,
            'batch_size': 32,
            'weight_decay': 1e-4,
            'optimizer': 'adamw',
            'scheduler': 'cosine',
            'dropout': 0.2,
            'alignment_epochs': 20,  # TODO: might need more epochs for better alignment
            'alignment_lr': 1e-4,
            'alignment_batch_size': 16,
            'alignment_distance': 'mse',
            'task_epochs': 50,
            'finetune_mode': 'fpt',
            'lora_rank': 16,
            'lora_alpha': 16,
            'lora_dropout': 0.1,
            'max_grad_norm': 1.0
        }


def create_optimizer(model, config: Dict):
    params = filter(lambda p: p.requires_grad, model.parameters())
    if config['optimizer'] == 'adam':
        return torch.optim.Adam(params, lr=config['lr'], weight_decay=config['weight_decay'])
    else:
        return torch.optim.AdamW(params, lr=config['lr'], weight_decay=config['weight_decay'])


def create_scheduler(optimizer, config: Dict, num_epochs: int):
    if config['scheduler'] == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    elif config['scheduler'] == 'linear':
        return torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=num_epochs)
    else:
        return None


def sample_text_embeddings(llm_model, num_samples=50000, device=None, num_classes=10, infer_labels=True):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
    llm_model.eval()
    embed_layer = llm_model.get_input_embeddings()
    vocab_size = embed_layer.weight.size(0)
    
    with torch.no_grad():
        token_ids = torch.randint(0, vocab_size, (num_samples, 64), device=device)
        token_embeds = embed_layer(token_ids)
        
        if infer_labels:
            embeds_flat = token_embeds.mean(dim=1).cpu().numpy()
            
            # use minibatch kmeans for large datasets, regular kmeans is fine for smaller ones
            if num_samples <= 10000:
                kmeans = KMeans(n_clusters=num_classes, random_state=42, n_init=10)
            else:
                kmeans = MiniBatchKMeans(n_clusters=num_classes, random_state=42, batch_size=10000, n_init=10)
            
            labels_np = kmeans.fit_predict(embeds_flat)
            labels = torch.from_numpy(labels_np).long().to(device)
        else:
            # just random labels if not inferring
            labels = torch.randint(0, num_classes, (num_samples,), device=device)
    
    return token_embeds, labels


class MSEDistance(nn.Module):
    def forward(self, x, y):
        if x.dim() == 3:
            x = x.mean(dim=1)
        if y.dim() == 3:
            y = y.mean(dim=1)
        return F.mse_loss(x, y)


class MMDDistance(nn.Module):
    def __init__(self, kernel_mul=2.0, kernel_num=5):
        super().__init__()
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num
    
    def gaussian_kernel(self, source, target):
        # compute pairwise distances for all samples
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1) ** 2).sum(2)
        # adaptive bandwidth based on median distance
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= self.kernel_mul ** (self.kernel_num // 2)
        # multi-scale kernels
        bandwidth_list = [bandwidth * (self.kernel_mul ** i) for i in range(self.kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)
    
    def forward(self, x, y):
        if x.dim() == 3:
            x = x.mean(dim=1)
        if y.dim() == 3:
            y = y.mean(dim=1)
        
        n_x = x.size(0)
        n_y = y.size(0)
        
        kernels = self.gaussian_kernel(x, y)
        
        XX = kernels[:n_x, :n_x] 
        YY = kernels[n_x:, n_x:] 
        XY = kernels[:n_x, n_x:] 
        YX = kernels[n_x:, :n_x]  
        loss = torch.mean(XX + YY - XY - YX)
        return loss
