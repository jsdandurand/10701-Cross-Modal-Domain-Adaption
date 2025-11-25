"""
train_stage2_align.py
Stage 2: Align embedder with LLM's embedding distribution
Implements multiple distance metrics for alignment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import AutoModel, AutoConfig, AutoTokenizer
import mlflow
import os
import argparse
from tqdm import tqdm
import numpy as np


# ===================== Config =====================
# Model selection: "gpt2" (117M) or "nickypro/tinyllama-110M" (~110M)
MODEL_NAME = "nickypro/tinyllama-110M"  # Options: "gpt2" or "nickypro/tinyllama-110M"
BATCH_SIZE = 16
EPOCHS = 20
LR = 1e-4
PATCH_SIZE = 4  # (32/4 = 8) yielding patch number 8x8 = 64 
EMBED_DIM = AutoConfig.from_pretrained(MODEL_NAME).hidden_size  # Automatically determined from model
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
NUM_SAMPLES = 50000  # Number of text samples to use for alignment


# ===================== Distance Metrics =====================
class DistanceMetric(nn.Module):
    """Base class for distance metrics"""
    def forward(self, x, y):
        raise NotImplementedError


class MSEDistance(DistanceMetric):
    """Mean Squared Error (L2) distance"""
    def forward(self, x, y):
        # x: [B, seq_len, embed_dim] or [B, embed_dim]
        # y: [B, seq_len, embed_dim] or [B, embed_dim]
        if x.dim() == 3:
            x = x.mean(dim=1)  # Average over sequence length
        if y.dim() == 3:
            y = y.mean(dim=1)
        return F.mse_loss(x, y)


class CosineDistance(DistanceMetric):
    """Cosine distance (1 - cosine similarity)"""
    def forward(self, x, y):
        # x: [B, seq_len, embed_dim] or [B, embed_dim]
        # y: [B, seq_len, embed_dim] or [B, embed_dim]
        if x.dim() == 3:
            x = x.mean(dim=1)  # Average over sequence length
        if y.dim() == 3:
            y = y.mean(dim=1)
        # Normalize
        x_norm = F.normalize(x, p=2, dim=1)
        y_norm = F.normalize(y, p=2, dim=1)
        # Cosine similarity
        cosine_sim = (x_norm * y_norm).sum(dim=1).mean()
        # Return distance (1 - similarity)
        return 1 - cosine_sim


class KLDistance(DistanceMetric):
    """KL Divergence (for distribution matching)"""
    def forward(self, x, y):
        # x: [B, seq_len, embed_dim] or [B, embed_dim]
        # y: [B, seq_len, embed_dim] or [B, embed_dim]
        if x.dim() == 3:
            x = x.mean(dim=1)  # Average over sequence length
        if y.dim() == 3:
            y = y.mean(dim=1)
        
        # Convert to probability distributions using softmax
        x_prob = F.softmax(x, dim=1)
        y_prob = F.softmax(y, dim=1)
        
        # KL divergence: KL(x || y)
        kl_div = F.kl_div(F.log_softmax(x, dim=1), y_prob, reduction='batchmean')
        return kl_div


class MMDDistance(DistanceMetric):
    """Maximum Mean Discrepancy (MMD) with RBF kernel"""
    def __init__(self, kernel_mul=2.0, kernel_num=5):
        super().__init__()
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num
    
    def gaussian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        """Compute Gaussian kernel for MMD"""
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1) ** 2).sum(2)
        
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)
    
    def forward(self, x, y):
        # x: [B, seq_len, embed_dim] or [B, embed_dim]
        # y: [B, seq_len, embed_dim] or [B, embed_dim]
        if x.dim() == 3:
            x = x.mean(dim=1)  # Average over sequence length
        if y.dim() == 3:
            y = y.mean(dim=1)
        
        # Compute MMD
        batch_size = x.size(0)
        xx = self.gaussian_kernel(x, x, self.kernel_mul, self.kernel_num)
        yy = self.gaussian_kernel(y, y, self.kernel_mul, self.kernel_num)
        xy = self.gaussian_kernel(x, y, self.kernel_mul, self.kernel_num)
        
        mmd = xx.mean() + yy.mean() - 2 * xy.mean()
        return mmd


# ===================== Image Embedder =====================
class ImageTokenizer(nn.Module):
    """Image tokenizer that will be aligned with LLM embeddings"""
    def __init__(self, patch_size=PATCH_SIZE, in_channels=3, embed_dim=EMBED_DIM):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(in_channels * patch_size * patch_size, embed_dim)
    
    def forward(self, x):
        B, C, H, W = x.shape  # (B, C, H, W) -> (B, num_patches, embed_dim)
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(B, C, -1, self.patch_size * self.patch_size)
        patches = patches.permute(0, 2, 1, 3).reshape(B, -1, C * self.patch_size * self.patch_size)
        return self.proj(patches)  # [B, num_patches, embed_dim]


# ===================== Alignment Model =====================
class AlignmentModel(nn.Module):
    """Model for aligning image embeddings with LLM embeddings"""
    def __init__(self, embed_dim=EMBED_DIM):
        super().__init__()
        self.image_tokenizer = ImageTokenizer(embed_dim=embed_dim)
        # Learnable positional embedding
        self.num_patches = (32 // 4) ** 2  # CIFAR-10: 32x32 image, patch=4 -> 8x8=64 patches
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
    
    def forward(self, images):
        # Get image embeddings
        tokens = self.image_tokenizer(images)  # [B, num_patches, embed_dim]
        tokens = tokens + self.pos_embed
        return tokens  # [B, num_patches, embed_dim]


# ===================== Text Embedding Sampler =====================
def sample_text_embeddings(llm_model, tokenizer, num_samples=NUM_SAMPLES, device=DEVICE):
    """
    Sample text embeddings from the LLM to use as target distribution.
    Uses random tokens from the vocabulary.
    """
    llm_model.eval()
    embeddings = []
    
    # Get the embedding layer
    embed_layer = llm_model.get_input_embeddings()
    vocab_size = embed_layer.weight.size(0)
    print(f"Vocab Size of Pretrained Model is {vocab_size}")
    # Sample random token IDs
    with torch.no_grad():
        # Sample random tokens
        token_ids = torch.randint(0, vocab_size, (num_samples, 64), device=device)  # 64 tokens per sample
        
        # Get embeddings
        token_embeds = embed_layer(token_ids)  # [num_samples, 64, embed_dim]
        embeddings.append(token_embeds)
    
    # Concatenate all embeddings
    all_embeddings = torch.cat(embeddings, dim=0)  # [num_samples, 64, embed_dim]
    return all_embeddings


# ===================== Training Loop =====================
def train_one_epoch(model, image_loader, target_embeddings, optimizer, distance_metric, device=DEVICE):
    model.train()
    total_loss = 0
    num_batches = 0
    
    for images, _ in image_loader:  # images, labels (we don't need labels)
        images = images.to(device)
        batch_size = images.size(0)
        
        # Randomly sample target embeddings for this batch
        indices = torch.randint(0, target_embeddings.size(0), (batch_size,), device=device)
        target_embeds = target_embeddings[indices].to(device)  # [B, seq_len, embed_dim]
        
        optimizer.zero_grad()
        
        # Get image embeddings
        image_embeds = model(images)  # [B, num_patches, embed_dim]
        
        # Compute distance
        loss = distance_metric(image_embeds, target_embeds)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0


# ===================== Main =====================
def main():
    parser = argparse.ArgumentParser(description='Stage 2: Align embedder with LLM embeddings')
    parser.add_argument('--distance', type=str, default='mse', 
                       choices=['mse', 'cosine', 'kl', 'mmd', 'all'],
                       help='Distance metric to use (or "all" to train with all metrics)')
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=LR, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size')
    args = parser.parse_args()
    
    # Initialize MLflow
    mlflow.set_tracking_uri("file:./mlruns")
    # Set experiment name based on model and distance metric
    model_short_name = MODEL_NAME.split("/")[-1].split("-")[0] if "/" in MODEL_NAME else MODEL_NAME
    experiment_name = f"Stage2_Align_{model_short_name}_{args.distance.upper()}"
    mlflow.set_experiment(experiment_name)
    
    print(f"Using model: {MODEL_NAME}")
    print(f"Embedding dimension: {EMBED_DIM}")
    
    # Load LLM model
    print(">>> Loading LLM model...")
    llm_model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    llm_model.eval()
    
    # Freeze LLM
    for param in llm_model.parameters():
        param.requires_grad = False
    
    # Sample text embeddings for alignment
    print(">>> Sampling text embeddings from LLM...")
    target_embeddings = sample_text_embeddings(llm_model, tokenizer, num_samples=NUM_SAMPLES, device=DEVICE)
    print(f"Sampled {target_embeddings.shape[0]} text embedding samples")
    
    # Load image dataset
    print(">>> Loading image dataset...")
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    
    # Distance metrics
    distance_metrics = {
        'mse': MSEDistance(),
        'cosine': CosineDistance(),
        'kl': KLDistance(),
        'mmd': MMDDistance()
    }
    
    # Determine which metrics to use
    if args.distance == 'all':
        metrics_to_use = list(distance_metrics.keys())
    else:
        metrics_to_use = [args.distance]
    
    # Train with each metric
    for metric_name in metrics_to_use:
        print(f"\n>>> Training with {metric_name.upper()} distance metric...")
        
        # Create model
        model = AlignmentModel(embed_dim=EMBED_DIM).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        
        # Get distance metric
        distance_metric = distance_metrics[metric_name].to(DEVICE)
        
        # Training loop
        with mlflow.start_run(run_name=f"align_{metric_name}"):
            mlflow.log_param("distance_metric", metric_name)
            mlflow.log_param("learning_rate", args.lr)
            mlflow.log_param("batch_size", args.batch_size)
            mlflow.log_param("epochs", args.epochs)
            
            start_epoch = 0
            ckpt_path = f"stage2_align_{metric_name}_checkpoint.pt"
            
            # Resume from checkpoint if exists
            if os.path.exists(ckpt_path):
                ckpt = torch.load(ckpt_path, map_location=DEVICE)
                model.load_state_dict(ckpt["model_state"])
                optimizer.load_state_dict(ckpt["optimizer_state"])
                start_epoch = ckpt["epoch"] + 1
                print(f"Resumed from epoch {start_epoch}")
            
            for epoch in range(start_epoch, args.epochs):
                loss = train_one_epoch(model, train_loader, target_embeddings, 
                                      optimizer, distance_metric, device=DEVICE)
                
                mlflow.log_metric("alignment_loss", loss, step=epoch)
                print(f"[Epoch {epoch+1}/{args.epochs}] Alignment Loss ({metric_name}): {loss:.6f}")
                
                # Save checkpoint
                ckpt = {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "distance_metric": metric_name
                }
                torch.save(ckpt, ckpt_path)
            
            # Save final model
            final_model_path = f"stage2_align_{metric_name}_model.pt"
            torch.save({
                "image_tokenizer": model.image_tokenizer.state_dict(),
                "pos_embed": model.pos_embed,
                "distance_metric": metric_name
            }, final_model_path)
            mlflow.log_artifact(final_model_path)
            print(f"Saved model to {final_model_path}")
    
    print("\n>>> Stage 2 alignment complete!")


if __name__ == "__main__":
    main()

