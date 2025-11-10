"""
train_baseline.py
Baseline implementation for Cross-Modal GPT-2 (Stage 1)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import AutoModel, AutoConfig, AutoTokenizer
import mlflow
import os
from tqdm import tqdm

# ===================== Config =====================
# Model selection: "gpt2" (117M) or "nickypro/tinyllama-110M" (~110M)
MODEL_NAME = "nickypro/tinyllama-110M"  # Options: "gpt2" or "nickypro/tinyllama-110M"
BATCH_SIZE = 64
EPOCHS = 50
LR = 1e-3  # Lower learning rate for better convergence
PATCH_SIZE = 4  # (32/4 = 8) yielding patch number 8x8 = 64 
EMBED_DIM = AutoConfig.from_pretrained(MODEL_NAME).hidden_size  # Automatically determined from model
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
NUM_CLASSES = 10
PRETRAINED_TOKENIZER_PATH = "stage2_align_mse_model.pt"  # Path to pretrained tokenizer from stage2 (e.g., "stage2_align_mse_model.pt")
CLASSIFIER_DROPOUT = 0.1  # Dropout rate for classification head

# ===================== Data Pipeline =====================
# Data augmentation for training
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomRotation(20),
    #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# No augmentation for testing
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

trainset = datasets.CIFAR10(root="./data", train=True, download=True, transform=train_transform)
testset = datasets.CIFAR10(root="./data", train=False, download=True, transform=test_transform)
train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

# Patchify: image 32x32 --> (num_patches, patch_dim)
class ImageTokenizer(nn.Module):
    def __init__(self, patch_size=PATCH_SIZE, in_channels=3, embed_dim=EMBED_DIM):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(in_channels * patch_size * patch_size, embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape # (B, C, H, W) -> (B, num_patches, patch_dim)
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(B, C, -1, self.patch_size*self.patch_size)
        patches = patches.permute(0, 2, 1, 3).reshape(B, -1, C * self.patch_size * self.patch_size)
        return self.proj(patches)  # [B, num_patches, embed_dim]


# ======================== Classification Head ========================
class ClassificationHead(nn.Module):
    def __init__(self, embed_dim, num_classes, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # Simple linear classifier (as used in the paper)
        self.classifier = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x):
        x = self.dropout(x)
        return self.classifier(x)


# ======================== Model ========================
class CrossModalGPT2(nn.Module):
    def __init__(self, base_model_name=MODEL_NAME, embed_dim=EMBED_DIM, num_classes=NUM_CLASSES, 
                 pretrained_tokenizer_path=None, in_channels=3):
        super().__init__()
        self.llm = AutoModel.from_pretrained(base_model_name)
        for param in self.llm.parameters():
            param.requires_grad = False  # frozen backbone

        # Unfreeze layer norm parameters (affine scale and bias)
        # GPT-2 has 2 layer norms per block (ln_1 before attention, ln_2 before feedforward)
        # Each has weight (scale) and bias parameters: 4 × ndim × nlayers = 4 × 768 × 12 = 36,864 params
        for name, param in self.llm.named_parameters():
            if "norm" in name.lower():
                param.requires_grad = True # unfreeze layer norm 

        # Patch Embedding
        self.tokenizer = ImageTokenizer(embed_dim=embed_dim, in_channels=in_channels)

        # Learnable positional embedding
        self.num_patches = (32 // 4) ** 2  # CIFAR-10: 32x32 image, patch=4 -> 8x8=64 patches
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # Load pretrained tokenizer from stage2 if provided
        if pretrained_tokenizer_path is not None and os.path.exists(pretrained_tokenizer_path):
            print(f"Loading pretrained tokenizer from {pretrained_tokenizer_path}")
            checkpoint = torch.load(pretrained_tokenizer_path, map_location=DEVICE)
            
            # Load image tokenizer state dict
            if "image_tokenizer" in checkpoint:
                self.tokenizer.load_state_dict(checkpoint["image_tokenizer"])
                # Ensure loaded parameters are trainable
                for param in self.tokenizer.parameters():
                    param.requires_grad = True
                print("Loaded pretrained image tokenizer (trainable)")
            else:
                print("Warning: 'image_tokenizer' not found in checkpoint, using random initialization")
            
            # Load positional embedding
            if "pos_embed" in checkpoint:
                self.pos_embed.data = checkpoint["pos_embed"].data
                # Ensure loaded positional embedding is trainable
                self.pos_embed.requires_grad = True
                print("Loaded pretrained positional embedding (trainable)")
            else:
                print("Warning: 'pos_embed' not found in checkpoint, using random initialization")
        elif pretrained_tokenizer_path is not None:
            print(f"Warning: Pretrained tokenizer path {pretrained_tokenizer_path} does not exist, using random initialization")

        # Classification head
        self.cls_head = ClassificationHead(embed_dim, num_classes, dropout=CLASSIFIER_DROPOUT)

        # Parameters finetuned: 
        # 1) input embedding, 
        # 2) layer norm, 
        # 3) positional embedding, 
        # 4) output layer

    def forward(self, x):
        tokens = self.tokenizer(x) 
        tokens = tokens + self.pos_embed

        outputs = self.llm(inputs_embeds=tokens)
        
        # Use mean pooling over all tokens instead of last token
        # Shape: [B, num_patches, embed_dim] -> [B, embed_dim]
        pooled_hidden = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
        return self.cls_head(pooled_hidden)


# ===================== Training Loop =====================
def train_one_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0

    for imgs, labels in tqdm(dataloader, desc="Training", leave=False):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        total_correct += (logits.argmax(1) == labels).sum().item()
        total_samples += imgs.size(0)

    return total_loss / total_samples, total_correct / total_samples

# ===================== Testing =====================
def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss, total_correct, total_samples = 0, 0, 0

    with torch.no_grad():
        for imgs, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            logits = model(imgs)
            loss = criterion(logits, labels)

            total_loss += loss.item() * imgs.size(0)
            total_correct += (logits.argmax(1) == labels).sum().item()
            total_samples += imgs.size(0)

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc


# ===================== Main =====================
def main():
    print(f"Using model: {MODEL_NAME}")
    print(f"Embedding dimension: {EMBED_DIM}")
    
    mlflow.set_tracking_uri("file:./mlruns")
    # Set experiment name based on model
    model_short_name = MODEL_NAME.split("/")[-1].split("-")[0] if "/" in MODEL_NAME else MODEL_NAME
    mlflow.set_experiment(f"CrossModal_{model_short_name}_Baseline")

    # Determine input channels (default to 3 for RGB)
    # Check if USE_GRAYSCALE flag exists in globals
    use_grayscale = globals().get('USE_GRAYSCALE', False)
    in_channels = 1 if use_grayscale else 3
    
    # Initialize model with optional pretrained tokenizer
    model = CrossModalGPT2(
        pretrained_tokenizer_path=PRETRAINED_TOKENIZER_PATH,
        in_channels=in_channels
    ).to(DEVICE)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=1e-4)
    # Add learning rate scheduler for better convergence
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()
    
    print("====== Trainable Parameters =======")
    for name, p in model.named_parameters():
        if p.requires_grad:
            print(name, p.numel())
    print("====================================")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of Trainable params: {trainable_params} / {total_params} = {trainable_params/total_params:.4f}")
    
    with mlflow.start_run():
        start_epoch = 0
        latest_ckpt = "latest_checkpoint.pt"

        if os.path.exists(latest_ckpt):
            ckpt = torch.load(latest_ckpt, map_location=DEVICE)
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            if "scheduler_state" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler_state"])
            start_epoch = ckpt["epoch"] + 1
            print(f"Resumed at epoch {start_epoch}")

        for epoch in range(start_epoch, EPOCHS):
            # ---------- Training ----------
            loss, acc = train_one_epoch(model, train_loader, optimizer, criterion)
            mlflow.log_metric("train_loss", loss, step=epoch)
            mlflow.log_metric("train_acc", acc, step=epoch)
            print(f"[Epoch {epoch+1}] Train Loss={loss:.4f}, Train Acc={acc:.4f}")

            # ---------- Evaluation ----------
            test_loss, test_acc = evaluate(model, test_loader, criterion)
            mlflow.log_metric("test_loss", test_loss, step=epoch)
            mlflow.log_metric("test_acc", test_acc, step=epoch)
            print(f"[Epoch {epoch+1}] Test Loss={test_loss:.4f}, Test Acc={test_acc:.4f}")

            # ---------- Update learning rate ----------
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            mlflow.log_metric("learning_rate", current_lr, step=epoch)

            # ---------- Save checkpoint ----------
            ckpt = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict()
            }
            torch.save(ckpt, latest_ckpt)

            if (epoch + 1) % 5 == 0: 
                torch.save(ckpt, f"checkpoint_epoch_{epoch+1}.pt")

        # ---------- Save final model ----------
        torch.save(model.state_dict(), "baseline_model.pt")
        mlflow.log_artifact("baseline_model.pt")


if __name__ == "__main__":
    main()
    print("Training complete!")
