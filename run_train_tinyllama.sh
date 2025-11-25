#!/bin/bash
#SBATCH --job-name=tinyllama_vision
#SBATCH --output="/user_data/horaja/workspace/10701/logs/tinyllama_vision_%j.out"
#SBATCH --error="/user_data/horaja/workspace/10701/logs/tinyllama_vision_%j.err"
#SBATCH --time=12:00:00
#SBATCH --mem=32gb
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu

echo "Job started on $(hostname) at $(date)"
nvidia-smi

# ============================================
# ENVIRONMENT SETUP
# ============================================
echo "Setting up environment..."
module load cuda-12.1

# Conda environment name from setup_local_env.sh
ENV_NAME="10701"

# Initialize conda
eval "$(mamba shell hook --shell bash)"

# Activate the environment (assumes it's already created via setup_local_env.sh)
mamba activate $ENV_NAME

echo "Python executable: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# Set up MLflow tracking
export MLFLOW_TRACKING_URI="file:$(pwd)/mlruns"
echo "MLflow tracking URI: $MLFLOW_TRACKING_URI"

echo "Environment setup complete."
echo "=========================================="

# ============================================
# RUN THE 3-STAGE PIPELINE
# ============================================

# Stage 1: Baseline (no pretrained tokenizer)
echo "Stage 1: Training baseline..."
python -u train_tinyllama.py
echo "Stage 1 complete."
echo "=========================================="

# Stage 2: Embedding Alignment
echo "Stage 2: Training embedding alignment..."
python -u train_stage2_align.py --distance mmd
echo "Stage 2 complete."
echo "=========================================="

# Stage 3: With aligned tokenizer
echo "Stage 3: Training with aligned tokenizer..."
rm -f latest_checkpoint.pt
python -u train_tinyllama.py --pretrained stage2_align_mse_model.pt
echo "Stage 3 complete."
echo "=========================================="

echo "Job finished at $(date)"
