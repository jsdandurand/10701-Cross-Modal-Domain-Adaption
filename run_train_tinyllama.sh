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
# PREVIOUS 3-STAGE PIPELINE (COMMENTED OUT)
# ============================================

# # Stage 1: Baseline (no pretrained tokenizer)
# echo "Stage 1: Training baseline..."
# python -u train_tinyllama.py
# echo "Stage 1 complete."
# echo "=========================================="

# # Stage 2: Embedding Alignment
# echo "Stage 2: Training embedding alignment..."
# python -u train_stage2_align.py --distance mmd
# echo "Stage 2 complete."
# echo "=========================================="

# # Stage 3: With aligned tokenizer
# echo "Stage 3: Training with aligned tokenizer..."
# rm -f latest_checkpoint.pt
# python -u train_tinyllama.py --pretrained stage2_align_mse_model.pt
# echo "Stage 3 complete."
# echo "=========================================="

# ============================================
# FINAL 50-EPOCH TRAINING RUNS
# Using optimal hyperparameters from ASHA tuning
# ============================================

# Alternative Experiment
echo "=========================================="
echo "Final Training: LoRA NOALIGN (50 epochs)"
echo "=========================================="
START_TIME=$(date +%s)
python -u train.py --finetune_mode lora --config final_lora_noalign.json
END_TIME=$(date +%s)
echo "LoRA+Align Runtime: $((END_TIME - START_TIME)) seconds"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv

echo "=========================================="
echo "Final Training: FPT NOALIGN (50 epochs)"
echo "=========================================="
START_TIME=$(date +%s)
python -u train.py --do_alignment --finetune_mode fpt --config final_fpt_align.json
END_TIME=$(date +%s)
echo "FPT+Align Runtime: $((END_TIME - START_TIME)) seconds"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv

# # --- CONFIG 1: FPT + Alignment ---
# echo "=========================================="
# echo "Final Training: FPT + Alignment (50 epochs)"
# echo "=========================================="
# START_TIME=$(date +%s)
# python -u train.py --do_alignment --finetune_mode fpt --config final_fpt_align.json
# END_TIME=$(date +%s)
# echo "FPT+Align Runtime: $((END_TIME - START_TIME)) seconds"
# nvidia-smi --query-gpu=memory.used,memory.total --format=csv

# # --- CONFIG 2: LoRA + Alignment ---
# echo "=========================================="
# echo "Final Training: LoRA + Alignment (50 epochs)"
# echo "=========================================="
# START_TIME=$(date +%s)
# python -u train.py --do_alignment --finetune_mode lora --config final_lora_align.json
# END_TIME=$(date +%s)
# echo "LoRA+Align Runtime: $((END_TIME - START_TIME)) seconds"
# nvidia-smi --query-gpu=memory.used,memory.total --format=csv

# # --- CONFIG 3: Full Finetune + Alignment ---
# echo "=========================================="
# echo "Final Training: Full Finetune + Alignment (50 epochs)"
# echo "=========================================="
# START_TIME=$(date +%s)
# python -u train.py --do_alignment --finetune_mode full --config final_full_align.json
# END_TIME=$(date +%s)
# echo "Full+Align Runtime: $((END_TIME - START_TIME)) seconds"
# nvidia-smi --query-gpu=memory.used,memory.total --format=csv

# # --- CONFIG 4: Full Finetune (No Alignment) ---
# echo "=========================================="
# echo "Final Training: Full Finetune - No Alignment (50 epochs)"
# echo "=========================================="
# START_TIME=$(date +%s)
# python -u train.py --finetune_mode full --config final_full_noalign.json
# END_TIME=$(date +%s)
# echo "Full(NoAlign) Runtime: $((END_TIME - START_TIME)) seconds"
# nvidia-smi --query-gpu=memory.used,memory.total --format=csv

# echo "All final training runs complete!"

# echo "Job finished at $(date)"
