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
# ASHA HYPERPARAMETER TUNING FOR ALL CONFIGS
# ============================================

# 1. FPT + Alignment
echo "=========================================="
echo "HP Tuning: FPT + Alignment"
echo "=========================================="
python -u tune_hp.py --do_alignment --finetune_mode fpt
mv best_hyperparameters.json best_hp_fpt_align.json 2>/dev/null || true
mv best_hyperparameters_summary.json best_hp_fpt_align_summary.json 2>/dev/null || true
echo "FPT + Alignment HP tuning complete."

# 2. LoRA + Alignment
echo "=========================================="
echo "HP Tuning: LoRA + Alignment"
echo "=========================================="
python -u tune_hp.py --do_alignment --finetune_mode lora
mv best_hyperparameters.json best_hp_lora_align.json 2>/dev/null || true
mv best_hyperparameters_summary.json best_hp_lora_align_summary.json 2>/dev/null || true
echo "LoRA + Alignment HP tuning complete."

# 3. Full Finetune + Alignment
echo "=========================================="
echo "HP Tuning: Full Finetune + Alignment"
echo "=========================================="
python -u tune_hp.py --do_alignment --finetune_mode full
mv best_hyperparameters.json best_hp_full_align.json 2>/dev/null || true
mv best_hyperparameters_summary.json best_hp_full_align_summary.json 2>/dev/null || true
echo "Full Finetune + Alignment HP tuning complete."

# 4. Full Finetune (No Alignment)
echo "=========================================="
echo "HP Tuning: Full Finetune (No Alignment)"
echo "=========================================="
python -u tune_hp.py --finetune_mode full
mv best_hyperparameters.json best_hp_full_noalign.json 2>/dev/null || true
mv best_hyperparameters_summary.json best_hp_full_noalign_summary.json 2>/dev/null || true
echo "Full Finetune (No Alignment) HP tuning complete."

echo "=========================================="
echo "All HP tuning runs complete!"
echo "=========================================="

echo "Job finished at $(date)"
