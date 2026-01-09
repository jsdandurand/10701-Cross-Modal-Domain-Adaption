# Cross-Modal Domain Adaptation with TinyLlama

![Python](https://img.shields.io/badge/Python-3.11-blue)

TL;DR: Cross-modal domain adaptation by aligning image patch embeddings with TinyLlama token embeddings; evaluated on CIFAR-10.

Key result: Full fine-tuning + MMD alignment reached 87.39% test accuracy on CIFAR-10.

Links: [Report PDF](10701__Final_Project_Report.pdf) | [Reproduce](#reproduce-full-runs) | [Code structure](#project-structure)

## Key Results (Test Accuracy)

| Method | Test Accuracy (%) | Delta vs. Baseline |
| --- | ---: | ---: |
| FPT Baseline | 67.06 | - |
| FPT + Alignment (MSE) | 66.74 | -0.32 |
| FPT + Alignment (MMD) | 66.96 | -0.10 |
| Full Fine-tuning | 85.26 | 18.20 |
| Full Fine-tuning + Alignment (MSE) | 85.48 | 18.42 |
| Full Fine-tuning + Alignment (MMD) | 87.39 | 20.33 |
| LoRA Fine-tuning | 81.42 | 14.36 |
| LoRA Fine-tuning + Alignment (MSE) | 81.87 | 14.81 |
| LoRA Fine-tuning + Alignment (MMD) | 83.86 | 16.80 |

## Quickstart (Smoke Test)

Install deps:

```bash
# Option A: conda/mamba (matches repo scripts)
bash setup_local_env.sh

# Option B: pip
python -m pip install -r requirements.txt
```

Inspect CLI:

```bash
python train.py --help
```

Tiny smoke run (see P0.4 in the report for recommended tiny settings):

```bash
cat > /tmp/smoke_config.json <<'JSON'
{
  "lr": 0.0001,
  "batch_size": 8,
  "weight_decay": 0.0,
  "optimizer": "adamw",
  "scheduler": "cosine",
  "dropout": 0.1,
  "alignment_epochs": 1,
  "alignment_lr": 0.0001,
  "alignment_batch_size": 8,
  "alignment_distance": "mse",
  "task_epochs": 1,
  "lora_rank": 8,
  "lora_alpha": 16,
  "lora_dropout": 0.1
}
JSON

python train.py --finetune_mode lora --config /tmp/smoke_config.json --val_split 0.1
```

## Reproduce Full Runs

LoRA (best reported):

```bash
python train.py --do_alignment --finetune_mode lora --config final_lora_align.json
```

Full fine-tune (best reported):

```bash
python train.py --do_alignment --finetune_mode full --config final_full_align.json
```

Alignment stage only (legacy standalone script):

```bash
python train_stage2_align.py --distance mmd
```

## Project Structure

- `train.py`: unified training pipeline with optional alignment + FPT/full/LoRA fine-tuning.
- `train_stage2_align.py`: standalone alignment stage (legacy pipeline).
- `train_tinyllama.py`: earlier baseline training script.
- `tune_hp.py`: ASHA hyperparameter search driver.
- `final_*.json`: configs for final runs used in the report.
- `run_train_tinyllama.sh`: SLURM script for running full experiments.
- `mlruns/`: MLflow logs.

## Notes / Limitations

- Dataset download: CIFAR-10 is downloaded automatically into `./data` on first run.
- Expected GPU: runs are intended for a single GPU; full fine-tuning can be slow on CPU.
- Runtime: 50-epoch runs can take hours depending on GPU; LoRA is faster than full fine-tuning.
- Avoid test-set tuning: use `--val_split` for tuning; do not use the test set for hyperparameter selection.
