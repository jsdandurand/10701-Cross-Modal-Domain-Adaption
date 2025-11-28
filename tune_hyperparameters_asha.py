"""
ASHA (Asynchronous Successive Halving Algorithm) Hyperparameter Tuning

This script implements ASHA for hyperparameter tuning, calling train_orca.py
with different configurations and progressively eliminating poor configurations.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import random
import argparse
import json
import os
from typing import Dict, List, Tuple, Optional
import subprocess
import sys
from collections import defaultdict
from datetime import datetime

# Import train_orca function
from train_orca import train_orca, load_hyperparameters, DEVICE

# ===================== ASHA Configuration =====================
RANDOM_SEED = 42
VAL_SPLIT = 0.2  # 20% of training data for validation

# ASHA parameters
MAX_RUNG = 4  # Number of rungs (halving stages)
MIN_RUNG = 0
REDUCTION_FACTOR = 2  # Keep top 1/2 at each rung
MAX_TRIALS = 50  # Maximum number of configurations to try


# ===================== Hyperparameter Search Space =====================
def get_hyperparameter_space(full_finetune: bool = False) -> Dict:
    """Define hyperparameter search space"""
    if full_finetune:
        return {
            'lr': [1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
            'batch_size': [16, 32, 64],
            'weight_decay': [0, 1e-4, 1e-2],
            'optimizer': ['adamw'],
            'scheduler': ['cosine', 'linear'],
            'dropout': [0.0, 0.1, 0.2],
            'alignment_lr': [1e-5, 1e-4, 5e-4],
            'alignment_batch_size': [8, 16, 32]
            # Note: task_epochs, alignment_epochs, and alignment_distance are set manually, not tuned
        }
    else:
        return {
            'lr': [1e-4, 5e-4, 1e-3, 5e-3],
            'batch_size': [32, 64, 128],
            'weight_decay': [0, 1e-4, 1e-2],
            'optimizer': ['adam', 'adamw'],
            'scheduler': ['cosine', 'linear'],
            'dropout': [0.0, 0.1, 0.2],
            'alignment_lr': [1e-5, 1e-4, 5e-4],
            'alignment_batch_size': [8, 16, 32]
            # Note: task_epochs, alignment_epochs, and alignment_distance are set manually, not tuned
        }


def sample_hyperparameters(hyperparameter_space: Dict) -> Dict:
    """Sample a random hyperparameter configuration"""
    config = {}
    for key, values in hyperparameter_space.items():
        config[key] = random.choice(values)
    return config


def get_epochs_for_rung(base_epochs: int, rung: int, max_rung: int) -> int:
    """Scale epochs based on rung level"""
    # Early rungs use fewer epochs, later rungs use more
    scale_factor = (rung + 1) / (max_rung + 1)
    return max(1, int(base_epochs * scale_factor))


# ===================== Data Splitting =====================
def create_train_val_split(val_split=VAL_SPLIT, random_seed=RANDOM_SEED):
    """Create train/validation split"""
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    trainset = datasets.CIFAR10(root="./data", train=True, download=True, 
                               transform=transforms.Compose([transforms.ToTensor()]))
    dataset_size = len(trainset)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    
    split = int(np.floor(val_split * dataset_size))
    val_indices = indices[:split]
    train_indices = indices[split:]
    
    return train_indices, val_indices


def create_val_loader(val_indices, batch_size=64):
    """Create validation data loader"""
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    val_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=val_transform)
    val_subset = Subset(val_dataset, val_indices)
    return DataLoader(val_subset, batch_size=batch_size, shuffle=False)


# ===================== ASHA Algorithm =====================
class ASHAScheduler:
    """ASHA scheduler for hyperparameter tuning"""
    
    def __init__(self, max_rung=MAX_RUNG, reduction_factor=REDUCTION_FACTOR, min_trials_per_rung=4):
        self.max_rung = max_rung
        self.reduction_factor = reduction_factor
        self.min_trials_per_rung = min_trials_per_rung
        self.rungs = defaultdict(list)  # rung -> list of (trial_id, config, result, original_rung)
        self.pending_trials = {}  # trial_id -> (config, rung, status)
        self.completed_trials = {}  # trial_id -> (config, rung, result)
        self.trial_counter = 0
    
    def suggest_trial(self, hyperparameter_space: Dict) -> Optional[Tuple[int, Dict, int]]:
        """
        Suggest a new trial configuration or promote existing trial
        
        Returns:
            (trial_id, config, rung_level) or None if no trial available
        """
        # First, check if we can promote trials from lower rungs
        for r in range(self.max_rung):
            completed_at_rung = [t for t in self.rungs[r] if t[3] == r]  # Only trials that completed at this rung
            if len(completed_at_rung) >= self.min_trials_per_rung:
                # Sort by result and promote top 50%
                completed_at_rung.sort(key=lambda x: x[2], reverse=True)
                top_k = max(1, len(completed_at_rung) // self.reduction_factor)
                
                # Promote top trials to next rung
                for trial_id, config, result, orig_rung in completed_at_rung[:top_k]:
                    if r + 1 <= self.max_rung:
                        # Mark for promotion (will be re-trained at next rung)
                        self.pending_trials[trial_id] = (config, r + 1, 'promoted')
                        print(f"Promoting trial {trial_id} from rung {r} to rung {r + 1} (val_acc={result:.4f})")
                
                # Remove promoted from current rung's active list
                self.rungs[r] = [t for t in self.rungs[r] if t[0] not in [t[0] for t in completed_at_rung[:top_k]]]
                break
        
        # Check for promoted trials to run
        for trial_id, (config, rung, status) in list(self.pending_trials.items()):
            if status == 'promoted':
                self.pending_trials[trial_id] = (config, rung, 'running')
                return trial_id, config, rung
        
        # Otherwise, start a new trial at rung 0
        config = sample_hyperparameters(hyperparameter_space)
        trial_id = self.trial_counter
        self.trial_counter += 1
        self.pending_trials[trial_id] = (config, 0, 'running')
        return trial_id, config, 0
    
    def report_result(self, trial_id: int, config: Dict, rung: int, result: float):
        """Report result for a trial"""
        self.rungs[rung].append((trial_id, config, result, rung))
        self.completed_trials[trial_id] = (config, rung, result)
        if trial_id in self.pending_trials:
            del self.pending_trials[trial_id]
    
    def get_best_config(self) -> Tuple[Optional[Dict], float]:
        """Get best configuration across all rungs"""
        if not self.completed_trials:
            return None, 0.0
        
        # Get best from highest rung with results
        for r in range(self.max_rung, -1, -1):
            rung_trials = [t for t in self.rungs[r] if t[3] == r]  # Only trials that completed at this rung
            if rung_trials:
                rung_trials.sort(key=lambda x: x[2], reverse=True)
                best_trial_id, best_config, best_result, _ = rung_trials[0]
                return best_config, best_result
        
        # Fallback: best across all
        all_results = [(trial_id, config, result) for trial_id, (config, rung, result) in self.completed_trials.items()]
        all_results.sort(key=lambda x: x[2], reverse=True)
        _, best_config, best_result = all_results[0]
        return best_config, best_result
    
    def get_stats(self) -> Dict:
        """Get statistics about current state"""
        stats = {}
        for r in range(self.max_rung + 1):
            completed = [t for t in self.rungs[r] if t[3] == r]
            stats[f'rung_{r}_completed'] = len(completed)
            if completed:
                stats[f'rung_{r}_best'] = max(t[2] for t in completed)
        stats['total_completed'] = len(self.completed_trials)
        stats['total_pending'] = len(self.pending_trials)
        return stats


# ===================== Trial Execution =====================
def run_trial(
    trial_id: int,
    config: Dict,
    rung: int,
    max_rung: int,
    do_alignment: bool,
    full_finetune: bool,
    val_loader: DataLoader,
    temp_config_path: str,
    fixed_task_epochs: int = 50,
    fixed_alignment_epochs: int = 20,
    fixed_alignment_distance: str = 'mse'
) -> float:
    """
    Run a single hyperparameter trial
    
    Args:
        fixed_task_epochs: Fixed number of task epochs (not tuned, scaled by rung)
        fixed_alignment_epochs: Fixed number of alignment epochs (not tuned)
        fixed_alignment_distance: Fixed alignment distance metric (not tuned)
    
    Returns:
        Validation accuracy
    """
    print(f"\n{'='*60}")
    print(f"Trial {trial_id} (Rung {rung}/{max_rung})")
    print(f"Config: {config}")
    print(f"{'='*60}")
    
    # Add fixed parameters (not tuned, but still needed in config)
    config['task_epochs'] = get_epochs_for_rung(fixed_task_epochs, rung, max_rung)
    config['alignment_epochs'] = fixed_alignment_epochs
    config['alignment_distance'] = fixed_alignment_distance
    
    # Save config to temporary file
    with open(temp_config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    try:
        # Run training
        val_acc, _ = train_orca(
            do_alignment=do_alignment,
            full_finetune=full_finetune,
            config_path=temp_config_path,
            val_loader=val_loader,
            return_val_acc=True
        )
        
        print(f"Trial {trial_id} completed: Val Acc = {val_acc:.4f}")
        return val_acc
    
    except Exception as e:
        print(f"Trial {trial_id} failed: {e}")
        return 0.0


# ===================== Main ASHA Tuning =====================
def main():
    parser = argparse.ArgumentParser(description='ASHA Hyperparameter Tuning')
    parser.add_argument('--do_alignment', action='store_true',
                       help='Perform embedding alignment')
    parser.add_argument('--full_finetune', action='store_true',
                       help='Use full fine-tuning')
    parser.add_argument('--max_trials', type=int, default=MAX_TRIALS,
                       help='Maximum number of trials')
    parser.add_argument('--max_rung', type=int, default=MAX_RUNG,
                       help='Maximum rung level')
    parser.add_argument('--output', type=str, default='best_hyperparameters.json',
                       help='Output file for best hyperparameters')
    parser.add_argument('--val_split', type=float, default=VAL_SPLIT,
                       help='Validation split ratio')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from progress file (e.g., best_hyperparameters_progress.json)')
    parser.add_argument('--task_epochs', type=int, default=25,
                       help='Fixed number of task epochs (not tuned, will be scaled by rung)')
    parser.add_argument('--alignment_epochs', type=int, default=15,
                       help='Fixed number of alignment epochs (not tuned)')
    parser.add_argument('--alignment_distance', type=str, default='mse',
                       choices=['mse', 'cosine', 'mmd', 'otdd'],
                       help='Fixed alignment distance metric (not tuned)')
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    
    print("="*60)
    print("ASHA Hyperparameter Tuning")
    print("="*60)
    print(f"Do Alignment: {args.do_alignment}")
    print(f"Full Fine-tuning: {args.full_finetune}")
    print(f"Max Trials: {args.max_trials}")
    print(f"Max Rung: {args.max_rung}")
    print(f"Fixed Task Epochs: {args.task_epochs} (scaled by rung)")
    print(f"Fixed Alignment Epochs: {args.alignment_epochs}")
    print(f"Fixed Alignment Distance: {args.alignment_distance}")
    print("="*60)
    
    # Create validation split
    print("\n>>> Creating validation split...")
    train_indices, val_indices = create_train_val_split(val_split=args.val_split)
    val_loader = create_val_loader(val_indices, batch_size=64)
    print(f"Validation set size: {len(val_indices)}")
    
    # Get hyperparameter space
    hyperparameter_space = get_hyperparameter_space(args.full_finetune)
    
    # Initialize ASHA scheduler
    scheduler = ASHAScheduler(max_rung=args.max_rung, reduction_factor=REDUCTION_FACTOR)
    
    # Temporary config file
    temp_config_path = "temp_asha_config.json"
    
    # Progress file for saving intermediate results
    progress_file = args.output.replace('.json', '_progress.json')
    summary_file = args.output.replace('.json', '_summary.json')
    
    # Resume from progress file if specified
    if args.resume and os.path.exists(args.resume):
        print(f">>> Resuming from {args.resume}...")
        with open(args.resume, 'r') as f:
            progress_data = json.load(f)
        trial_results = progress_data['trial_results']
        best_overall_acc = progress_data['best_val_acc']
        trial_count = progress_data['trials_completed']
        start_time = datetime.fromisoformat(progress_data['start_time'])
        
        # Restore fixed parameters from progress file if available
        if 'config' in progress_data and 'fixed_task_epochs' in progress_data['config']:
            if args.task_epochs == 25:  # Only override if using default
                args.task_epochs = progress_data['config']['fixed_task_epochs']
            if args.alignment_epochs == 20:  # Only override if using default
                args.alignment_epochs = progress_data['config'].get('fixed_alignment_epochs', 20)
            if args.alignment_distance == 'mse':  # Only override if using default
                args.alignment_distance = progress_data['config']['fixed_alignment_distance']
        
        # Restore scheduler state (approximate - will continue from where we left off)
        scheduler = ASHAScheduler(max_rung=args.max_rung, reduction_factor=REDUCTION_FACTOR)
        for result in trial_results:
            scheduler.report_result(result['trial_id'], result['config'], result['rung'], result['val_acc'])
        scheduler.trial_counter = max([r['trial_id'] for r in trial_results]) + 1 if trial_results else 0
        
        print(f">>> Resumed: {trial_count} trials completed, best val_acc: {best_overall_acc:.4f}")
        print(f">>> Using fixed params: task_epochs={args.task_epochs}, alignment_epochs={args.alignment_epochs}, alignment_distance={args.alignment_distance}")
    else:
        # Run trials
        print(f"\n>>> Starting ASHA hyperparameter search...")
        trial_count = 0
        best_overall_acc = 0.0
        trial_results = []  # Store all trial results for summary
        start_time = datetime.now()
        
        # Save initial progress
        progress_data = {
            'start_time': start_time.isoformat(),
        'config': {
            'do_alignment': args.do_alignment,
            'full_finetune': args.full_finetune,
            'max_trials': args.max_trials,
            'max_rung': args.max_rung,
            'reduction_factor': REDUCTION_FACTOR,
            'val_split': args.val_split,
            'fixed_task_epochs': args.task_epochs,
            'fixed_alignment_epochs': args.alignment_epochs,
            'fixed_alignment_distance': args.alignment_distance
        },
        'fixed_params': {
            'task_epochs': args.task_epochs,
            'alignment_epochs': args.alignment_epochs,
            'alignment_distance': args.alignment_distance
        },
        'trial_results': [],
            'best_val_acc': 0.0,
            'trials_completed': 0
        }
    
    while trial_count < args.max_trials:
        trial_suggestion = scheduler.suggest_trial(hyperparameter_space)
        if trial_suggestion is None:
            # Check if we have pending trials to wait for
            if len(scheduler.pending_trials) == 0:
                # No more trials available
                print("No more trials to run")
                break
            else:
                # Wait a bit and try again (in real implementation, would wait for async completion)
                # For now, just continue to next iteration
                continue
        
        trial_id, config, rung = trial_suggestion
        
        # Run trial
        print(f"\n>>> Running Trial {trial_id} (Rung {rung})...")
        val_acc = run_trial(
            trial_id=trial_id,
            config=config,
            rung=rung,
            max_rung=args.max_rung,
            do_alignment=args.do_alignment,
            full_finetune=args.full_finetune,
                val_loader=val_loader,
                temp_config_path=temp_config_path,
                fixed_task_epochs=args.task_epochs,
                fixed_alignment_epochs=args.alignment_epochs,
                fixed_alignment_distance=args.alignment_distance
        )
        
        # Report result
        scheduler.report_result(trial_id, config, rung, val_acc)
        trial_count += 1
        
        # Store result
        trial_result = {
            'trial_id': trial_id,
            'rung': rung,
            'val_acc': float(val_acc),
            'config': config,
            'timestamp': datetime.now().isoformat()
        }
        trial_results.append(trial_result)
        
        # Update best overall
        if val_acc > best_overall_acc:
            best_overall_acc = val_acc
            print(f"*** New best validation accuracy: {best_overall_acc:.4f} ***")
        
        # Update progress data
        progress_data['trial_results'] = trial_results
        progress_data['best_val_acc'] = float(best_overall_acc)
        progress_data['trials_completed'] = trial_count
        progress_data['last_update'] = datetime.now().isoformat()
        
        # Save progress periodically (every trial)
        with open(progress_file, 'w') as f:
            json.dump(progress_data, f, indent=2)
        
        # Print progress
        if trial_count % 5 == 0:
            best_config, best_result = scheduler.get_best_config()
            stats = scheduler.get_stats()
            print(f"\n{'='*60}")
            print(f"Progress: {trial_count}/{args.max_trials} trials")
            print(f"Current best: {best_result:.4f}")
            print(f"Rung stats: {stats}")
            print(f"{'='*60}")
    
    # Create comprehensive summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # Best per rung
    rung_bests = {}
    for result in trial_results:
        rung = result['rung']
        if rung not in rung_bests or result['val_acc'] > rung_bests[rung]['val_acc']:
            rung_bests[rung] = result
    
    # Sort all results by accuracy
    trial_results.sort(key=lambda x: x['val_acc'], reverse=True)
    
    # Create summary
    summary = {
        'start_time': start_time.isoformat(),
        'end_time': end_time.isoformat(),
        'duration_seconds': duration,
        'config': {
            'do_alignment': args.do_alignment,
            'full_finetune': args.full_finetune,
            'max_trials': args.max_trials,
            'max_rung': args.max_rung,
            'reduction_factor': REDUCTION_FACTOR,
            'val_split': args.val_split,
            'fixed_task_epochs': args.task_epochs,
            'fixed_alignment_epochs': args.alignment_epochs,
            'fixed_alignment_distance': args.alignment_distance
        },
        'fixed_params': {
            'task_epochs': args.task_epochs,
            'alignment_epochs': args.alignment_epochs,
            'alignment_distance': args.alignment_distance
        },
        'results': {
            'total_trials': len(trial_results),
            'best_val_acc': float(best_overall_acc),
            'trials_by_rung': {str(r): len([t for t in trial_results if t['rung'] == r]) 
                              for r in range(args.max_rung + 1)},
            'best_per_rung': {
                str(rung): {
                    'trial_id': result['trial_id'],
                    'val_acc': float(result['val_acc']),
                    'config': result['config']
                }
                for rung, result in rung_bests.items()
            }
        },
        'top_10_configs': [
            {
                'rank': i + 1,
                'trial_id': t['trial_id'],
                'rung': t['rung'],
                'val_acc': float(t['val_acc']),
                'config': t['config']
            }
            for i, t in enumerate(trial_results[:10])
        ],
        'all_trials': trial_results  # Full list for detailed analysis
    }
    
    # Save summary
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n>>> Summary saved to {summary_file}")
    
    # Get best configuration
    best_config, best_result = scheduler.get_best_config()
    
    print("\n" + "="*60)
    print("ASHA Tuning Complete!")
    print("="*60)
    print(f"Total Trials: {trial_count}")
    print(f"Best Validation Accuracy: {best_result:.4f}")
    print(f"Duration: {duration/60:.1f} minutes")
    print(f"\nBest Hyperparameters:")
    for key, value in best_config.items():
        print(f"  {key}: {value}")
    print(f"\nFixed Parameters (not tuned):")
    print(f"  task_epochs: {args.task_epochs}")
    print(f"  alignment_epochs: {args.alignment_epochs}")
    print(f"  alignment_distance: {args.alignment_distance}")
    print("="*60)
    
    # Save best configuration (for easy loading in train_orca.py)
    # Add fixed parameters to the saved config
    best_config_with_fixed = best_config.copy()
    best_config_with_fixed['task_epochs'] = args.task_epochs
    best_config_with_fixed['alignment_epochs'] = args.alignment_epochs
    best_config_with_fixed['alignment_distance'] = args.alignment_distance
    
    output_dict = {
        'best_val_acc': float(best_result),
        'hyperparameters': best_config_with_fixed,
        'do_alignment': args.do_alignment,
        'full_finetune': args.full_finetune,
        'num_trials': trial_count,
        'fixed_params': {
            'task_epochs': args.task_epochs,
            'alignment_epochs': args.alignment_epochs,
            'alignment_distance': args.alignment_distance
        },
        'timestamp': end_time.isoformat()
    }
    
    with open(args.output, 'w') as f:
        json.dump(output_dict, f, indent=2)
    
    print(f"\n>>> Best hyperparameters saved to {args.output}")
    print(f">>> Progress log saved to {progress_file}")
    print(f">>> Full summary saved to {summary_file}")
    
    # Clean up
    if os.path.exists(temp_config_path):
        os.remove(temp_config_path)


if __name__ == "__main__":
    main()

