#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Experiment Runner for FedSiam-D, Feature Alignment, and Knowledge Distillation

Runs all combinations of:
- Methods: FedSiam-D (baseline), FA, KD
- Datasets: MNIST, CIFAR-10, SVHN
- Partitions: IID, Non-IID-I, Non-IID-II, Non-IID-III, Non-IID-IV

Usage:
    python run_experiments.py
    
    # Or with custom output:
    python run_experiments.py --output results.csv
    
    # Or run specific experiments only:
    python run_experiments.py --methods baseline fa --datasets cifar
"""

import subprocess
import re
import csv
import os
import sys
import time
import argparse
from datetime import datetime
from pathlib import Path


# ============================================================================
# CONFIGURATION
# ============================================================================

# Methods to run
METHODS = {
    'baseline': 'main_fedsiam_d.py',
    'fa': 'train_fedsiam_d_with_feature_alignment.py',
    'kd': 'train_fedsiam_d_with_knowledge_distillation.py'
}

# Datasets
DATASETS = ['mnist', 'cifar', 'svhn']

# Partitions (IID settings)
PARTITIONS = {
    'iid': 'iid',
    'noniid_i': 'noniid_tradition',      # Non-IID-I (traditional)
    'noniid_ii': 'noniid_improve',       # Non-IID-II (improved)
    'noniid_iii': 'noniid_ssl',          # Non-IID-III (SSL-specific)
    'noniid_iv': 'noniid_extreme'        # Non-IID-IV (Extreme-case)
}

# Fixed parameters
FIXED_PARAMS = {
    'epochs': 50,
    'gpu': 0,
    'label_rate': 0.1,
    'phi_g': 3
}

# Additional dataset-specific parameters
DATASET_PARAMS = {
    'cifar': {
        'ramp': 'rectangle',
        'local_bs': 30
    },
    'svhn': {},
    'mnist': {}
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def parse_accuracy_from_output(output_text):
    """
    Extract accuracies from script output.
    
    Expected format:
        s:Training accuracy: 95.6123%
        s:Testing accuracy: 94.2341%
        
        t:Training accuracy: 96.1234%
        t:Testing accuracy: 95.3421%
    
    Returns:
        dict with keys: s_train, s_test, t_train, t_test
    """
    results = {
        's_train': None,
        's_test': None,
        't_train': None,
        't_test': None
    }
    
    # Regex patterns
    patterns = {
        's_train': r's:Training accuracy:\s*([\d.]+)%',
        's_test': r's:Testing accuracy:\s*([\d.]+)%',
        't_train': r't:Training accuracy:\s*([\d.]+)%',
        't_test': r't:Testing accuracy:\s*([\d.]+)%'
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, output_text)
        if match:
            results[key] = float(match.group(1))
    
    return results


def build_command(method, dataset, partition, fixed_params, dataset_params):
    """Build the command to run an experiment."""
    script = METHODS[method]
    partition_arg = PARTITIONS[partition]
    
    # Base command
    cmd = [
        'python', script,
        '--dataset', dataset,
        '--iid', partition_arg,
        '--epochs', str(fixed_params['epochs']),
        '--gpu', str(fixed_params['gpu']),
        '--label_rate', str(fixed_params['label_rate']),
        '--phi_g', str(fixed_params['phi_g'])
    ]
    
    # Add dataset-specific parameters
    if dataset in dataset_params and dataset_params[dataset]:
        for param, value in dataset_params[dataset].items():
            cmd.extend([f'--{param}', str(value)])
    
    return cmd


def run_experiment(method, dataset, partition, output_dir='experiment_outputs'):
    """
    Run a single experiment and return results.
    
    Returns:
        dict with experiment info and results
    """
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Build command
    cmd = build_command(method, dataset, partition, FIXED_PARAMS, DATASET_PARAMS)
    cmd_str = ' '.join(cmd)
    
    print(f"\n{'='*80}")
    print(f"Running: {method.upper()} | {dataset.upper()} | {partition.upper()}")
    print(f"Command: {cmd_str}")
    print(f"{'='*80}\n")
    
    # Output file for this experiment
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(
        output_dir, 
        f'{method}_{dataset}_{partition}_{timestamp}.log'
    )
    
    # Run experiment
    start_time = time.time()
    
    try:
        # Run command and capture output
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=7200  # 2 hour timeout per experiment
        )
        
        output_text = result.stdout
        elapsed_time = time.time() - start_time
        
        # Save full output to file
        with open(output_file, 'w') as f:
            f.write(f"Command: {cmd_str}\n")
            f.write(f"Started: {timestamp}\n")
            f.write(f"Duration: {elapsed_time:.2f} seconds\n")
            f.write(f"Return code: {result.returncode}\n")
            f.write(f"\n{'='*80}\n")
            f.write(f"OUTPUT:\n")
            f.write(f"{'='*80}\n\n")
            f.write(output_text)
        
        # Parse accuracies
        accuracies = parse_accuracy_from_output(output_text)
        
        # Prepare result
        experiment_result = {
            'method': method,
            'dataset': dataset,
            'partition': partition,
            'status': 'success' if result.returncode == 0 else 'failed',
            'duration_seconds': elapsed_time,
            'timestamp': timestamp,
            'output_file': output_file,
            **accuracies
        }
        
        # Print results
        print(f"\n{'='*80}")
        print(f"COMPLETED: {method.upper()} | {dataset.upper()} | {partition.upper()}")
        print(f"Duration: {elapsed_time:.2f} seconds ({elapsed_time/60:.1f} minutes)")
        print(f"Status: {experiment_result['status']}")
        if accuracies['s_test'] is not None:
            print(f"Test Accuracy (s): {accuracies['s_test']:.2f}%")
            print(f"Test Accuracy (t): {accuracies['t_test']:.2f}%")
        print(f"Full output saved to: {output_file}")
        print(f"{'='*80}\n")
        
        return experiment_result
        
    except subprocess.TimeoutExpired:
        elapsed_time = time.time() - start_time
        print(f"\n[ERROR] Experiment TIMEOUT after {elapsed_time:.2f} seconds")
        
        return {
            'method': method,
            'dataset': dataset,
            'partition': partition,
            'status': 'timeout',
            'duration_seconds': elapsed_time,
            'timestamp': timestamp,
            'output_file': output_file,
            's_train': None,
            's_test': None,
            't_train': None,
            't_test': None
        }
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"\n[ERROR] Experiment FAILED: {str(e)}")
        
        return {
            'method': method,
            'dataset': dataset,
            'partition': partition,
            'status': 'error',
            'duration_seconds': elapsed_time,
            'timestamp': timestamp,
            'output_file': None,
            's_train': None,
            's_test': None,
            't_train': None,
            't_test': None,
            'error': str(e)
        }


def save_results_to_csv(results, output_file='results.csv'):
    """Save all results to CSV file."""
    if not results:
        print("No results to save.")
        return
    
    # CSV headers
    fieldnames = [
        'method',
        'dataset', 
        'partition',
        'status',
        's_train_acc',
        's_test_acc',
        't_train_acc',
        't_test_acc',
        'duration_seconds',
        'timestamp',
        'output_file'
    ]
    
    # Write CSV
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in results:
            writer.writerow({
                'method': result['method'],
                'dataset': result['dataset'],
                'partition': result['partition'],
                'status': result['status'],
                's_train_acc': result.get('s_train', ''),
                's_test_acc': result.get('s_test', ''),
                't_train_acc': result.get('t_train', ''),
                't_test_acc': result.get('t_test', ''),
                'duration_seconds': result['duration_seconds'],
                'timestamp': result['timestamp'],
                'output_file': result.get('output_file', '')
            })
    
    print(f"\n✅ Results saved to: {output_file}")


def print_summary(results):
    """Print summary of all experiments."""
    print(f"\n{'='*80}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*80}\n")
    
    total = len(results)
    success = sum(1 for r in results if r['status'] == 'success')
    failed = sum(1 for r in results if r['status'] in ['failed', 'error'])
    timeout = sum(1 for r in results if r['status'] == 'timeout')
    
    total_time = sum(r['duration_seconds'] for r in results)
    
    print(f"Total experiments: {total}")
    print(f"Success: {success}")
    print(f"Failed: {failed}")
    print(f"Timeout: {timeout}")
    print(f"\nTotal time: {total_time/3600:.2f} hours")
    print(f"Average time per experiment: {total_time/total/60:.1f} minutes")
    
    # Best results per dataset
    print(f"\n{'='*80}")
    print("BEST RESULTS PER DATASET (Test Accuracy - t)")
    print(f"{'='*80}\n")
    
    for dataset in DATASETS:
        dataset_results = [r for r in results if r['dataset'] == dataset and r.get('t_test') is not None]
        if dataset_results:
            best = max(dataset_results, key=lambda x: x['t_test'])
            print(f"{dataset.upper():8s} | Best: {best['t_test']:.2f}% | {best['method']:8s} | {best['partition']}")
    
    print(f"\n{'='*80}\n")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Run FedSiam-D experiments')
    parser.add_argument('--output', type=str, default='results.csv',
                       help='Output CSV file (default: results.csv)')
    parser.add_argument('--output_dir', type=str, default='experiment_outputs',
                       help='Directory for experiment logs (default: experiment_outputs)')
    parser.add_argument('--methods', nargs='+', choices=list(METHODS.keys()) + ['all'],
                       default=['all'],
                       help='Methods to run (default: all)')
    parser.add_argument('--datasets', nargs='+', choices=DATASETS + ['all'],
                       default=['all'],
                       help='Datasets to run (default: all)')
    parser.add_argument('--partitions', nargs='+', choices=list(PARTITIONS.keys()) + ['all'],
                       default=['all'],
                       help='Partitions to run (default: all)')
    
    args = parser.parse_args()
    
    # Resolve 'all' options
    methods_to_run = list(METHODS.keys()) if 'all' in args.methods else args.methods
    datasets_to_run = DATASETS if 'all' in args.datasets else args.datasets
    partitions_to_run = list(PARTITIONS.keys()) if 'all' in args.partitions else args.partitions
    
    # Build experiment list
    experiments = [
        (method, dataset, partition)
        for method in methods_to_run
        for dataset in datasets_to_run
        for partition in partitions_to_run
    ]
    
    print("\n" + "="*80)
    print("FEDERATED SEMI-SUPERVISED LEARNING EXPERIMENTS")
    print("="*80)
    print(f"\nTotal experiments to run: {len(experiments)}")
    print(f"  Methods: {', '.join(methods_to_run)}")
    print(f"  Datasets: {', '.join(datasets_to_run)}")
    print(f"  Partitions: {', '.join(partitions_to_run)}")
    print(f"\nOutput CSV: {args.output}")
    print(f"Output directory: {args.output_dir}")
    
    # Estimate time
    avg_time_per_exp = 30  # minutes (conservative estimate)
    estimated_hours = len(experiments) * avg_time_per_exp / 60
    print(f"\nEstimated total time: {estimated_hours:.1f} hours")
    print("  (assuming ~30 min per experiment)")
    
    # Confirm
    print("\n" + "="*80)
    response = input("Proceed with experiments? [y/N]: ")
    if response.lower() not in ['y', 'yes']:
        print("Aborted.")
        return
    
    # Run all experiments
    results = []
    start_time = time.time()
    
    for idx, (method, dataset, partition) in enumerate(experiments, 1):
        print(f"\n{'#'*80}")
        print(f"EXPERIMENT {idx}/{len(experiments)}")
        print(f"{'#'*80}")
        
        result = run_experiment(method, dataset, partition, args.output_dir)
        results.append(result)
        
        # Save incrementally (so we don't lose data if crash)
        save_results_to_csv(results, args.output)
        
        # Print progress
        elapsed = time.time() - start_time
        avg_time = elapsed / idx
        remaining = avg_time * (len(experiments) - idx)
        
        print(f"\nProgress: {idx}/{len(experiments)} ({idx/len(experiments)*100:.1f}%)")
        print(f"Elapsed: {elapsed/3600:.2f} hours")
        print(f"Estimated remaining: {remaining/3600:.2f} hours")
    
    # Final save and summary
    save_results_to_csv(results, args.output)
    print_summary(results)
    
    print(f"\nAll experiments completed!")
    print(f"Results saved to: {args.output}")
    print(f"Logs saved to: {args.output_dir}/")


if __name__ == '__main__':
    main()
