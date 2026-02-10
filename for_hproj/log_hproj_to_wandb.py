#!/usr/bin/env python3
"""
Log H-Proj results from CSV files to wandb.

This script reads H-Proj CSV results and logs them to wandb for comparison with other methods.
It extracts configuration parameters (prob_type, num_var, num_eq, num_ineq, seed, train_seed, proj_type)
from the CSV filename and directory structure.
"""

import os
import sys
import argparse
import pandas as pd
import wandb
from pathlib import Path
from scipy.stats import gmean
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from constants import TEST_METRICS_DICT_KEYS


def parse_csv_filename(csv_path):
    """
    Parse configuration from CSV filename and directory path.
    
    Expected format:
    - Filename: hproj_{prob_type}_{proj_type}_var{num_var}_ineq{num_ineq}_eq{num_eq}_trainseed{train_seed}.csv
    - Or:       hproj_{prob_type}_{proj_type}_var{num_var}_ineq{num_ineq}_eq{num_eq}_unseeded.csv
    - Directory: .../results/{prob_type}/var{num_var}_ineq{num_ineq}_eq{num_eq}_seed{seed}/hproj/
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        dict: Configuration parameters
    """
    csv_path = Path(csv_path)
    filename = csv_path.stem  # Remove .csv extension
    dir_path = csv_path.parent.parent  # Go up to var{}_ineq{}_eq{}_seed{} directory
    
    config = {}
    
    # Parse from filename: hproj_{prob_type}_{proj_type}_var{}_ineq{}_eq{}_trainseed{}
    parts = filename.split('_')
    
    # Find prob_type (cvx_qcqp or noncvx) - check noncvx first to avoid cvx match
    if 'noncvx' in filename:
        config['prob_type'] = 'noncvx'
        # Find index of noncvx
        prob_idx = parts.index('noncvx')
        start_idx = prob_idx + 1
    elif 'cvx_qcqp' in filename:
        config['prob_type'] = 'cvx_qcqp'
        # Find index of qcqp
        prob_idx = parts.index('qcqp')
        start_idx = prob_idx + 1
    else:
        raise ValueError(f"Cannot determine prob_type from filename: {filename}")
    
    # Find proj_type (next part after prob_type)
    # Look for projection type patterns (H, Bis combined as H_Bis, or WS, Proj, D_Proj, etc.)
    proj_parts = []
    i = start_idx
    while i < len(parts) and not parts[i].startswith('var'):
        # Accumulate projection type parts
        proj_parts.append(parts[i])
        i += 1
    
    config['proj_type'] = '_'.join(proj_parts) if proj_parts else 'H_Bis'
    
    # Parse num_var, num_ineq, num_eq, trainseed from filename
    for i, part in enumerate(parts):
        if part.startswith('var'):
            # Extract number after 'var'
            num_str = part[3:]
            if num_str.isdigit():
                config['num_var'] = int(num_str)
        elif part.startswith('ineq'):
            num_str = part[4:]
            if num_str.isdigit():
                config['num_ineq'] = int(num_str)
        elif part.startswith('eq') and not part.startswith('eq_'):
            num_str = part[2:]
            if num_str.isdigit():
                config['num_eq'] = int(num_str)
        elif part.startswith('trainseed'):
            num_str = part[9:]  # After 'trainseed'
            if num_str.isdigit():
                config['train_seed'] = int(num_str)
        elif part == 'unseeded':
            config['train_seed'] = None
    
    # Parse seed from directory path
    dir_name = dir_path.name
    if 'seed' in dir_name:
        seed_parts = dir_name.split('_')
        for part in seed_parts:
            if part.startswith('seed'):
                num_str = part[4:]
                if num_str.isdigit():
                    config['seed'] = int(num_str)
                    break
    
    # Set default values if not found
    config.setdefault('num_var', 100)
    config.setdefault('num_ineq', 50)
    config.setdefault('num_eq', 50)
    config.setdefault('seed', 2023)
    config.setdefault('train_seed', None)
    
    return config


def load_csv_results(csv_path):
    """
    Load results from CSV file.
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        dict: Results dictionary
    """
    df = pd.read_csv(csv_path)
    
    if len(df) == 0:
        raise ValueError(f"Empty CSV file: {csv_path}")
    
    # Convert first row to dictionary
    results = df.iloc[0].to_dict()
    
    return results


def create_wandb_config(config, results):
    """
    Create wandb config from parsed configuration and results.
    
    Args:
        config: Configuration dictionary from filename/path parsing
        results: Results dictionary from CSV
        
    Returns:
        dict: wandb config dictionary
    """
    wandb_config = {
        # Dataset configuration
        'dataset': {
            'prob_type': config['prob_type'],
            'num_var': config['num_var'],
            'num_ineq': config['num_ineq'],
            'num_eq': config['num_eq'],
            'seed': config['seed'],
        },
        # Model configuration
        'model': {
            'name': 'hproj',
            'proj_type': config['proj_type'],
        },
        # Training configuration
        'train_seed': config.get('train_seed'),
        'method': results.get('method', f"H-Proj-{config['proj_type']}"),
    }
    
    return wandb_config


def log_results_to_wandb(csv_path, wandb_project, wandb_name=None, dry_run=False):
    """
    Log results from CSV to wandb.
    
    Args:
        csv_path: Path to CSV file
        wandb_project: wandb project name
        wandb_name: Optional custom run name
        dry_run: If True, print what would be logged without actually logging
        
    Returns:
        bool: True if successful
    """
    # Parse configuration from filename
    config = parse_csv_filename(csv_path)
    
    # Load results from CSV
    results = load_csv_results(csv_path)
    
    # Create wandb config
    wandb_config = create_wandb_config(config, results)
    
    # Create default run name if not provided
    if wandb_name is None:
        train_seed_str = f"_trainseed{config['train_seed']}" if config['train_seed'] is not None else "_unseeded"
        wandb_name = f"seed{config['seed']}_hproj_{config['proj_type']}{train_seed_str}"
    
    if dry_run:
        print("\n" + "=" * 80)
        print("DRY RUN - Would log the following to wandb:")
        print("=" * 80)
        print(f"\nProject: {wandb_project}")
        print(f"Run name: {wandb_name}")
        print(f"\nConfig:")
        for key, value in wandb_config.items():
            print(f"  {key}: {value}")
        print(f"\nMetrics (filtered by TEST_METRICS_DICT_KEYS):")
        # Create allowed metric keys (strip 'test/' prefix from TEST_METRICS_DICT_KEYS)
        allowed_keys = set()
        for key in TEST_METRICS_DICT_KEYS:
            if key.startswith('test/'):
                allowed_keys.add(key.replace('test/', ''))
        # Always include feasibility_rate
        allowed_keys.add('feasibility_rate')
        for key, value in results.items():
            if key not in ['method', 'train_seed'] and key in allowed_keys:
                print(f"  test/{key}: {value}")
        return True
    
    # Initialize wandb run
    print(f"\nLogging to wandb project: {wandb_project}")
    print(f"Run name: {wandb_name}")
    
    with wandb.init(project=wandb_project, name=wandb_name, config=wandb_config) as run:
        # Create allowed metric keys (strip 'test/' prefix from TEST_METRICS_DICT_KEYS)
        allowed_keys = set()
        for key in TEST_METRICS_DICT_KEYS:
            if key.startswith('test/'):
                allowed_keys.add(key.replace('test/', ''))
        # Always include feasibility_rate
        allowed_keys.add('feasibility_rate')
        
        # Log only allowed metrics with 'test/' prefix
        metrics = {}
        for key, value in results.items():
            if key not in ['method', 'train_seed'] and key in allowed_keys:
                # Convert to float if possible
                try:
                    if pd.notna(value):
                        metrics[f'test/{key}'] = float(value)
                    else:
                        metrics[f'test/{key}'] = None
                except (ValueError, TypeError):
                    metrics[f'test/{key}'] = value
        
        # Log metrics
        wandb.log(metrics)
        
        print(f"✓ Successfully logged {len(metrics)} metrics to wandb")
        
    return True


def process_directory(results_dir, wandb_project, recursive=False, dry_run=False):
    """
    Process all CSV files in a directory.
    
    Args:
        results_dir: Directory containing CSV files
        wandb_project: wandb project name
        recursive: If True, search recursively for CSV files
        dry_run: If True, print what would be logged without actually logging
        
    Returns:
        int: Number of files processed
    """
    results_dir = Path(results_dir)
    
    if not results_dir.exists():
        print(f"Error: Directory not found: {results_dir}")
        return 0
    
    # Find CSV files
    if recursive:
        csv_files = list(results_dir.rglob('hproj_*.csv'))
    else:
        csv_files = list(results_dir.glob('hproj_*.csv'))
    
    if not csv_files:
        print(f"No H-Proj CSV files found in {results_dir}")
        return 0
    
    print(f"\nFound {len(csv_files)} CSV file(s) to process")
    
    # Process each file
    success_count = 0
    for csv_file in csv_files:
        print("\n" + "=" * 80)
        print(f"Processing: {csv_file.name}")
        print("=" * 80)
        
        try:
            if log_results_to_wandb(csv_file, wandb_project, dry_run=dry_run):
                success_count += 1
        except Exception as e:
            print(f"✗ Error processing {csv_file.name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 80)
    print(f"Successfully processed {success_count}/{len(csv_files)} files")
    print("=" * 80)
    
    return success_count


def main():
    parser = argparse.ArgumentParser(
        description='Log H-Proj CSV results to wandb',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Log a single CSV file
  python log_hproj_to_wandb.py --csv results/noncvx/.../hproj_noncvx_H_Bis_var100_ineq50_eq50_trainseed123.csv --project noncvx_var100_ineq50_eq50_seed2025
  
  # Log all CSV files in a directory
  python log_hproj_to_wandb.py --dir results/noncvx/var100_ineq50_eq50_seed2025/hproj/ --project noncvx_var100_ineq50_eq50_seed2025
  
  # Log all CSV files recursively
  python log_hproj_to_wandb.py --dir results/noncvx/ --project noncvx --recursive
  
  # Dry run (preview without logging)
  python log_hproj_to_wandb.py --dir results/noncvx/var100_ineq50_eq50_seed2025/hproj/ --project test --dry_run
        """
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--csv', type=str,
                            help='Path to a single CSV file')
    input_group.add_argument('--dir', type=str,
                            help='Directory containing CSV files')
    
    # Wandb options
    parser.add_argument('--project', type=str, required=True,
                       help='wandb project name')
    parser.add_argument('--name', type=str,
                       help='Optional custom run name (only for single CSV)')
    
    # Processing options
    parser.add_argument('--recursive', action='store_true',
                       help='Search recursively for CSV files (only with --dir)')
    parser.add_argument('--dry_run', action='store_true',
                       help='Print what would be logged without actually logging')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("H-Proj Results → wandb Logger")
    print("=" * 80)
    
    if args.csv:
        # Process single file
        success = log_results_to_wandb(args.csv, args.project, args.name, args.dry_run)
        exit(0 if success else 1)
    else:
        # Process directory
        count = process_directory(args.dir, args.project, args.recursive, args.dry_run)
        exit(0 if count > 0 else 1)


if __name__ == '__main__':
    main()
