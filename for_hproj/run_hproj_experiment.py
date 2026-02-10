"""
Run Homeomorphic-Projection experiments on our datasets.

This script sets up and runs H-Proj experiments with our dataset configurations.
"""

import os
import sys
import pickle
import argparse
import subprocess

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def create_hproj_config(num_var, num_ineq, num_eq, num_examples, seed=2023, 
                       proj_type='H_Bis', test_size=1024, train_seed=None, prob_type='cvx_qcqp'):
    """
    Create configuration dictionary for H-Proj experiments.
    
    Args:
        num_var: Number of decision variables
        num_ineq: Number of inequality constraints
        num_eq: Number of equality constraints
        num_examples: Total number of training examples
        seed: Random seed for dataset identification
        proj_type: Projection type - 'WS', 'Proj', 'D_Proj', or 'H_Bis'
        test_size: Number of test examples
        train_seed: Random seed for training (PyTorch, NumPy, etc.)
        prob_type: Problem type - 'cvx_qcqp' or 'noncvx'
    
    Returns:
        dict: Configuration dictionary compatible with H-Proj
    """
    config = {}
    
    # Model configuration
    config['predType'] = 'NN_Eq'  # Use equality-constrained NN
    config['projType'] = proj_type  # H_Bis is their novel homeomorphic bisection
    config['probType'] = prob_type
    config['probSize'] = [num_var, num_ineq, num_eq, num_examples]
    config['testSize'] = test_size
    config['saveAllStats'] = False
    config['resultsSaveFreq'] = 1000
    config['seed'] = seed
    if train_seed:
        config['train_seed'] = train_seed  # Separate seed for training reproducibility
    
    # Homeomorphic mapping parameters
    config['mapping_para'] = {
        'training': True,
        'testing': False,
        'n_samples': 1024,
        't_samples': 10000,
        'bound': [0, 1],
        'scale_ratio': 1,
        'shape': 'square',
        'total_iteration': 20000,
        'batch_size': 512,
        'num_layer': 3,
        'lr': 1e-4,
        'lr_decay': 0.9,
        'lr_decay_step': 1000,
        'penalty_coefficient': 10,
        'distortion_coefficient': 1,
        'transport_coefficient': 0,
        'testing_samples': 1024
    }
    
    # Neural network parameters
    config['nn_para'] = {
        'training': True,
        'testing': True,
        'approach': 'unsupervise',
        'total_iteration': 20000,
        'batch_size': 512,
        'lr': 1e-3,
        'lr_decay': 0.9,
        'lr_decay_step': 1000,
        'num_layer': 3,
        'objWeight': 0.1,
        'softWeightInEqFrac': 10,
        'softWeightEqFrac': 10
    }
    
    # Projection parameters
    config['proj_para'] = {
        'useTestCorr': False,
        'corrMode': 'partial',
        'corrTestMaxSteps': 100,
        'corrBis': 0.9,
        'corrEps': 1e-5,
        'corrLr': 1e-5,
        'corrMomentum': 0.1
    }
    
    return config


def update_hproj_default_args(config):
    """
    Update hproj_repo/default_args.py with our configuration.
    
    Args:
        config: Configuration dictionary
    """
    default_args_path = os.path.join('..', 'hproj_repo', 'default_args.py')
    
    # Read current file
    with open(default_args_path, 'r') as f:
        content = f.read()
    
    # Create new config function
    new_config = f"""def config():
    defaults = {{}}
    defaults['predType'] = '{config['predType']}'
    defaults['projType'] = '{config['projType']}'
    defaults['probType'] = '{config['probType']}'
    defaults['probSize'] = {config['probSize']}
    defaults['testSize'] = {config['testSize']}
    defaults['saveAllStats'] = {config['saveAllStats']}
    defaults['resultsSaveFreq'] = {config['resultsSaveFreq']}
    defaults['seed'] = {config['seed']}

    defaults['mapping_para'] = \\
        {config['mapping_para']}

    defaults['nn_para'] = \\
        {config['nn_para']}

    defaults['proj_para'] = \\
        {config['proj_para']}

    return defaults
"""
    
    # Write updated file
    with open(default_args_path, 'w') as f:
        f.write(new_config)
    
    print(f"Updated {default_args_path}")


def run_hproj_training(config):
    """
    Run H-Proj training script.
    
    Args:
        config: Configuration dictionary
    """
    # Change to hproj_repo directory
    original_dir = os.getcwd()
    hproj_dir = os.path.join(original_dir, '..', 'hproj_repo')
    
    try:
        os.chdir(hproj_dir)
        
        # Run training script
        print("\n" + "=" * 80)
        print("Running H-Proj training...")
        print("=" * 80)
        
        # Import and run the training function
        sys.path.insert(0, hproj_dir)
        from training_all import run_instance
        
        run_instance(config)
        
        # Verify that mapping.pth was created if mapping training was enabled
        if config['mapping_para']['training']:
            prob_type = config['probType']
            prob_size = config['probSize']
            # Use correct class name based on problem type
            class_name = 'QCQPProblem' if prob_type == 'cvx_qcqp' else 'NonconvexProblem'
            data_str = f'{class_name}-{prob_size[0]}-{prob_size[1]}-{prob_size[2]}-{prob_size[3]}'
            model_save_dir = os.path.join('models', prob_type, data_str, config['predType'])
            mapping_path = os.path.join(model_save_dir, 'mapping.pth')
            
            if not os.path.exists(mapping_path):
                print("\n" + "!" * 80)
                print("WARNING: Homeomorphic mapping training was enabled but mapping.pth not found!")
                print(f"Expected location: {mapping_path}")
                print("Training may have failed. Check the output above for errors.")
                print("!" * 80)
        
        print("\n" + "=" * 80)
        print("H-Proj training complete!")
        print("=" * 80)
        
    finally:
        os.chdir(original_dir)
        # Remove hproj_dir from path
        if hproj_dir in sys.path:
            sys.path.remove(hproj_dir)


def main():
    parser = argparse.ArgumentParser(description='Run H-Proj experiments on our datasets')
    parser.add_argument('--prob_type', type=str, default='cvx_qcqp',
                       choices=['cvx_qcqp', 'noncvx'],
                       help='Problem type')
    parser.add_argument('--num_var', type=int, default=100, help='Number of decision variables')
    parser.add_argument('--num_ineq', type=int, default=50, help='Number of inequality constraints')
    parser.add_argument('--num_eq', type=int, default=50, help='Number of equality constraints')
    parser.add_argument('--n_examples', type=int, default=10000, help='Total number of examples')
    parser.add_argument('--test_size', type=int, default=None, help='Number of test examples (default: auto-calculated as 8.33%%)')
    parser.add_argument('--seed', type=int, default=2023, help='Random seed for dataset identification')
    parser.add_argument('--train_seed', type=int, default=None, help='Random seed for training (PyTorch, NumPy). If not specified, training is not seeded (non-deterministic)')
    parser.add_argument('--proj_type', type=str, default='H_Bis', 
                       choices=['WS', 'Proj', 'D_Proj', 'H_Bis'],
                       help='Projection type')
    parser.add_argument('--convert_only', action='store_true',
                       help='Only convert dataset, do not run training')
    parser.add_argument('--skip_mapping', action='store_true',
                       help='Skip Stage 1 (homeomorphic mapping training), use existing mapping.pth')
    parser.add_argument('--skip_nn', action='store_true',
                       help='Skip Stage 2 (NN predictor training)')
    
    args = parser.parse_args()
    
    # Calculate test_size to match QCQP class split (test_frac=0.0833)
    if args.test_size is None:
        test_frac = 0.0833
        args.test_size = int(args.n_examples * test_frac)
        print(f"Auto-calculated test_size: {args.test_size} ({test_frac*100:.2f}% of {args.n_examples})")
    
    print("=" * 80)
    print("Homeomorphic-Projection Experiment")
    print("=" * 80)
    print("\nProblem Configuration:")
    print(f"  Problem type: {args.prob_type}")
    print(f"  Variables: {args.num_var}")
    print(f"  Inequality constraints: {args.num_ineq}")
    print(f"  Equality constraints: {args.num_eq}")
    print(f"  Total examples: {args.n_examples}")
    print(f"  Test size: {args.test_size} (test_frac=0.0833 matching QCQP class)")
    print(f"  Dataset seed: {args.seed}")
    print(f"  Training seed: {args.train_seed if args.train_seed is not None else 'None (unseeded)'}")
    print(f"  Projection type: {args.proj_type}")
    
    # Step 1: Convert dataset
    print("\n" + "=" * 80)
    print("Step 1: Converting dataset to H-Proj format")
    print("=" * 80)
    
    if args.prob_type == 'cvx_qcqp':
        cmd = [
            'python', 'convert_cvx_qcqp_for_hproj.py',
            f'num_var={args.num_var}',
            f'num_ineq={args.num_ineq}',
            f'num_eq={args.num_eq}',
            f'n_examples={args.n_examples}',
            f'seed={args.seed}'
        ]
    elif args.prob_type == 'noncvx':
        cmd = [
            'python', 'convert_noncvx_for_hproj.py',
            f'num_var={args.num_var}',
            f'num_ineq={args.num_ineq}',
            f'num_eq={args.num_eq}',
            f'n_examples={args.n_examples}',
            f'seed={args.seed}'
        ]
    
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        print("\nDataset conversion failed!")
        if args.prob_type == 'cvx_qcqp':
            print("Make sure the dataset exists in datasets/cvx_qcqp/")
        else:
            print("Make sure the dataset exists in datasets/noncvx/")
        return
    
    if args.convert_only:
        print("\nDataset conversion complete. Exiting (--convert_only flag set)")
        return
    
    # Step 2: Create and update configuration
    print("\n" + "=" * 80)
    print("Step 2: Updating H-Proj configuration")
    print("=" * 80)
    
    config = create_hproj_config(
        num_var=args.num_var,
        num_ineq=args.num_ineq,
        num_eq=args.num_eq,
        num_examples=args.n_examples,
        seed=args.seed,
        proj_type=args.proj_type,
        test_size=args.test_size,
        train_seed=args.train_seed,
        prob_type=args.prob_type
    )
    
    # Apply skip flags
    if args.skip_mapping:
        print("  Skipping Stage 1: Homeomorphic mapping training (--skip_mapping flag)")
        config['mapping_para']['training'] = False
    if args.skip_nn:
        print("  Skipping Stage 2: NN predictor training (--skip_nn flag)")
        config['nn_para']['training'] = False
    
    update_hproj_default_args(config)
    
    # Step 3: Run training
    print("\n" + "=" * 80)
    print("Step 3: Running H-Proj training")
    print("=" * 80)
    
    run_hproj_training(config)
    
    # Step 4: Print results location
    prob_str = f"{args.prob_type}/QCQPProblem-{args.num_var}-{args.num_ineq}-{args.num_eq}-{args.n_examples}" if args.prob_type == 'cvx_qcqp' else f"{args.prob_type}/NonconvexProblem-{args.num_var}-{args.num_ineq}-{args.num_eq}-{args.n_examples}"
    model_dir = f"../hproj_repo/models/{prob_str}/NN_Eq"
    result_dir = f"../hproj_repo/results/{prob_str}/NN_Eq{args.proj_type}"
    
    print("\n" + "=" * 80)
    print("Experiment Complete!")
    print("=" * 80)
    print(f"\nResults saved to:")
    print(f"  Models: {model_dir}")
    print(f"  Results: {result_dir}")
    print(f"\nTo extract and compare results, use:")
    print(f"  python extract_hproj_results.py --result_dir {result_dir}")


if __name__ == '__main__':
    main()
