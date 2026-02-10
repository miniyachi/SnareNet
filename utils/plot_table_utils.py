"""
Utility functions for plotting WandB experiment results.
"""

from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from omegaconf import DictConfig
import pandas as pd
from scipy.stats import gmean

# Set Times New Roman font for academic plots
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Liberation Serif', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'cm'  # Computer Modern for better LaTeX compatibility
plt.rcParams['mathtext.rm'] = 'serif'
plt.rcParams['mathtext.tt'] = 'monospace'

from utils.utils import get_hyperparam_str, get_method_name, get_value_or_die
from constants import HISTORY_DF_COLS, TEST_METRICS_DICT_KEYS

# Color scheme for models
MODEL_TO_COLOR = {
    'snarenet': 'C0',  # Blue
    'hproj': 'C4',  # Purple
    'hardnetaff': 'C1', # Orange
    'dc3': 'C2',        # Green
    'cvxpy': 'C3',     # Red
}

# Marker scheme for training strategies
def _get_marker_for_method(method_name):
    """
    Determine the marker shape based on the training strategy used in the method.
    
    Args:
        method_name: Full method name string
    
    Returns:
        marker: Matplotlib marker string ('o', 's', 'D', etc.)
    """
    method_lower = method_name.lower()
    
    # Check for combined strategy (both soft and adaRel)
    if 'adarel' in method_lower and 'soft' in method_lower:
        return 'D'  # Diamond for combined strategy
    # Check for AdaRel only
    elif 'adarel' in method_lower:
        return 's'  # Square for AdaRel
    # Check for soft penalty only (or no special training)
    elif 'soft' in method_lower:
        return 'o'  # Circle for soft penalty
    else:
        return 'o'  # Default circle


def _get_marker_size_for_method(method_name, methods_to_include, base_size=6):
    """
    Determine the marker size based on the method's position within its model group.
    Lighter colors (earlier in sequence) get larger markers, darker colors get smaller markers.
    
    Args:
        method_name: Full method name string
        methods_to_include: Dictionary of methods in display order
        base_size: Base marker size (default: 6)
    
    Returns:
        marker_size: Float marker size
    """
    if methods_to_include is None:
        return base_size
    
    # Get list of all methods in order
    all_methods = list(methods_to_include.keys())
    
    # Extract model from method name
    model = method_name.split('_')[0]
    
    # Find methods with the same model
    same_model_methods = [m for m in all_methods if m.split('_')[0] == model]
    
    if len(same_model_methods) <= 1:
        return base_size
    
    # Find position in the same-model group (0-indexed)
    try:
        position = same_model_methods.index(method_name)
    except ValueError:
        return base_size
    
    # Calculate size: first (lightest) is largest, middle and last are small
    # For 3 methods: position 0 (lightest) -> large (1.3), positions 1 & 2 -> small (0.7)
    n_methods = len(same_model_methods)
    
    if position == 0:
        # First method (lightest) gets largest marker
        size_factor = 1.3
    else:
        # All other methods (middle and darkest) get smaller markers
        size_factor = 0.7
    
    return base_size * size_factor

# Display names for models
MODEL_TO_DISPLAYNAME = {
    'snarenet': 'SnareNet',
    'hardnetaff': 'HardNet',
    'dc3': 'DC3'
}


def get_run_history_df(run, cols=None, max_samples=10000):
    """
    Collect history dataframe for a single run with specified columns.
    
    Args:
        run: WandB run object
        cols: List of column names to extract from history. 
                      If None, uses HISTORY_DF_COLS from constants
    
    Returns:
        pd.DataFrame: History dataframe with selected columns plus metadata columns
                      (seed, model, method_params, method_name)
    """
    
    if cols is None:
        cols = HISTORY_DF_COLS
    
    # Get the full history
    history = run.history(samples=max_samples)
    
    # Add metadata columns from config first
    config = DictConfig(run.config)
    df = pd.DataFrame({
        'model': config.model.name,
        'method_name': get_method_name(config),
        'seed': config.seed if 'seed' in config else config.train_seed if 'seed' in config else config.train_seed,
    }, index=history.index)
    
    # Select and append the metric columns we care about (that exist in the history)
    available_cols = [col for col in cols if col in history.columns]
    for col in available_cols:
        df[col] = history[col]
    
    return df


def get_run_test_metrics_df(run, test_metric_keys=None):
    """
    Collect test metrics from run summary and organize into a dataframe.
    
    Args:
        run: WandB run object
        test_metric_keys: List of keys to extract from summary.
                          If None, uses TEST_METRICS_DICT_KEYS from constants
    
    Returns:
        pd.DataFrame: Single-row dataframe with test metrics plus metadata columns
                      (seed, model, method_params, method_name)
    """    
    if test_metric_keys is None:
        test_metric_keys = TEST_METRICS_DICT_KEYS
    
    # Get the summary
    summary = run.summary
    
    # Add metadata from config first
    config = DictConfig(run.config)
    metrics = {
        'model': config.model.name,
        'method_name': get_method_name(config),
        'seed': config.seed if 'seed' in config else config.train_seed,
    }
    
    # Extract and append test metrics that exist in summary
    for key in test_metric_keys:
        if key in summary:
            metrics[key] = get_value_or_die(summary, key)
    
    # Create single-row dataframe
    df = pd.DataFrame([metrics])
    
    return df


def get_runs_dfs(api, workspace, project, max_samples=10000):
    """
    Collect and concatenate history and test metrics dataframes from all runs in a project.
    
    Args:
        api: WandB API object
        workspace: WandB workspace name
        project: WandB project name
        max_samples: Max number of samples in history to load per run
    
    Returns:
        tuple: (history_df, test_metrics_df) where:
            - history_df: Concatenated dataframe with all runs' training histories
            - test_metrics_df: Concatenated dataframe with all runs' test metrics
            Both include metadata columns (seed, model, method_name) followed by metric columns
    """    
    runs = api.runs(f"{workspace}/{project}")
    
    history_dfs = []
    test_dfs = []
    
    for run in runs:
        # Skip non-finished runs
        if run.state != 'finished':
            print(f"Skipping run {run.id} with state {run.state}")
            continue
        
        history_df = get_run_history_df(run, max_samples=max_samples)
        test_df = get_run_test_metrics_df(run)
        history_dfs.append(history_df)
        test_dfs.append(test_df)
    
    # Concatenate all dataframes
    history_df = pd.concat(history_dfs, ignore_index=True)
    test_df = pd.concat(test_dfs, ignore_index=True)
    
    return history_df, test_df


def _filter_history_df(history_df, methods_to_include):
    """
    Filter history dataframe based on methods specification.
    
    Args:
        history_df: History dataframe with metadata columns (model, method_name, seed)
        methods_to_include: Dict mapping full method names to display names, or None
                           Example: {'snarenet_pinv_lambda0.01_adaRel500_soft100': 'SnareNet (rtol=1e-2)'}
    """
    if methods_to_include is None:
        return history_df
    
    # Filter dataframe using method_name column (keys of methods_to_include)
    valid_method_names = list(methods_to_include.keys())
    return history_df[history_df['method_name'].isin(valid_method_names)]


def _compute_metric_stats_df(history_df, metric, zero_replacement=1e-16):
    """
    Compute geometric mean and confidence bounds for a metric across seeds.
    
    Args:
        history_df: History dataframe with columns (model, method_name, seed, epoch, metric)
        metric: Metric column name to compute statistics for
        zero_replacement: Value to replace zeros/negatives for log scale
    
    Returns:
        DataFrame with columns (method_name, epoch, gmean, lower, upper, n_seeds)
    """    
    if metric not in history_df.columns:
        return pd.DataFrame()
    
    # Replace non-positive values
    df = history_df.copy()
    # Convert to numeric, coercing errors to NaN
    df[metric] = pd.to_numeric(df[metric], errors='coerce')
    # Replace NaN, zeros, and negative values with zero_replacement
    df[metric] = df[metric].where(df[metric] > 0, zero_replacement)
    
    # Group by method_name and epoch, compute log statistics across seeds
    def compute_geom_stats(group):
        values = group[metric].values
        if len(values) == 0:
            return pd.Series({'gmean': np.nan, 'lower': np.nan, 'upper': np.nan, 'n_seeds': 0})
        
        log_values = np.log(values)
        log_mean = np.mean(log_values)
        log_std = np.std(log_values)
        
        return pd.Series({
            'gmean': np.exp(log_mean),
            'lower': np.exp(log_mean - log_std),
            'upper': np.exp(log_mean + log_std),
            'n_seeds': len(values)
        })
    
    stats_df = df.groupby(['method_name', 'epoch']).apply(compute_geom_stats, include_groups=False).reset_index()
    
    return stats_df


def _get_model_from_method_name(method_name):
    """
    Extract model name from method_name.
    
    Example:
        "snarenet_pinv_lambda0.01_adaRel500_soft100" -> "snarenet"
    """
    return method_name.split('_')[0]


def _get_method_params_from_method_name(method_name):
    """
    Extract method parameters from method_name.
    
    Example:
        "snarenet_pinv_lambda0.01_adaRel500_soft100" -> "pinv_lambda0.01_adaRel500_soft100"
    """
    parts = method_name.split('_', 1)
    return parts[1] if len(parts) > 1 else ""


def _generate_method_display_name(method_name, methods_to_include):
    """
    Get display name for a method from methods_to_include dict.
    
    Args:
        method_name: Full method name (model_params)
        methods_to_include: Dict mapping method names to display names
    
    Returns:
        Display name string
    """
    if methods_to_include is None:
        # Fallback to old auto-generation behavior
        model = _get_model_from_method_name(method_name)
        return MODEL_TO_DISPLAYNAME.get(model, model)
    
    return methods_to_include.get(method_name, method_name)


def _get_method_to_color(history_df, methods_to_include=None):
    """
    Generate a color mapping for each unique method in the history dataframe.
    Methods with the same model share the same base color, with different shades.
    Within the same model, methods appearing earlier are lighter and later ones are darker.
    
    Args:
        history_df: History dataframe with method_name and model columns
        methods_to_include: Optional dict mapping method names to display names.
                           If provided, colors will follow the order of this dict.
    
    Returns:
        dict: Mapping from method_name to color (hex string or rgba tuple)
    """
    if methods_to_include is not None:
        # Use the order from methods_to_include (dict maintains insertion order in Python 3.7+)
        methods = list(methods_to_include.keys())
    else:
        # Fall back to sorted alphabetical order
        methods = sorted(history_df['method_name'].unique())
    
    # Get the model for each method from the dataframe
    method_to_model = {}
    for method in methods:
        # Get the model from the filtered dataframe
        model_rows = history_df[history_df['method_name'] == method]['model']
        if not model_rows.empty:
            method_to_model[method] = model_rows.iloc[0]
        else:
            # Fallback: extract from method name (first part before underscore)
            method_to_model[method] = method.split('_')[0]
    
    # Group methods by model, preserving order
    model_to_methods = defaultdict(list)
    for method in methods:
        model = method_to_model[method]
        model_to_methods[model].append(method)
    
    # Assign base colors to unique models
    # Use MODEL_TO_COLOR if model is defined there, otherwise assign colors in order of first appearance
    unique_models = []
    for method in methods:
        model = method_to_model[method]
        if model not in unique_models:
            unique_models.append(model)
    
    model_to_base_color = {}
    color_index = 0
    for model in unique_models:
        if model in MODEL_TO_COLOR:
            # Use predefined color from MODEL_TO_COLOR
            model_to_base_color[model] = MODEL_TO_COLOR[model]
        else:
            # Assign color from default cycle
            model_to_base_color[model] = f'C{color_index}'
            color_index += 1
    
    # Create color map with shades for each method
    color_map = {}
    for model, model_methods in model_to_methods.items():
        base_color = model_to_base_color[model]
        n_methods = len(model_methods)
        
        if n_methods == 1:
            # Single method for this model - use the base color
            color_map[model_methods[0]] = base_color
        else:
            # Multiple methods - create shades from light to dark
            # Convert base color to RGB
            rgb = mcolors.to_rgb(base_color)
            
            for i, method in enumerate(model_methods):
                # Create shade: earlier methods are lighter, later are darker
                # Use more moderate range to keep all colors visible
                t = i / (n_methods - 1)  # 0 for first, 1 for last
                
                # For lighter shades: blend with white (reduced from 0.6 to 0.35)
                # For darker shades: darken the color (reduced from 1.0 to 0.6)
                if t < 0.5:
                    # Lighter half: interpolate towards white
                    # Reduced maximum white blend to keep colors more visible
                    white_blend = (1 - 2*t) * 0.35  # 0.35 at start, 0 at middle
                    shaded_rgb = tuple(c + (1 - c) * white_blend for c in rgb)
                else:
                    # Darker half: darken the color
                    # More moderate darkening to maintain better balance
                    dark_factor = 1 - (t - 0.5) * 0.6  # 1.0 at middle, 0.7 at end
                    shaded_rgb = tuple(c * dark_factor for c in rgb)
                
                color_map[method] = shaded_rgb
    
    return color_map

def _compute_test_stats_df(test_df, metric_name, methods_order=None):
    """
    Compute mean and std for a test metric across seeds for each method.
    
    Args:
        test_df: Test metrics dataframe with columns (model, method_name, seed, and metric columns)
        metric_name: Metric column name to compute statistics for
        methods_order: Optional list of method names specifying the desired order.
                      If None, methods will be sorted alphabetically.
    
    Returns:
        DataFrame with columns (method_name, mean, std, n_seeds)
    """
    if metric_name not in test_df.columns:
        return pd.DataFrame()
    
    # Convert metric to numeric, coercing errors to NaN
    test_df_copy = test_df.copy()
    test_df_copy[metric_name] = pd.to_numeric(test_df_copy[metric_name], errors='coerce')
    
    # Group by method_name and compute statistics
    stats_df = test_df_copy.groupby('method_name')[metric_name].agg(['mean', 'std', 'count']).reset_index()
    stats_df['std'] = stats_df['std'].fillna(0)  # Handle single-seed case where std is NaN
    stats_df.rename(columns={'count': 'n_seeds'}, inplace=True)
    
    # Order by methods_order if provided, otherwise sort alphabetically
    if methods_order is not None:
        # Create a categorical type with the specified order
        stats_df['method_name'] = pd.Categorical(stats_df['method_name'], 
                                                  categories=methods_order, 
                                                  ordered=True)
        stats_df = stats_df.sort_values('method_name').reset_index(drop=True)
    else:
        # Sort by method_name for consistent ordering
        stats_df = stats_df.sort_values('method_name').reset_index(drop=True)
    
    return stats_df

def plot_opt_metrics(history_df, metric_agg=None, methods_to_include=None, 
                              figsize=(18, 5), save_path=None, zero_replacement=1e-16,
                              show_legend=True, use_markers=False, markevery=100, share_ylim=False,
                              suptitle=None, vline_epoch=None):
    """
    Create a 1x3 comparison plot showing optimality gap, inequality error, 
    and equality error with geometric mean and variation across seeds.
    
    This version works with the dataframe output from get_runs_dfs().
    
    Args:
        history_df: History dataframe from get_runs_dfs() with columns:
                   (model, method_name, seed, epoch, and metric columns)
        metric_agg: Aggregation type for metrics ('max' or 'gmean'). 
                   Determines which metric to plot (e.g., 'valid/opt_gap_max' vs 'valid/opt_gap_gmean').
                   Defaults to 'max' if None.
        methods_to_include: Dictionary mapping full method names to display names.
                        Format: {'method_name': 'Display Name', ...}
                        Example: {'snarenet_pinv_lambda0.01_adaRel500_rtol0.01': 'SnareNet (rtol=1e-2)',
                                 'dc3_trainCorr10_testCorr10': 'DC3'}
                        If None, plots all methods and hyperparameters.
        figsize: Figure size tuple
        save_path: Base path for saving plots (will save as .pdf and .png)
        zero_replacement: Value to replace zeros/negatives for log scale
        show_legend: Whether to include legend in the figure (default: True)
        use_markers: Whether to use different marker shapes to distinguish training strategies (default: False)
        markevery: Show marker every N points to avoid clutter (default: 100)
        share_ylim: Whether to share y-axis limits. 
                   - If True: shares limits between 2nd and 3rd subplots (ineq & eq violations)
                   - If list of indices: shares limits between specified subplots
                   - If False: each subplot has independent limits (default: False)
        suptitle: Overall title for the entire figure (default: None)
        vline_epoch: Epoch at which to draw a vertical red dashed line. If None, no line is drawn (default: None)
    
    Returns:
        fig, axes: Matplotlib figure and axes objects
    """
    if metric_agg is None:
        metric_agg = 'max'
    if metric_agg not in ['max', 'gmean']:
        raise ValueError("metric_agg must be 'max' or 'gmean'")

    # Optimization metrics and titles
    metrics = [f'valid/opt_gap_{metric_agg}', f'valid/ineq_err_{metric_agg}', f'valid/eq_err_{metric_agg}']
    titles = [f'Optimality Gap', 
              f'Inequality Violation ({metric_agg.capitalize()})', 
              f'Equality Violation ({metric_agg.capitalize()})']
    ylabel = f'{metric_agg.capitalize()} Over Instances'
    
    # Filter dataframe
    filtered_df = _filter_history_df(history_df, methods_to_include)
    
    if filtered_df.empty:
        print("No data remaining after filtering")
        return None, None
    
    # Check for NaN values or non-numeric values in metrics and collect all problematic (method, seed) combinations
    combinations_to_remove = set()
    
    for metric in metrics:
        if metric in filtered_df.columns:
            # Try to convert to numeric to catch both NaN and non-numeric strings
            numeric_values = pd.to_numeric(filtered_df[metric], errors='coerce')
            nan_mask = numeric_values.isna()
            
            if nan_mask.any():
                # Get unique (method_name, seed) combinations with NaN or non-numeric values
                nan_combinations = filtered_df.loc[nan_mask, ['method_name', 'seed']].drop_duplicates()
                
                for _, row in nan_combinations.iterrows():
                    combinations_to_remove.add((row['method_name'], row['seed']))
    
    # Print summary and remove all problematic combinations
    if combinations_to_remove:
        print("Remove the following (method, seed) due to NaN values:")
        for method_name, seed in sorted(combinations_to_remove):
            print(f"  - method={method_name}, seed={seed}")
        
        for method_name, seed in combinations_to_remove:
            filtered_df = filtered_df[~((filtered_df['method_name'] == method_name) & (filtered_df['seed'] == seed))]
    
    if filtered_df.empty:
        print("No data remaining after removing invalid values")
        return None, None
    
    # Get color mapping for methods (respects order of methods_to_include)
    color_map = _get_method_to_color(filtered_df, methods_to_include)
    
    # Create figure with extra space at bottom for legend
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 3, hspace=0.3, bottom=0.2)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    
    # Track legend entries to avoid duplication
    legend_handles = []
    legend_labels = []
    label_set = set()
    
    for ax_idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[ax_idx]
        
        # Compute statistics for this metric
        stats_df = _compute_metric_stats_df(filtered_df, metric, zero_replacement)
        
        if stats_df.empty:
            print(f"Warning: No data for metric {metric}")
            continue
        
        # Plot each method
        for method_name in methods_to_include.keys() if methods_to_include else stats_df['method_name'].unique():
            method_stats = stats_df[stats_df['method_name'] == method_name]
            
            if method_stats.empty:
                continue
            
            epochs = method_stats['epoch'].values
            gmean = method_stats['gmean'].values
            lower = method_stats['lower'].values
            upper = method_stats['upper'].values
            
            # Get color and label
            color = color_map[method_name]
            label = _generate_method_display_name(method_name, methods_to_include)
            
            # Get marker shape and size for this method
            marker = _get_marker_for_method(method_name) if use_markers else None
            markersize = _get_marker_size_for_method(method_name, methods_to_include, base_size=10) if use_markers else 6
            
            # Plot geometric mean
            line = ax.semilogy(epochs, gmean, color=color, 
                               linewidth=2, label=label, alpha=0.8,
                               marker=marker, markevery=markevery if use_markers else None,
                               markersize=markersize, markerfacecolor=color, markeredgewidth=1.5,
                               markeredgecolor='white')[0]
            
            # Collect legend entry only once (from first subplot)
            if ax_idx == 0 and label not in label_set:
                legend_handles.append(line)
                legend_labels.append(label)
                label_set.add(label)
            
            # Add shaded region for variation
            ax.fill_between(epochs, lower, upper, color=color, alpha=0.2)
        
        # Format subplot
        ax.set_xlabel('Epoch', fontsize=16, weight='normal')
        ax.set_title(title, fontsize=18, weight='normal')
        ax.grid(True, alpha=0.3, which='both')
        ax.tick_params(labelsize=12)
        
        # Add vertical line at specified epoch if requested
        if vline_epoch is not None:
            ax.axvline(x=vline_epoch, color='red', linestyle='--', linewidth=3, alpha=0.7)
        
        # Add y-axis label only to first subplot
        if ax_idx == 0:
            ax.set_ylabel(ylabel, fontsize=16, weight='normal')
    
    # Apply shared y-limits if requested
    if share_ylim:
        if isinstance(share_ylim, bool) and share_ylim:
            # Share y-limits only between second and third subplots (ineq and eq violations)
            # First subplot (optimality gap) keeps its own limits
            ylim_1 = axes[1].get_ylim()
            ylim_2 = axes[2].get_ylim()
            shared_ymin = min(ylim_1[0], ylim_2[0])
            shared_ymax = max(ylim_1[1], ylim_2[1])
            axes[1].set_ylim(shared_ymin, shared_ymax)
            axes[2].set_ylim(shared_ymin, shared_ymax)
        elif isinstance(share_ylim, list):
            # If a list is provided, share y-limits for specified subplot indices
            if len(share_ylim) > 1:
                shared_ylims = [axes[i].get_ylim() for i in share_ylim]
                shared_ymin = min(ylim[0] for ylim in shared_ylims)
                shared_ymax = max(ylim[1] for ylim in shared_ylims)
                for i in share_ylim:
                    axes[i].set_ylim(shared_ymin, shared_ymax)
    
    # Add single legend at the bottom of the figure if requested
    if show_legend:
        fig.legend(
            legend_handles, legend_labels, 
            loc='upper center', 
            ncol=min(len(legend_labels), 4), 
            fontsize=16,
            prop={'family': 'serif', 'size': 16},
            bbox_to_anchor=(0.5, 0.05), 
            frameon=True)
    
    # Add suptitle if provided
    if suptitle:
        fig.suptitle(suptitle, fontsize=20, weight='bold', y=1.05)
    
    # Save plots if path provided
    if save_path:
        plt.savefig(f'{save_path}.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(f'{save_path}.png', dpi=300, bbox_inches='tight')
        print(f"Plots saved as '{save_path}.pdf' and '{save_path}.png'")
    
    return fig, axes


def plot_legend(methods_to_include, history_df=None, test_df=None, figsize=(10, 2), 
                save_path=None, legend_ncol=None, legend_type='line', use_markers=False):
    """
    Create a standalone figure containing only the legend for the methods.
    
    This is useful when you want to save the legend separately from the main plots,
    especially when creating figures without legends using plot_opt_metrics(show_legend=False)
    or plot_test_metrics_bars(show_legend=False).
    
    Args:
        methods_to_include: Dictionary mapping full method names to display names.
                        Format: {'method_name': 'Display Name', ...}
                        Example: {'snarenet_pinv_lambda0.01_adaRel500_rtol0.01': 'SnareNet (rtol=1e-2)',
                                 'dc3_trainCorr10_testCorr10': 'DC3'}
        history_df: Optional history dataframe for determining colors consistently.
                   Should have 'method_name' and 'model' columns.
        test_df: Optional test dataframe for determining colors if history_df is not provided.
                Should have 'method_name' and 'model' columns.
        figsize: Figure size tuple (default: (10, 2))
        save_path: Base path for saving plots (will save as .pdf and .png)
        legend_ncol: Number of columns in the legend. If None, uses min(len(methods), 4)
        legend_type: Type of legend to generate. Either 'line' for line plots or 'bar' for bar plots.
                    Default is 'line'.
        use_markers: Whether to include marker shapes to distinguish training strategies (default: True)
    
    Returns:
        fig: Matplotlib figure object containing only the legend
    """
    # Get color mapping
    if history_df is not None:
        # Filter to include only the methods we want
        filtered_df = _filter_history_df(history_df, methods_to_include)
        color_map = _get_method_to_color(filtered_df, methods_to_include)
    elif test_df is not None:
        # Use test_df if provided
        if methods_to_include is not None:
            valid_method_names = list(methods_to_include.keys())
            filtered_df = test_df[test_df['method_name'].isin(valid_method_names)]
        else:
            filtered_df = test_df
        color_map = _get_method_to_color(filtered_df, methods_to_include)
    else:
        # Fallback: extract model from method name (first part before underscore)
        # and create simple color mapping
        print("Warning: Neither history_df nor test_df provided. Using simple color mapping based on method name.")
        color_map = {method: f'C{i}' for i, method in enumerate(methods_to_include.keys())}
    
    # Create legend handles and labels (preserving order from methods_to_include)
    legend_handles = []
    legend_labels = []
    
    for method_name in methods_to_include.keys():  # Iterate in dict order, not sorted
        color = color_map.get(method_name, 'black')
        label = methods_to_include[method_name]
        
        # Get marker shape and size for this method
        marker = _get_marker_for_method(method_name) if use_markers else None
        markersize = _get_marker_size_for_method(method_name, methods_to_include, base_size=10) if use_markers else 6
        
        # Create appropriate handle based on legend type
        if legend_type == 'line':
            # Create a line object for line plots
            if use_markers:
                handle = plt.Line2D([0], [0], color=color, linewidth=2, alpha=0.8,
                                   marker=marker, markersize=markersize, 
                                   markerfacecolor=color, markeredgewidth=1.5,
                                   markeredgecolor='white')
            else:
                handle = plt.Line2D([0], [0], color=color, linewidth=2, alpha=0.8)
        elif legend_type == 'bar':
            # Create a rectangle patch for bar plots
            handle = plt.Rectangle((0, 0), 1, 1, facecolor=color, alpha=0.8)
        else:
            raise ValueError(f"legend_type must be 'line' or 'bar', got '{legend_type}'")
        
        legend_handles.append(handle)
        legend_labels.append(label)
    
    # Create figure with only legend
    fig = plt.figure(figsize=figsize)
    
    # Determine number of columns
    if legend_ncol is None:
        legend_ncol = min(len(legend_labels), 4)
    
    # Add legend centered in the figure
    fig.legend(
        legend_handles, legend_labels,
        loc='center',
        ncol=legend_ncol,
        fontsize=16,
        prop={'family': 'serif', 'size': 16},
        frameon=True
    )
    
    # Remove axes
    fig.gca().set_axis_off()
    
    # Reduce margins by using tight_layout with minimal padding
    fig.tight_layout(pad=0.0)
    
    # Save if path provided
    if save_path:
        plt.savefig(f'{save_path}.pdf', dpi=300, bbox_inches='tight', pad_inches=0.0)
        plt.savefig(f'{save_path}.png', dpi=300, bbox_inches='tight', pad_inches=0.0)
        print(f"Legend saved as '{save_path}.pdf' and '{save_path}.png'")
    
    return fig


def plot_test_metrics_bars(test_df, methods_to_include=None, 
                            metrics_to_plot=None, figsize=(18, 5), save_path=None,
                            show_legend=True, legend_ncol=None, suptitle=None, share_ylim=False,
                            share_ylabel=False):
    """
    Create a bar chart comparison plot showing test metrics across methods.
    
    Args:
        test_df: Test metrics dataframe from get_runs_dfs() with columns:
                 (model, method_name, seed, and test metric columns)
        methods_to_include: Dictionary mapping full method names to display names.
                        Format: {'method_name': 'Display Name', ...}
                        If None, plots all methods.
        metrics_to_plot: List of full column names to plot (e.g., ['test/opt_gap_max', 'test/time']).
                        If None, defaults to ['test/opt_gap_max', 'test/ineq_err_max', 'test/eq_err_max',
                                              'test/ineq_err_nviol', 'test/eq_err_nviol']
        figsize: Figure size tuple
        save_path: Base path for saving plots (will save as .pdf and .png)
        show_legend: Whether to include legend in the figure (default: True)
        legend_ncol: Number of columns in the legend. If None, uses min(len(legend_labels), 4)
        suptitle: Overall title for the entire figure (default: None)
        share_ylim: Whether to share y-axis limits across all subplots.
                   - If True: all subplots share the same y-axis limits
                   - If list of indices: only specified subplots share limits
                   - If False: each subplot has independent limits (default: False)
        share_ylabel: Whether to show y-axis labels on all subplots.
                   - If True: only the first subplot shows y-axis label
                   - If False: all subplots show their own y-axis labels (default: False)
    
    Returns:
        fig, axes: Matplotlib figure and axes objects
    """
    # Default metrics to plot
    if metrics_to_plot is None:
        metrics_to_plot = ['test/opt_gap_max', 'test/ineq_err_max', 
                          'test/eq_err_max', 'test/ineq_err_nviol', 'test/eq_err_nviol']
    
    # Build metric names and titles
    metric_names = []
    titles = []
    ylabels = []
    use_log_scale = []
    
    for metric in metrics_to_plot:
        metric_names.append(metric)
        
        # Generate title from column name
        # Extract the part after the last '/'
        metric_part = metric.split('/')[-1]
        
        # Handle known patterns
        if 'opt_gap' in metric_part:
            titles.append('Optimality Gap')
            use_log_scale.append(True)
            # Determine ylabel based on aggregation
            if '_max' in metric_part:
                ylabels.append('Max Over Instances')
            elif '_gmean' in metric_part:
                ylabels.append('GMean Over Instances')
            else:
                ylabels.append('Value')
        elif metric_part.startswith('ineq_err_nviol'):
            titles.append('Inequality Num Violations')
            use_log_scale.append(False)
            ylabels.append('Count')
        elif 'ineq_err' in metric_part:
            # Extract aggregation type if present
            if '_max' in metric_part or '_gmean' in metric_part:
                agg_type = 'Max' if '_max' in metric_part else 'GMean'
                titles.append(f'Ineq. Vio. ({agg_type})')
                ylabels.append('Max Over Instances' if '_max' in metric_part else 'GMean Over Instances')
            else:
                titles.append('Ineq. Vio.')
                ylabels.append('Value')
            use_log_scale.append(True)
        elif metric_part.startswith('eq_err_nviol'):
            titles.append('Equality Num Violations')
            use_log_scale.append(False)
            ylabels.append('Count')
        elif 'eq_err' in metric_part:
            # Extract aggregation type if present
            if '_max' in metric_part or '_gmean' in metric_part:
                agg_type = 'Max' if '_max' in metric_part else 'GMean'
                titles.append(f'Eq. Vio. ({agg_type})')
                ylabels.append('Max Over Instances' if '_max' in metric_part else 'GMean Over Instances')
            else:
                titles.append('Eq. Vio.')
                ylabels.append('Value')
            use_log_scale.append(True)
        elif 'time' in metric_part.lower():
            titles.append('Time (s)')
            use_log_scale.append(False)
            ylabels.append('Seconds')
        elif 'last_iter_taken' in metric_part:
            titles.append('# of Repair Iterations')
            use_log_scale.append(False)
            ylabels.append('Iterations')
        else:
            # Generic title from metric name
            titles.append(metric_part.replace('_', ' ').title())
            use_log_scale.append(False)
            ylabels.append('Value')
    
    # Filter methods
    if methods_to_include is not None:
        valid_method_names = list(methods_to_include.keys())
        filtered_df = test_df[test_df['method_name'].isin(valid_method_names)]
    else:
        filtered_df = test_df
    
    if filtered_df.empty:
        print("No data remaining after filtering")
        return None, None
    
    # Replace 0.0 values with 1e-16 for constraint violation metrics (for log scale plotting)
    for metric in metrics_to_plot:
        if ('ineq_err' in metric or 'eq_err' in metric) and metric in filtered_df.columns:
            # Only replace for aggregated metrics (max/gmean), not count metrics (nviol)
            if 'nviol' not in metric:
                filtered_df.loc[filtered_df[metric] == 0.0, metric] = 1e-16
    
    # Get color mapping for methods (respects order of methods_to_include)
    color_map = _get_method_to_color(filtered_df, methods_to_include)
    
    # Create figure with extra space at bottom for legend
    n_metrics = len(metric_names)
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, n_metrics, wspace=0.3, hspace=0.3, bottom=0.2)
    axes = [fig.add_subplot(gs[0, i]) for i in range(n_metrics)]
    
    # Get method order from methods_to_include, otherwise sort alphabetically
    if methods_to_include is not None:
        methods_order = list(methods_to_include.keys())
    else:
        methods_order = None
    
    # Track legend entries
    legend_handles = []
    legend_labels = []
    
    for ax_idx, (metric_name, title, ylabel, use_log) in enumerate(zip(metric_names, titles, ylabels, use_log_scale)):
        ax = axes[ax_idx]
        
        # Compute statistics for this metric (respecting methods_order)
        stats_df = _compute_test_stats_df(filtered_df, metric_name, methods_order)
        
        if stats_df.empty:
            print(f"Warning: No data for metric {metric_name}")
            ax.set_visible(False)
            continue
        
        # Create bar positions
        x_pos = np.arange(len(stats_df))
        bar_width = 0.6
        
        # Plot bars
        bars = []
        for i, row in stats_df.iterrows():
            method_name = row['method_name']
            color = color_map[method_name]
            label = _generate_method_display_name(method_name, methods_to_include)
            
            bar = ax.bar(i, row['mean'], bar_width, 
                        yerr=row['std'], 
                        color=color, 
                        alpha=0.8,
                        capsize=5,
                        error_kw={'linewidth': 1.5})
            
            # Collect legend entry only from first subplot
            if ax_idx == 0:
                bars.append(bar[0])
                legend_labels.append(label)
        
        if ax_idx == 0:
            legend_handles = bars
        
        # Format subplot
        ax.set_title(title, fontsize=18, weight='normal')
        # Only show y-label on first subplot if share_ylabel is True
        if not share_ylabel or ax_idx == 0:
            ax.set_ylabel(ylabel, fontsize=16, weight='normal')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([])  # Hide x-tick labels
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(labelsize=12)
        
        # Set y-scale
        if use_log:
            ax.set_yscale('log')
    
    # Apply shared y-limits if requested
    if share_ylim:
        if isinstance(share_ylim, bool) and share_ylim:
            # Share y-limits across all subplots
            all_ylims = [ax.get_ylim() for ax in axes if ax.get_visible()]
            if all_ylims:
                shared_ymin = min(ylim[0] for ylim in all_ylims)
                shared_ymax = max(ylim[1] for ylim in all_ylims)
                for ax in axes:
                    if ax.get_visible():
                        ax.set_ylim(shared_ymin, shared_ymax)
        elif isinstance(share_ylim, list):
            # If a list is provided, share y-limits for specified subplot indices
            if len(share_ylim) > 1:
                shared_ylims = [axes[i].get_ylim() for i in share_ylim if i < len(axes) and axes[i].get_visible()]
                if shared_ylims:
                    shared_ymin = min(ylim[0] for ylim in shared_ylims)
                    shared_ymax = max(ylim[1] for ylim in shared_ylims)
                    for i in share_ylim:
                        if i < len(axes) and axes[i].get_visible():
                            axes[i].set_ylim(shared_ymin, shared_ymax)
    
    # Add single legend at the bottom of the figure if requested
    if show_legend and legend_handles:
        ncol = legend_ncol if legend_ncol is not None else min(len(legend_labels), 4)
        fig.legend(
            legend_handles, legend_labels, 
            loc='upper center', 
            ncol=ncol, 
            fontsize=16,
            prop={'family': 'serif', 'size': 16},
            bbox_to_anchor=(0.5, 0.05), 
            frameon=True)
    
    # Add suptitle if provided
    if suptitle:
        fig.suptitle(suptitle, fontsize=20, weight='bold', y=1.05)
    
    # Save plots if path provided
    if save_path:
        plt.savefig(f'{save_path}.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(f'{save_path}.png', dpi=300, bbox_inches='tight')
        print(f"Plots saved as '{save_path}.pdf' and '{save_path}.png'")
    
    return fig, axes


def plot_test_metrics_bars_groups(test_dfs_dict, methods_to_include=None, 
                                   metrics_to_plot=None, figsize=(18, 5), save_path=None,
                                   show_legend=True, legend_ncol=None, group_xlabel='Group',
                                   share_ylim_first_n=None, share_ylim_indices=None, quartiles=False):
    """
    Create grouped bar chart comparing methods across different problem classes/configurations.
    
    Args:
        test_dfs_dict: Dictionary mapping group labels to test dataframes.
                      Format: {group_label: test_df, ...}
                      Example: {'10 Ineq': test_df10, '50 Ineq': test_df50, '100 Ineq': test_df100}
        methods_to_include: Dictionary mapping full method names to display names.
                        Format: {'method_name': 'Display Name', ...}
                        If None, plots all methods.
        metrics_to_plot: List of full column names to plot (e.g., ['test/opt_gap_max', 'test/time']).
                        If None, defaults to ['test/opt_gap_max', 'test/ineq_err_max', 'test/eq_err_max']
        figsize: Figure size tuple
        save_path: Base path for saving plots (will save as .pdf and .png)
        show_legend: Whether to include legend in the figure (default: True)
        legend_ncol: Number of columns in the legend. If None, uses min(len(legend_labels), 4)
        group_xlabel: Label for x-axis describing what groups represent (e.g., 'Number of Inequalities')
        share_ylim_first_n: If specified, the first N subplots will share the same y-axis limits
        share_ylim_indices: If specified, list of subplot indices that should share y-limits (takes precedence over share_ylim_first_n)
        quartiles: If True, plot interquartile range (Q1-median-Q3) as error bars instead of standard deviation
    
    Returns:
        fig, axes: Matplotlib figure and axes objects
    """
    # Default metrics to plot
    if metrics_to_plot is None:
        metrics_to_plot = ['test/opt_gap_max', 'test/ineq_err_max', 'test/eq_err_max']
    
    # Build metric names and titles
    metric_names = []
    titles = []
    ylabels = []
    use_log_scale = []
    
    for metric in metrics_to_plot:
        metric_names.append(metric)
        
        # Generate title from column name
        metric_part = metric.split('/')[-1]
        
        # Handle known patterns
        if 'opt_gap' in metric_part:
            titles.append('Optimality Gap (↓)')
            use_log_scale.append(True)
            if '_max' in metric_part:
                ylabels.append('Max Over Instances')
            elif '_gmean' in metric_part:
                ylabels.append('GMean Over Instances')
            else:
                ylabels.append('Value')
        elif metric_part.startswith('ineq_err_nviol'):
            titles.append('Inequality Num Violations')
            use_log_scale.append(False)
            ylabels.append('Count')
        elif 'ineq_err' in metric_part:
            if '_max' in metric_part or '_gmean' in metric_part:
                agg_type = 'Max' if '_max' in metric_part else 'GMean'
                titles.append(f'Inequality Violation ({agg_type})')
                ylabels.append('Max Over Instances' if '_max' in metric_part else 'GMean Over Instances')
                # ylabels.append("")
            else:
                titles.append('Inequality Violation')
                ylabels.append('Value')
            use_log_scale.append(True)
        elif metric_part.startswith('eq_err_nviol'):
            titles.append('Equality Num Violations')
            use_log_scale.append(False)
            ylabels.append('Count')
        elif 'eq_err' in metric_part:
            if '_max' in metric_part or '_gmean' in metric_part:
                agg_type = 'Max' if '_max' in metric_part else 'GMean'
                titles.append(f'Equality Violation ({agg_type})')
                # ylabels.append('Max Over Instances' if '_max' in metric_part else 'GMean Over Instances')
                ylabels.append("")
            else:
                titles.append('Equality Violation')
                ylabels.append('Value')
            use_log_scale.append(True)
        elif 'time' in metric_part.lower():
            titles.append('Time (↓)')
            # use_log_scale.append(False)
            use_log_scale.append(True)
            ylabels.append('Seconds')
        elif 'last_iter_taken' in metric_part:
            titles.append('# of Repair Iterations')
            use_log_scale.append(False)
            ylabels.append('Iterations')
        elif 'infeasible' in metric_part:
            titles.append('# Infeasible Instances')
            use_log_scale.append(False)
            ylabels.append('Count')
        elif 'feasible_rate' in metric_part:
            titles.append('Feasibility Rate (↑)')
            use_log_scale.append(False)
            ylabels.append('Percentage')
        else:
            titles.append(metric_part.replace('_', ' ').title())
            use_log_scale.append(False)
            ylabels.append('Value')
    
    # Get method order from methods_to_include
    if methods_to_include is not None:
        methods_order = list(methods_to_include.keys())
    else:
        # Get all unique methods across all dataframes
        all_methods = set()
        for test_df in test_dfs_dict.values():
            all_methods.update(test_df['method_name'].unique())
        methods_order = sorted(list(all_methods))
    
    # Get group labels in order
    group_labels = list(test_dfs_dict.keys())
    n_groups = len(group_labels)
    n_methods = len(methods_order)
    
    # Use the first non-empty dataframe for color mapping
    sample_df = None
    for test_df in test_dfs_dict.values():
        if not test_df.empty:
            sample_df = test_df
            break
    
    if sample_df is None:
        print("All dataframes are empty")
        return None, None
    
    # Replace 0.0 values with 1e-16 for constraint violation metrics (for log scale plotting)
    for group_label, test_df in test_dfs_dict.items():
        for metric in metrics_to_plot:
            if ('ineq_err' in metric or 'eq_err' in metric) and metric in test_df.columns:
                # Only replace for aggregated metrics (max/gmean), not count metrics (nviol)
                if 'nviol' not in metric:
                    test_df.loc[test_df[metric] == 0.0, metric] = 1e-16
    
    # Detect and warn about NaN values
    nan_combinations = []
    for group_label, test_df in test_dfs_dict.items():
        for metric in metrics_to_plot:
            if metric not in test_df.columns:
                continue
            # Convert to numeric to detect non-numeric values
            numeric_values = pd.to_numeric(test_df[metric], errors='coerce')
            nan_mask = numeric_values.isna()
            if nan_mask.any():
                # Get affected (method, seed) combinations
                affected_rows = test_df[nan_mask]
                for _, row in affected_rows.iterrows():
                    method_name = row['method_name']
                    seed = row['seed']
                    combo = (method_name, seed, group_label, metric)
                    if combo not in nan_combinations:
                        nan_combinations.append(combo)
    
    if nan_combinations:
        print("Remove the following (method, seed, group) due to NaN values:")
        for method_name, seed, group_label, metric in nan_combinations:
            print(f"  - {method_name}, seed={seed}, group={group_label}, metric={metric}")
        
        # Remove affected (method, seed, group) combinations
        for method_name, seed, group_label, metric in nan_combinations:
            test_df = test_dfs_dict[group_label]
            mask = (test_df['method_name'] == method_name) & (test_df['seed'] == seed)
            test_dfs_dict[group_label] = test_df[~mask]
    
    # Get color mapping for methods
    color_map = _get_method_to_color(sample_df, methods_to_include)
    
    # Create figure with 2x2 layout
    n_metrics = len(metric_names)
    fig = plt.figure(figsize=figsize)
    
    # Determine layout based on number of metrics
    if n_metrics == 4:
        # 2x2 layout for 4 metrics
        gs = fig.add_gridspec(2, 2, wspace=0.3, hspace=0.35, bottom=0.15)
        axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
                fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    else:
        # Default 1-row layout for other cases
        gs = fig.add_gridspec(1, n_metrics, wspace=0.4, hspace=0.3, bottom=0.2)
        axes = [fig.add_subplot(gs[0, i]) for i in range(n_metrics)]
    
    # Track legend entries
    legend_handles = []
    legend_labels = []
    
    # Determine which axes should share y-limits
    if share_ylim_indices is not None:
        # Use explicit list of indices
        axes_to_share = share_ylim_indices
    elif share_ylim_first_n and share_ylim_first_n > 0:
        # Use first N axes
        axes_to_share = list(range(share_ylim_first_n))
    else:
        axes_to_share = []
    
    # Track y-limits for shared axes
    if axes_to_share:
        shared_ylims = [float('inf'), float('-inf')]  # [min, max]
        shared_axes_use_log = None
    
    for ax_idx, (metric_name, title, ylabel, use_log) in enumerate(zip(metric_names, titles, ylabels, use_log_scale)):
        ax = axes[ax_idx]
        
        # Calculate bar positions
        bar_width = 0.8 / n_methods  # Total width of 0.8 divided by number of methods
        group_positions = np.arange(n_groups)
        
        # Plot bars for each method
        for method_idx, method_name in enumerate(methods_order):
            means = []
            errors = []
            
            # Collect statistics across groups
            for group_label in group_labels:
                test_df = test_dfs_dict[group_label]
                
                # Filter for this method
                if methods_to_include is not None:
                    valid_method_names = list(methods_to_include.keys())
                    filtered_df = test_df[test_df['method_name'].isin(valid_method_names)]
                else:
                    filtered_df = test_df
                
                # Get stats for this method and metric
                method_df = filtered_df[filtered_df['method_name'] == method_name]
                
                if not method_df.empty and metric_name in method_df.columns:
                    # Convert to numeric to handle any non-numeric values
                    values = pd.to_numeric(method_df[metric_name], errors='coerce').values
                    if len(values) > 0:
                        mean_val = np.nanmean(values)
                        
                        if quartiles:
                            # Compute interquartile range
                            q1 = np.nanpercentile(values, 0)
                            q3 = np.nanpercentile(values, 100)
                            median_val = np.nanpercentile(values, 50)
                            # Error bars represent distance from median to quartiles
                            lower_error = mean_val - q1
                            upper_error = q3 - mean_val
                            error_val = np.array([lower_error, upper_error])
                        else:
                            # Use standard deviation
                            std_val = np.nanstd(values)
                            error_val = std_val
                    else:
                        mean_val = 0
                        error_val = 0 if not quartiles else np.array([0, 0])
                else:
                    mean_val = 0
                    error_val = 0 if not quartiles else np.array([0, 0])
                
                # For log scale, replace zero with a small floor value for visualization
                if use_log and mean_val == 0:
                    mean_val = 1e-16
                
                means.append(mean_val)
                errors.append(error_val)
            
            # Calculate x positions for this method's bars
            x_positions = group_positions + (method_idx - n_methods/2 + 0.5) * bar_width
            
            # Get color and label
            color = color_map.get(method_name, f'C{method_idx}')
            label = _generate_method_display_name(method_name, methods_to_include)
            
            # Plot bars with appropriate error bars
            if quartiles:
                # For quartiles, we need asymmetric error bars
                # Separate lower and upper errors
                lower_errors = np.array([e[0] if isinstance(e, np.ndarray) else 0 for e in errors])
                upper_errors = np.array([e[1] if isinstance(e, np.ndarray) else 0 for e in errors])
                bars = ax.bar(x_positions, means, bar_width, 
                             yerr=[lower_errors, upper_errors], 
                             color=color, 
                             alpha=0.8,
                             capsize=3,
                             error_kw={'linewidth': 1.5},
                             label=label)
            else:
                # For standard deviation, use symmetric error bars
                bars = ax.bar(x_positions, means, bar_width, 
                             yerr=errors, 
                             color=color, 
                             alpha=0.8,
                             capsize=3,
                             error_kw={'linewidth': 1.5},
                             label=label)
            
            # Collect legend entry only from first subplot
            if ax_idx == 0:
                legend_handles.append(bars[0])
                legend_labels.append(label)
        
        # Format subplot
        ax.set_title(title, fontsize=18, weight='normal')
        ax.set_ylabel(ylabel, fontsize=16, weight='normal')
        ax.set_xlabel(r'$n_{\text{ineq}}$', fontsize=16, weight='normal')
        ax.set_xticks(group_positions)
        ax.set_xticklabels(group_labels, fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(labelsize=12)
        
        # Set y-scale
        if use_log:
            ax.set_yscale('log')
        
        # Track y-limits for sharing if applicable (after setting scale)
        if axes_to_share and ax_idx in axes_to_share:
            if ax_idx == axes_to_share[0]:
                shared_axes_use_log = use_log
            current_ylim = ax.get_ylim()
            # For log scale, ensure we only track positive limits
            if use_log:
                current_ylim = (max(current_ylim[0], 1e-16), current_ylim[1])
            shared_ylims[0] = min(shared_ylims[0], current_ylim[0])
            shared_ylims[1] = max(shared_ylims[1], current_ylim[1])
    
    # Apply shared y-limits to specified subplots
    if axes_to_share:
        # Ensure limits are positive for log scale, and set minimum below floor value
        if shared_axes_use_log:
            if shared_ylims[0] <= 0:
                shared_ylims[0] = 1e-17
            # If minimum is at the floor value (1e-16), lower it to make bars visible
            elif shared_ylims[0] >= 1e-16:
                shared_ylims[0] = 1e-17
        for ax_idx in axes_to_share:
            if ax_idx < len(axes):
                axes[ax_idx].set_ylim(shared_ylims)
    
    # Add legend at the bottom if requested
    if show_legend and legend_handles:
        ncol = legend_ncol if legend_ncol is not None else min(len(legend_labels), 4)
        fig.legend(
            legend_handles, legend_labels, 
            loc='upper center', 
            ncol=ncol, 
            fontsize=16,
            prop={'family': 'serif', 'size': 16},
            bbox_to_anchor=(0.5, 0.05), 
            frameon=True)
    
    # Save plots if path provided
    if save_path:
        plt.savefig(f'{save_path}.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(f'{save_path}.png', dpi=300, bbox_inches='tight')
        print(f"Plots saved as '{save_path}.pdf' and '{save_path}.png'")
    
    return fig, axes


def _format_table_cell(mean_val, std_val, use_scientific=True, precision=2):
    """
    Format a table cell value with optional standard deviation.
    
    Args:
        mean_val: Mean value
        std_val: Standard deviation value (can be NaN if only one sample)
        use_scientific: Whether to use scientific notation
        precision: Number of decimal places
    
    Returns:
        Formatted LaTeX string for the table cell
    """
    # Check if std_val is NaN or zero (only show mean in these cases)
    show_std = not (np.isnan(std_val) or std_val == 0)
    
    if use_scientific:
        # Format in scientific notation: (value ± std) × 10^{power}
        # Handle exactly zero or near-zero values
        if abs(mean_val) < 1e-16:
            return "$0.00$"
        
        mean_exp = int(np.floor(np.log10(abs(mean_val))))
        mean_mantissa = mean_val / (10 ** mean_exp)
        
        # Don't show × 10^0 when exponent is 0
        if mean_exp == 0:
            if show_std:
                std_mantissa = std_val / (10 ** mean_exp)
                return f"${mean_mantissa:.{precision}f} \\pm {std_mantissa:.{precision}f}$"
            else:
                return f"${mean_mantissa:.{precision}f}$"
        else:
            if show_std:
                std_mantissa = std_val / (10 ** mean_exp)
                return f"$({mean_mantissa:.{precision}f} \\pm {std_mantissa:.{precision}f}) \\times 10^{{{mean_exp}}}$"
            else:
                return f"${mean_mantissa:.{precision}f} \\times 10^{{{mean_exp}}}$"
    else:
        # Regular formatting without scientific notation
        # Check if value rounds to zero but isn't exactly zero
        if mean_val != 0 and abs(mean_val) < 0.5 / (10 ** precision):
            # Would round to zero - use scientific notation instead
            mean_exp = int(np.floor(np.log10(abs(mean_val))))
            mean_mantissa = mean_val / (10 ** mean_exp)
            if show_std:
                std_mantissa = std_val / (10 ** mean_exp)
                return f"$({mean_mantissa:.{precision}f} \\pm {std_mantissa:.{precision}f}) \\times 10^{{{mean_exp}}}$"
            else:
                return f"${mean_mantissa:.{precision}f} \\times 10^{{{mean_exp}}}$"
        else:
            # Regular formatting
            if show_std:
                return f"${mean_val:.{precision}f} \\pm {std_val:.{precision}f}$"
            else:
                return f"${mean_val:.{precision}f}$"


def generate_test_metrics_latex_table_from_df(test_df, methods_to_include=None, 
                                                include_opt_gap=True, precision=2):
    """
    Generate LaTeX table code for test set metrics from test dataframe.
    
    This version works with the test_df output from get_runs_dfs().
    
    Args:
        test_df: Test metrics dataframe from get_runs_dfs() with columns:
                (model, method_name, seed, and test metric columns)
        methods_to_include: Dictionary mapping full method names to display names.
                           Format: {'method_name': 'Display Name', ...}
                           Example: {'snarenet_pinv_lambda0.01_adaRel500_rtol0.01': 'SnareNet (rtol=1e-2)',
                                    'dc3_trainCorr10_testCorr10': 'DC3'}
                           If None, includes all methods and hyperparameters.
        include_opt_gap: Whether to include optimality gap columns (default: True)
        precision: Number of decimal places for scientific notation exponent (default: 2)
    
    Returns:
        str: LaTeX table code
    """    
    # Filter test dataframe
    if methods_to_include is not None:
        valid_method_names = list(methods_to_include.keys())
        filtered_df = test_df[test_df['method_name'].isin(valid_method_names)]
    else:
        filtered_df = test_df
    
    if filtered_df.empty:
        print("No data remaining after filtering")
        return ""
    
    # Metrics to include in table: (metric_name, column_header, use_scientific)
    base_metrics = [    
        # ('test/eval',           'Obj. Value', False),
        ('test/ineq_err_max',   'Max Ineq. Error', True),
        ('test/ineq_err_gmean', 'GMean Ineq. Error', True),
        ('test/ineq_err_nviol', '\# Ineq Violations', False),
        ('test/eq_err_max',     'Max Eq. Error', True),
        ('test/eq_err_gmean',   'GMean Eq. Error', True),
        ('test/eq_err_nviol',   '\# Eq Violations', False),
        ('test/time',           'Test Time (s)', False),
    ]
    
    if include_opt_gap:
        opt_gap_metrics = [
            ('test/opt_gap_max',    'Max Opt. Gap', True),
            ('test/opt_gap_gmean',  'GMean Opt. Gap', True),
        ]
        # Insert opt gap metrics after eval
        metrics = opt_gap_metrics + base_metrics[1:]
    else:
        metrics = base_metrics
    
    # Collect data for each method
    table_data = []
    for method_name in sorted(filtered_df['method_name'].unique()):
        method_df = filtered_df[filtered_df['method_name'] == method_name]
        
        # Get model and method params
        model = method_df['model'].iloc[0]
        method_params = '_'.join(method_name.split('_')[1:])
        
        # Get test metrics across seeds
        row_data = {
            'model': model,
            'method_name': method_name,
            'method_params': method_params,
        }
        
        # Compute mean and std for each metric
        for metric_name, _, _ in metrics:
            if metric_name in method_df.columns:
                # Convert to numeric, coercing errors to NaN
                values = pd.to_numeric(method_df[metric_name], errors='coerce').dropna()
                if len(values) > 0:
                    mean_val = values.mean()
                    std_val = values.std()
                    row_data[metric_name] = (mean_val, std_val)
                else:
                    row_data[metric_name] = None
            else:
                row_data[metric_name] = None
        
        table_data.append(row_data)
    
    # Generate LaTeX code
    latex_lines = []
    
    # Table header
    latex_lines.append("\\begin{table*}[h]")
    latex_lines.append("\\centering")
    latex_lines.append("\\caption{Evaluation metrics on the test set. Values shown as mean $\\pm$ std across random seeds.}")
    latex_lines.append("\\label{tab:test_metrics}")
    latex_lines.append("\\resizebox{\\linewidth}{!}{")
    latex_lines.append("\\begingroup")
    latex_lines.append("\\begin{tabular}{l" + "c" * len(metrics) + "}")
    latex_lines.append("\\toprule")
    
    # Column headers
    header_line = "Method"
    for _, col_header, _ in metrics:
        header_line += f" & {col_header}"
    header_line += " \\\\"
    latex_lines.append(header_line)
    latex_lines.append("\\midrule")
    
    # Data rows
    for row in table_data:
        # Generate display name
        method_label = _generate_method_display_name(row['method_name'], methods_to_include)
        # Escape underscores for LaTeX
        method_label = method_label.replace('_', '\\_')
        
        row_line = method_label
        
        for metric_name, _, use_scientific in metrics:
            if row[metric_name] is not None:
                mean_val, std_val = row[metric_name]
                cell_text = _format_table_cell(mean_val, std_val, use_scientific, precision)
                row_line += f" & {cell_text}"
            else:
                row_line += " & --"
        
        row_line += " \\\\"
        latex_lines.append(row_line)
    
    # Table footer
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\endgroup")
    latex_lines.append("}")
    latex_lines.append("\\end{table*}")
    
    return "\n".join(latex_lines)
