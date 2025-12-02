#!/usr/bin/env python3
"""
Analyze training statistics from saved pickle files.

Usage:
    python scripts/analyze_training_stats.py <path_to_trainingStats_file>
    python scripts/analyze_training_stats.py <path1> <path2> ...  # Compare multiple runs
    python scripts/analyze_training_stats.py --dir <log_directory>  # Analyze all in directory
"""

import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys


def load_stats(filepath):
    """Load training stats from pickle file."""
    with open(filepath, 'rb') as f:
        stats = pickle.load(f)
    return stats


def print_statistics(stats, name="Run"):
    """Print summary statistics for a training run."""
    test_loss = stats['testLoss']
    test_cer = stats['testCER']
    
    print(f"\n{'='*60}")
    print(f"{name}")
    print(f"{'='*60}")
    
    print(f"\nTest Loss:")
    print(f"  Initial:  {test_loss[0]:.6f}")
    print(f"  Final:    {test_loss[-1]:.6f}")
    print(f"  Best:     {np.min(test_loss):.6f} (at evaluation {np.argmin(test_loss)})")
    print(f"  Worst:    {np.max(test_loss):.6f}")
    print(f"  Mean:     {np.mean(test_loss):.6f}")
    print(f"  Std:      {np.std(test_loss):.6f}")
    
    print(f"\nCharacter Error Rate (CER):")
    print(f"  Initial:  {test_cer[0]:.6f}")
    print(f"  Final:    {test_cer[-1]:.6f}")
    print(f"  Best:     {np.min(test_cer):.6f} (at evaluation {np.argmin(test_cer)})")
    print(f"  Worst:    {np.max(test_cer):.6f}")
    print(f"  Mean:     {np.mean(test_cer):.6f}")
    print(f"  Std:      {np.std(test_cer):.6f}")
    
    print(f"\nTraining Progress:")
    print(f"  Total evaluations: {len(test_loss)}")
    print(f"  Estimated batches: {len(test_loss) * 100}")  # Assuming eval every 100 batches
    print(f"  Improvement (loss): {test_loss[0] - test_loss[-1]:.6f} ({((test_loss[0] - test_loss[-1]) / test_loss[0] * 100):.2f}%)")
    print(f"  Improvement (CER):  {test_cer[0] - test_cer[-1]:.6f} ({((test_cer[0] - test_cer[-1]) / test_cer[0] * 100):.2f}%)")


def plot_single_run(stats, name="Training Run", save_path=None):
    """Plot learning curves for a single run."""
    test_loss = stats['testLoss']
    test_cer = stats['testCER']
    
    # Convert evaluation index to batch number (assuming eval every 100 batches)
    batches = np.arange(len(test_loss)) * 100
    
    # Plot loss on separate figure
    fig1, ax1 = plt.subplots(1, 1, figsize=(10, 6))
    ax1.plot(batches, test_loss, 'b-', linewidth=2, label='Test Loss')
    ax1.axhline(y=np.min(test_loss), color='r', linestyle='--', alpha=0.5, label=f'Best: {np.min(test_loss):.6f}')
    ax1.set_xlabel('Batch Number')
    ax1.set_ylabel('Test Loss')
    ax1.set_title(f'{name} - Test Loss')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    plt.tight_layout()
    
    if save_path:
        # Save loss plot
        loss_path = save_path.replace('.png', '_loss.png') if save_path.endswith('.png') else f"{save_path}_loss.png"
        plt.savefig(loss_path, dpi=150, bbox_inches='tight')
        print(f"Loss plot saved to: {loss_path}")
    else:
        plt.show()
    
    # Plot CER on separate figure
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))
    ax2.plot(batches, test_cer, 'g-', linewidth=2, label='Test CER')
    ax2.axhline(y=np.min(test_cer), color='r', linestyle='--', alpha=0.5, label=f'Best: {np.min(test_cer):.6f}')
    ax2.set_xlabel('Batch Number')
    ax2.set_ylabel('Character Error Rate')
    ax2.set_title(f'{name} - Character Error Rate')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    plt.tight_layout()
    
    if save_path:
        # Save CER plot
        cer_path = save_path.replace('.png', '_cer.png') if save_path.endswith('.png') else f"{save_path}_cer.png"
        plt.savefig(cer_path, dpi=150, bbox_inches='tight')
        print(f"CER plot saved to: {cer_path}")
    else:
        plt.show()


def plot_comparison(stats_list, names_list, save_path=None):
    """Plot learning curves for multiple runs for comparison."""
    colors = plt.cm.tab10(np.linspace(0, 1, len(stats_list)))
    
    # Plot loss comparison on separate figure
    fig1, ax1 = plt.subplots(1, 1, figsize=(12, 6))
    for i, (stats, name, color) in enumerate(zip(stats_list, names_list, colors)):
        test_loss = stats['testLoss']
        batches = np.arange(len(test_loss)) * 100
        ax1.plot(batches, test_loss, color=color, linewidth=2, label=name, alpha=0.8)
    
    ax1.set_xlabel('Batch Number')
    ax1.set_ylabel('Test Loss')
    ax1.set_title('Test Loss Comparison')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    plt.tight_layout()
    
    if save_path:
        # Save loss comparison plot
        loss_path = save_path.replace('.png', '_loss.png') if save_path.endswith('.png') else f"{save_path}_loss.png"
        plt.savefig(loss_path, dpi=150, bbox_inches='tight')
        print(f"Loss comparison plot saved to: {loss_path}")
    else:
        plt.show()
    
    # Plot CER comparison on separate figure
    fig2, ax2 = plt.subplots(1, 1, figsize=(12, 6))
    for i, (stats, name, color) in enumerate(zip(stats_list, names_list, colors)):
        test_cer = stats['testCER']
        batches = np.arange(len(test_cer)) * 100
        ax2.plot(batches, test_cer, color=color, linewidth=2, label=name, alpha=0.8)
    
    ax2.set_xlabel('Batch Number')
    ax2.set_ylabel('Character Error Rate')
    ax2.set_title('Character Error Rate Comparison')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    plt.tight_layout()
    
    if save_path:
        # Save CER comparison plot
        cer_path = save_path.replace('.png', '_cer.png') if save_path.endswith('.png') else f"{save_path}_cer.png"
        plt.savefig(cer_path, dpi=150, bbox_inches='tight')
        print(f"CER comparison plot saved to: {cer_path}")
    else:
        plt.show()


def find_training_stats_files(directory):
    """Find all trainingStats files in a directory (recursively)."""
    directory = Path(directory)
    stats_files = list(directory.rglob('trainingStats'))
    return stats_files


def main():
    parser = argparse.ArgumentParser(
        description='Analyze training statistics from pickle files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a single run
  python scripts/analyze_training_stats.py logs/speech_logs/baseline/trainingStats
  
  # Compare multiple runs
  python scripts/analyze_training_stats.py logs/speech_logs/baseline/trainingStats logs/speech_logs/test_adamw/trainingStats
  
  # Analyze all runs in a directory
  python scripts/analyze_training_stats.py --dir logs/speech_logs/
  
  # Save plots instead of displaying
  python scripts/analyze_training_stats.py --save-plots logs/speech_logs/baseline/trainingStats
        """
    )
    parser.add_argument('files', nargs='*', help='Path(s) to trainingStats pickle file(s)')
    parser.add_argument('--dir', type=str, help='Directory to search for trainingStats files')
    parser.add_argument('--save-plots', action='store_true', help='Save plots instead of displaying')
    parser.add_argument('--no-plot', action='store_true', help='Skip plotting, only print statistics')
    parser.add_argument('--output-dir', type=str, help='Directory to save plots (default: same as stats file)')
    
    args = parser.parse_args()
    
    # Collect files to analyze
    files_to_analyze = []
    names = []
    
    if args.dir:
        stats_files = find_training_stats_files(args.dir)
        if not stats_files:
            print(f"No trainingStats files found in {args.dir}")
            sys.exit(1)
        files_to_analyze = stats_files
        # Use parent directory name as the run name
        names = [f.parent.name for f in stats_files]
    elif args.files:
        files_to_analyze = [Path(f) for f in args.files]
        names = [f.parent.name if f.parent.name else f.stem for f in files_to_analyze]
    else:
        parser.print_help()
        sys.exit(1)
    
    # Load all stats
    stats_list = []
    valid_files = []
    valid_names = []
    
    for filepath, name in zip(files_to_analyze, names):
        if not filepath.exists():
            print(f"Warning: File not found: {filepath}", file=sys.stderr)
            continue
        try:
            stats = load_stats(filepath)
            stats_list.append(stats)
            valid_files.append(filepath)
            valid_names.append(name)
        except Exception as e:
            print(f"Error loading {filepath}: {e}", file=sys.stderr)
            continue
    
    if not stats_list:
        print("No valid trainingStats files found.")
        sys.exit(1)
    
    # Print statistics for each run
    for stats, name, filepath in zip(stats_list, valid_names, valid_files):
        print_statistics(stats, name)
    
    # Plotting
    if not args.no_plot:
        if len(stats_list) == 1:
            # Single run
            save_path = None
            if args.save_plots:
                output_dir = Path(args.output_dir) if args.output_dir else valid_files[0].parent
                output_dir.mkdir(parents=True, exist_ok=True)
                save_path = output_dir / f"{valid_names[0]}_training_curves.png"
            plot_single_run(stats_list[0], valid_names[0], save_path)
        else:
            # Multiple runs - comparison
            save_path = None
            if args.save_plots:
                output_dir = Path(args.output_dir) if args.output_dir else valid_files[0].parent
                output_dir.mkdir(parents=True, exist_ok=True)
                save_path = output_dir / "training_comparison.png"
            plot_comparison(stats_list, valid_names, save_path)


if __name__ == '__main__':
    main()

