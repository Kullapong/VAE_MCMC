#!/usr/bin/env python
import argparse
import os
import sys
import seaborn as sns
import matplotlib.pyplot as plt

# Ensure this script sees modules in src/
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from metrics import extract_metrics_from_paths

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--pattern',
        default=r'K:\Paper2\Manuscript\VAE_MCMC\data\Img\*.png',
        help='Glob for input images'
    )
    parser.add_argument(
        '--max-files',
        type=int,
        default=None,
        help='Maximum number of files to process'
    )
    parser.add_argument(
        '--output',
        default=os.path.join(script_dir, '..', 'user_data', 'metrics.csv'),
        help='Path to CSV output (default /user_data/metrics.csv)'
    )
    args = parser.parse_args()

    # Ensure output directory exists
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Extract metrics with progress indicator
    df = extract_metrics_from_paths(args.pattern, args.max_files)

    # Save CSV
    df.to_csv(args.output, index=False)
    print(f"Saved metrics to {args.output}")

    # Generate and save pair plot of all features (excluding filename)
    pairplot_path = args.output.replace('.csv', '_pairplot.png')
    sns.pairplot(df.drop(columns=['filename']))
    plt.savefig(pairplot_path)
    print(f"Saved pair plot to {pairplot_path}")

if __name__ == '__main__':
    main()