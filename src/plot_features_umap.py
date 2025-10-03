#!/usr/bin/env python
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--metrics_csv',
        default=os.path.join(os.path.dirname(__file__), '..', 'data', 'metrics.csv'),
        help='Path to CSV with original features (filename, AF, MI, BI, OR, lambda)'
    )
    parser.add_argument(
        '--umap_csv',
        default=os.path.join(os.path.dirname(__file__), '..', 'user_data', 'VAE', 'features_umap','metrics_umap.csv'),
        help='Path to CSV with U1 and U2 (filename, U1, U2)'
    )
    parser.add_argument(
        '--out_dir',
        default=os.path.join(os.path.dirname(__file__), '..', 'user_data', 'VAE', 'features_umap', 'plots'),
        help='Directory to save scatter plots'
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load data
    df_feat = pd.read_csv(args.metrics_csv)
    df_umap = pd.read_csv(args.umap_csv)
    df = pd.merge(
        df_umap,
        df_feat[['filename', 'AF', 'MI', 'BI', 'ORI', 'lambda']],
        on='filename'
    )

    features = ['AF', 'MI', 'BI', 'ORI', 'lambda']
    for feat in features:
        plt.figure(figsize=(6,6))
        sc = plt.scatter(
            df['U1'], df['U2'], c=df[feat], cmap='viridis', s=5, alpha=0.7
        )
        plt.colorbar(sc, label=feat)
        plt.title(f'UMAP (U1 vs U2) colored by {feat}')
        plt.xlabel('U1')
        plt.ylabel('U2')
        out_path = os.path.join(args.out_dir, f'umap_{feat}.png')
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"Saved plot for {feat} at {out_path}")