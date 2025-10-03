#!/usr/bin/env python
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set(style='whitegrid')


def main():
    parser = argparse.ArgumentParser(
        description='Plot UMAP-based MCMC results from multi-chain outputs'
    )
    parser.add_argument(
        '--mcmc_dir',
        default=os.path.join(os.path.dirname(__file__), '..', 'user_data', 'mcmc_out'),
        help='Directory containing chain_N/chain.csv files'
    )
    args = parser.parse_args()
    parser.add_argument(
        '--out_dir',
        default=os.path.join(args.mcmc_dir,'mcmc_plots'),
        help='Directory to save output plots'
    )
    args = parser.parse_args()

    # Prepare output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Aggregate all chain CSVs
    dfs = []
    for chain_path in sorted(Path(args.mcmc_dir).iterdir()):
        csv_path = chain_path / 'chain.csv'
        if csv_path.exists():
            df_chain = pd.read_csv(csv_path)
            df_chain['chain'] = chain_path.name
            dfs.append(df_chain)
    if not dfs:
        print(f"No chain.csv files found under {args.mcmc_dir}")
        return
    df = pd.concat(dfs, ignore_index=True)

    # Extract UMAP coords and loss
    x = df['U1']
    y = df['U2']
    loss = df['loss']

    # Determine common axis limits
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    '''
    # 1) Scatter plot colored by loss
    fig, ax = plt.subplots(figsize=(6, 6))
    sc = ax.scatter(x, y, c=loss, cmap='magma_r', s=5, alpha=0.7)
    fig.colorbar(sc, ax=ax, label='Loss')
    ax.set_title('MCMC UMAP Scatter by Loss')
    ax.set_xlabel('U1')
    ax.set_ylabel('U2')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.axis('equal')
    scatter_path = out_dir / 'mcmc_umap_scatter.png'
    fig.savefig(scatter_path, dpi=300)
    plt.close(fig)
    print(f"Saved scatter plot to {scatter_path}")
    '''
    # 2) KDE contour with loss scatter and best star overlay
    fig, ax = plt.subplots(figsize=(6, 6))
    # KDE background
    sns.kdeplot(x=x, y=y, levels=15, fill=True, cmap='Reds', ax=ax)
    # Scatter colored by loss
    sc_kde = ax.scatter(x, y, c=loss, cmap='magma_r', s=2, alpha=0.7)
    fig.colorbar(sc_kde, ax=ax, label='Loss')
    # Highlight best-loss point
    idx_best = df['loss'].idxmin()
    best = df.loc[idx_best]
    best_loss = best['loss']
    ax.scatter(best['U1'], best['U2'], c='green', s=100, marker='*', label='Best sample')
    ax.legend(loc='upper left')
    # Annotate best loss at top-right

    kde_path = out_dir / 'mcmc_umap_kde_scatter_best.png'
    fig.savefig(kde_path, dpi=300)
    plt.close(fig)
    print(f"Saved KDE+Scatter+Best plot to {kde_path}")
    print(f"Saved KDE contour to {kde_path}")

    '''
    # 3) Highlight best-loss point
    idx_best = df['loss'].idxmin()
    best = df.loc[idx_best]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(x, y, c='lightgray', s=5, alpha=0.5)
    ax.scatter(best['U1'], best['U2'], c='red', s=60, marker='*')
    ax.set_title('Best MCMC Sample (Red Star)')
    ax.set_xlabel('U1')
    ax.set_ylabel('U2')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.axis('equal')
    best_path = out_dir / 'mcmc_umap_best.png'
    fig.savefig(best_path, dpi=300)
    plt.close(fig)
    print(f"Saved best-sample plot to {best_path}")
    '''
    # 4) Combined figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)
    # Scatter
    axes[0].scatter(x, y, c=loss, cmap='magma_r', s=5, alpha=0.7)
    axes[0].set_title('Scatter by Loss')
    # KDE
    sns.kdeplot(x=x, y=y, levels=10, fill=True, cmap='Reds', ax=axes[1])
    axes[1].set_title('KDE Contour')
    # Best point
    axes[2].scatter(x, y, c='lightgray', s=5, alpha=0.5)
    axes[2].scatter(best['U1'], best['U2'], c='green', s=100, marker='*', label='Best sample')
    axes[2].set_title('Best Sample')
    for ax in axes:
        ax.set_xlabel('U1')
        ax.set_ylabel('U2')
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal', adjustable='box')
        ax.legend(loc='upper left')
    combined_path = out_dir / 'mcmc_umap_all.png'
    fig.tight_layout()
    fig.savefig(combined_path, dpi=300)
    plt.close(fig)
    print(f"Saved combined plot to {combined_path}")


if __name__ == '__main__':
    main()
