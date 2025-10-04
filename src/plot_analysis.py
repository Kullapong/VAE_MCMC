#!/usr/bin/env python
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set(style='whitegrid')

# ---- Plotting Functions ----

def plot_distributions(metrics_csv: str, out_dir: Path):
    """Plot histograms with KDE for each image feature."""
    df = pd.read_csv(metrics_csv)
    features = ['AF', 'MI', 'BI', 'ORI', 'lambda']
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()
    for ax, feat in zip(axes, features):
        sns.histplot(df[feat], kde=True, ax=ax)
        ax.set_title(f'Distribution of {feat}')
    axes[-1].axis('off')
    fig.tight_layout()
    path = os.path.join(out_dir, 'dataset_distributions.png')
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved distributions to {path}")


def plot_loss(training_log: str, out_dir: Path):
    """Plot BCE, KLD, and total loss curves over epochs."""
    df = pd.read_csv(training_log)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(df['epoch'], df['bce'], label='BCE')
    ax.plot(df['epoch'], df['kld'], label='KLD')
    ax.plot(df['epoch'], df['total'], label='Total')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.set_title('Training Loss Curves')
    path = os.path.join(out_dir, 'training_loss.png')
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved loss curves to {path}")


def plot_dataset_umap(metrics_csv: str, metrics_umap_csv: str, out_dir: Path):
    """Merge UMAP coords with image features, save combined CSV, and scatter plot by AF."""
    df_feat = pd.read_csv(metrics_csv)
    df_umap = pd.read_csv(metrics_umap_csv)
    df = pd.merge(
        df_umap,
        df_feat[['filename', 'AF', 'MI', 'BI', 'ORI', 'lambda']],
        on='filename', how='left'
    )
    # Save merged UMAP coords with features
    merged_out = os.path.join(out_dir, 'metrics_umap_features.csv')
    df.to_csv(merged_out, index=False)
    print(f"Saved merged UMAP + features to {merged_out}")

    fig, ax = plt.subplots(figsize=(6, 6))
    sc = ax.scatter(df['U1'], df['U2'], c=df['AF'], cmap='viridis', s=5)
    fig.colorbar(sc, ax=ax, label='AF')
    ax.set_title('Dataset UMAP colored by AF')
    ax.set_xlabel('U1')
    ax.set_ylabel('U2')
    ax.axis('equal')
    path = os.path.join(out_dir, 'dataset_umap_AF.png')
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved dataset UMAP to {path}")


def plot_mcmc_umap(mcmc_dir: str, out_dir: Path, plot_scatter: bool, plot_kde: bool, plot_best: bool):
    """Generate MCMC UMAP plots: scatter by loss, KDE contour, and highlight best sample."""
    dfs = []
    for chain in sorted(Path(mcmc_dir).iterdir()):
        csv_path = chain / 'chain.csv'
        if csv_path.exists():
            df_chain = pd.read_csv(csv_path)
            dfs.append(df_chain)
    if not dfs:
        print('No MCMC data found.')
        return
    df = pd.concat(dfs, ignore_index=True)
    # Save merged MCMC samples with UMAP coords and metrics
    merged_mcmc = os.path.join(out_dir, 'mcmc_umap_metrics.csv')
    df.to_csv(merged_mcmc, index=False)
    print(f"Saved merged MCMC data to {merged_mcmc}")

    x, y, loss = df['U1'], df['U2'], df['loss']

    if plot_scatter:
        fig, ax = plt.subplots(figsize=(6,6))
        sc = ax.scatter(x, y, c=loss, cmap='magma', s=5)
        fig.colorbar(sc, ax=ax, label='Loss')
        ax.set_title('MCMC UMAP scatter by loss')
        ax.axis('equal')
        path = os.path.join(out_dir, 'mcmc_umap_scatter.png')
        fig.savefig(path, dpi=300)
        plt.close(fig)
        print(f"Saved MCMC scatter to {path}")

    if plot_kde:
        fig, ax = plt.subplots(figsize=(6,6))
        sns.kdeplot(x=x, y=y, levels=10, fill=True, cmap='Reds', ax=ax)
        ax.set_title('MCMC UMAP KDE contour')
        path = os.path.join(out_dir, 'mcmc_umap_kde.png')
        fig.savefig(path, dpi=300)
        plt.close(fig)
        print(f"Saved MCMC KDE to {path}")

    if plot_best:
        idx = df['loss'].idxmin()
        best = df.loc[idx]
        fig, ax = plt.subplots(figsize=(6,6))
        ax.scatter(x, y, c='lightgray', s=5)
        ax.scatter(best['U1'], best['U2'], c='red', s=50, marker='*')
        ax.set_title('Best MCMC sample')
        ax.axis('equal')
        path = os.path.join(out_dir, 'mcmc_best_point.png')
        fig.savefig(path, dpi=300)
        plt.close(fig)
        print(f"Saved MCMC best point to {path}")


def main():
    parser = argparse.ArgumentParser(description='Comprehensive analysis plots')
    parser.add_argument('--metrics_csv', default=os.path.join('..','data', 'metrics.csv'))
    parser.add_argument('--training_log', default=os.path.join('..','user_data', 'VAE', 'training_log.csv'))
    parser.add_argument('--metrics_umap_csv', default=os.path.join('..','user_data', 'VAE', 'features_umap', 'metrics_umap.csv'))
    parser.add_argument('--mcmc_dir', default=os.path.join('..','user_data', 'mcmc_out'))
    parser.add_argument('--out_dir', default=os.path.join('..','user_data', 'plots'))

    parser.add_argument('--no_distributions', action='store_false', dest='do_dist',   help='Skip feature distributions')
    parser.add_argument('--no_loss',          action='store_false', dest='do_loss',   help='Skip loss curves')
    parser.add_argument('--no_dataset_umap',  action='store_false', dest='do_dumap',  help='Skip dataset UMAP')
    parser.add_argument('--no_mcmc_scatter',  action='store_false', dest='do_scat',   help='Skip MCMC scatter')
    parser.add_argument('--no_mcmc_kde',      action='store_false', dest='do_kde',    help='Skip MCMC KDE')
    parser.add_argument('--no_mcmc_best',     action='store_false', dest='do_best',   help='Skip best-sample plot')
    parser.set_defaults(do_dist=True, do_loss=True, do_dumap=True, do_scat=True, do_kde=True, do_best=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.do_dist:
        plot_distributions(args.metrics_csv, Path(args.out_dir))
    if args.do_loss:
        plot_loss(args.training_log, Path(args.out_dir))
    if args.do_dumap:
        plot_dataset_umap(args.metrics_csv, args.metrics_umap_csv, Path(args.out_dir))
    if args.do_scat or args.do_kde or args.do_best:
        plot_mcmc_umap(
            args.mcmc_dir,
            Path(args.out_dir),
            args.do_scat,
            args.do_kde,
            args.do_best
        )

if __name__ == '__main__':
    main()