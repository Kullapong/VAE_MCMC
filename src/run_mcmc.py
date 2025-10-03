#!/usr/bin/env python
import os
import sys
# OpenMP workaround on Windows
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import argparse
import csv
import math
import random
from pathlib import Path
from multiprocessing import Process, cpu_count

import torch
import joblib

from metrics import (
    compute_area_fraction,
    compute_morans_i,
    compute_band_metrics,
    compute_thickness_lambda
)
from vae import VAE

# ─── VAE Loader ─────────────────────────────────────────────────────────

def load_vae(path: str, device: torch.device) -> VAE:
    model = VAE().to(device)
    state = torch.load(path, map_location=device)
    state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    return model

# ─── MCMC Hyperparameters ───────────────────────────────────────────────
LAT_MIN, LAT_MAX = -5.0, 5.0
SIG_MIN, SIG_MAX = 1e-4, (LAT_MAX - LAT_MIN) / 2
TARGET_ACC, ADAPT_BLOCK, ADAPT_RATE = 0.23, 100, 0.10

# ─── Utilities ──────────────────────────────────────────────────────────

def reflect(z: torch.Tensor) -> torch.Tensor:
    """Reflect z into [LAT_MIN, LAT_MAX]."""
    z = torch.where(z < LAT_MIN,  2*LAT_MIN - z, z)
    z = torch.where(z > LAT_MAX,  2*LAT_MAX - z, z)
    return z.clamp(LAT_MIN, LAT_MAX)

def loss_fn(metrics: dict, target: tuple) -> float:
    """Weighted MSE loss between metrics and targets."""
    weights = [5, 1, 1, 0.1, 1]
    keys = ['AF', 'MI', 'BI', 'Angle', 'Lambda']
    loss = 0.0
    for key, w, t in zip(keys, weights, target):
        if t is not None:
            loss += w * (metrics[key] - t) ** 2
    return loss

# ─── Single‐Chain Worker ─────────────────────────────────────────────────

def run_chain(idx: int, args, target: tuple):
    random.seed(args.seed + idx)
    torch.manual_seed(args.seed + idx)

    device = torch.device('cpu')
    vae = load_vae(args.vae_path, device)

    # prepare chain directory
    chain_dir = args.out_dir / f"chain_{idx}"
    chain_dir.mkdir(parents=True, exist_ok=True)
    csv_path = chain_dir / 'chain.csv'

    # CSV header
    header = ['iter']
    if args.save_latents:
        header += [f'z{i}' for i in range(args.latent_dim)]
    header += ['loss','AF','MI','BI','Angle','Lambda','sigma','acc_rate']

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        # initialize z
        z = reflect(torch.randn(1, args.latent_dim))
        sigma = args.sigma
        n_acc = acc_block = 0

        # compute initial metrics
        img = vae.decode(z).squeeze().detach().cpu().numpy()
        metrics = {
            'AF':    compute_area_fraction(img),
            'MI':    compute_morans_i(img),
            'BI':    compute_band_metrics(img)[0],
            'Angle': compute_band_metrics(img)[1],
            'Lambda':compute_thickness_lambda(img)
        }
        loss = loss_fn(metrics, target)
        best_z = z.clone(); best_loss = loss

        # MCMC loop
        for it in range(1, args.steps + 1):
            # propose
            z_prop = reflect(z + sigma * torch.randn_like(z))
            img_p  = vae.decode(z_prop).squeeze().detach().cpu().numpy()
            metrics_p = {
                'AF':    compute_area_fraction(img_p),
                'MI':    compute_morans_i(img_p),
                'BI':    compute_band_metrics(img_p)[0],
                'Angle': compute_band_metrics(img_p)[1],
                'Lambda':compute_thickness_lambda(img_p)
            }
            loss_p = loss_fn(metrics_p, target)

            # accept/reject
            if loss_p < loss or random.random() < math.exp(-args.beta * (loss_p - loss)):
                z, loss, metrics = z_prop, loss_p, metrics_p
                n_acc += 1; acc_block += 1
                if loss < best_loss:
                    best_loss, best_z = loss, z.clone()

            # adapt sigma
            if it % ADAPT_BLOCK == 0:
                rate = acc_block / ADAPT_BLOCK
                sigma *= math.exp(ADAPT_RATE * (rate - TARGET_ACC))
                sigma = max(SIG_MIN, min(SIG_MAX, sigma))
                acc_block = 0

            # record
            row = [it]
            if args.save_latents:
                row += z.detach().cpu().numpy().ravel().tolist()
            row += [
                loss,
                metrics['AF'], metrics['MI'], metrics['BI'], metrics['Angle'], metrics['Lambda'],
                sigma,
                n_acc/it
            ]
            writer.writerow(row)

            if it % 100 == 0:
                print(f"Chain{idx}  iter={it:<6}  loss={loss:.2e}  acc={n_acc/it:.2%}  sigma={sigma:.2e}")

    # save best reconstruction
    recon = vae.decode(best_z).squeeze().detach().cpu().numpy()
    from PIL import Image
    Image.fromarray((recon*255).astype('uint8')).save(chain_dir / 'best_reconstruction.png')

    # post‐hoc UMAP (if enabled)
    if args.enable_umap and args.save_latents:
        import pandas as pd
        df = pd.read_csv(csv_path)
        latent_cols = [f'z{i}' for i in range(args.latent_dim)]
        Z = df[latent_cols].values
        umap_model = joblib.load(args.umap_model_path)
        U = umap_model.transform(Z)
        df['U1'], df['U2'] = U[:,0], U[:,1]
        df.to_csv(csv_path, index=False)
        print(f"Appended UMAP coords to {csv_path}")

# ─── Main Launcher ─────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--vae_path', type=str,
        default=os.path.join(os.path.dirname(__file__), '..', 'data', 'VAE', 'vae_best.pth'),
        help='Path to trained VAE model'
    )
    parser.add_argument('--steps',      type=int, default=500, help='MCMC steps per chain')
    parser.add_argument('--sigma',      type=float, default=0.05, help='Initial proposal σ')
    parser.add_argument('--beta',       type=float, default=1.0,  help='Inverse temperature')
    parser.add_argument('--chains',     type=int, default=cpu_count(), help='Parallel chains')
    parser.add_argument(
        '--out_dir', type=Path, default=os.path.join(os.path.dirname(__file__), '..', 'user_data'),
        help='(Optional) Directory for MCMC outputs; if unspecified, use user_data/<data_subdir>'
    )
    parser.add_argument(
        '--data_subdir', type=Path, default='mcmc_out',
        help='Name of the subdirectory under user_data to write outputs into (used if --out_dir is not set)'
    )
    parser.add_argument('--save_latents',default=True, action='store_true', help='Save raw latent vectors')
    parser.add_argument('--latent_dim',  type=int, default=128, help='Latent dimension')
    parser.add_argument('--seed',        type=int, default=0, help='Random seed offset')
    parser.add_argument('--target_af',      type=float, help='Target AF')
    parser.add_argument('--target_mi',      type=float, help='Target MI')
    parser.add_argument('--target_bi',      type=float, help='Target BI')
    parser.add_argument('--target_ang',     type=float, help='Target orientation angle')
    parser.add_argument('--target_lambda',  type=float, help='Target local thickness rate')
    parser.add_argument(
        '--umap_model_path', type=str,
        default=os.path.join(os.path.dirname(__file__), '..', 'data', 'VAE', 'features_umap', 'latent_umap_model.pkl'),
        help='Path to pretrained UMAP model'
    )
    parser.add_argument(
        '--no_umap', action='store_false', dest='enable_umap',
        help='Disable post‐hoc UMAP computation'
    )
    parser.set_defaults(enable_umap=True)
    args = parser.parse_args()

    # determine output directory

    args.out_dir = Path(args.out_dir) / args.data_subdir
    args.out_dir.mkdir(parents=True, exist_ok=True)

    target = (
        args.target_af,
        args.target_mi,
        args.target_bi,
        args.target_ang,
        args.target_lambda
    )
    if not any(v is not None for v in target):
        parser.error('Specify at least one target metric via --target_*')

    processes = [Process(target=run_chain, args=(i, args, target)) for i in range(args.chains)]
    for p in processes: p.start()
    for p in processes: p.join()
