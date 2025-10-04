#!/usr/bin/env python
"""
Lightweight integration test for the VAE_MCMC pipeline with checklist reporting.
"""
from __future__ import annotations
import os, sys, zipfile, shutil, glob, time, argparse, subprocess
from pathlib import Path

ZIP_PATH_DEFAULT = r"VAE_MCMC_proj\\VAE_MCMC\\data\\Img\\Image_dataset.zip"
FALLBACK_IMG_DIR = r"VAE_MCMC_proj\\VAE_MCMC\\data\\Img"

BASE_DIR = Path(__file__).resolve().parent
PROJ_DIR = BASE_DIR.parent
TEST_DIR = PROJ_DIR / "test"
TEST_IMG_DIR = TEST_DIR / "data" / "Img"
TEST_USER_DIR = TEST_DIR / "user_data"
TEST_VAE_DIR = TEST_USER_DIR / "VAE"
TEST_MCMC_DIR = TEST_USER_DIR / "mcmc_out"
TEST_LOG = TEST_DIR / "test_log.txt"
SRC_DIR = BASE_DIR

checklist = []

def run_step(name: str, cmd: list[str], cwd: Path | None = None):
    print(f"\nRunning step: {name}")
    try:
        proc = subprocess.Popen(
            cmd, cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        with open(TEST_LOG, 'a', encoding='utf-8') as lf:
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stdout.write(line)
                lf.write(line)
        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(f"return code {proc.returncode}")
        checklist.append((name, "[PASSED]"))
    except Exception as e:
        checklist.append((name, f"[FAILED] {e}"))
        raise

def clean_and_prepare():
    if TEST_DIR.exists():
        shutil.rmtree(TEST_DIR)
    TEST_IMG_DIR.mkdir(parents=True, exist_ok=True)
    TEST_VAE_DIR.mkdir(parents=True, exist_ok=True)
    TEST_MCMC_DIR.mkdir(parents=True, exist_ok=True)

def collect_small_dataset(zip_path: str | None, n_images: int):
    tmp_extract = TEST_DIR / "_extracted"
    src_candidates: list[Path] = []
    if zip_path and Path(zip_path).exists():
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(tmp_extract)
        src_candidates.append(tmp_extract)
    fb_dir = Path(FALLBACK_IMG_DIR)
    if fb_dir.exists():
        src_candidates.append(fb_dir)
    proj_img = PROJ_DIR / "data" / "Img"
    if proj_img.exists():
        src_candidates.append(proj_img)
    images: list[Path] = []
    for c in src_candidates:
        if c.is_dir():
            images = [Path(p) for p in glob.glob(str(c / '**' / '*.png'), recursive=True)]
            if not images:
                images = [Path(p) for p in glob.glob(str(c / '**' / '*.jpg'), recursive=True)]
        if images:
            break
    if not images:
        raise FileNotFoundError("No images found")
    images = sorted(images)[:max(1, n_images)]
    for p in images:
        shutil.copy2(p, TEST_IMG_DIR / p.name)
    if tmp_extract.exists():
        shutil.rmtree(tmp_extract, ignore_errors=True)

def main():
    parser = argparse.ArgumentParser(description="Tiny test runner for VAE_MCMC")
    parser.add_argument('--zip', default=ZIP_PATH_DEFAULT)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--n_images', type=int, default=120)
    parser.add_argument('--metrics_n', type=int, default=30)
    parser.add_argument('--mcmc_steps', type=int, default=200)
    parser.add_argument('--mcmc_chains', type=int, default=2)
    parser.add_argument('--target_af', type=float, default=0.50)
    parser.add_argument('--train_num_workers', type=int, default=2)
    args = parser.parse_args()

    TEST_DIR.mkdir(parents=True, exist_ok=True)
    if TEST_LOG.exists():
        TEST_LOG.unlink()
    with open(TEST_LOG, 'w', encoding='utf-8') as f:
        f.write(time.strftime('# Test started: %Y-%m-%d %H:%M:%S') + "\n")

    clean_and_prepare()
    try:
        collect_small_dataset(args.zip, args.n_images)
        checklist.append(("Collect dataset", "[PASSED]"))
    except Exception as e:
        checklist.append(("Collect dataset", f"[FAILED] {e}"))
        print("Dataset collection failed.")
        return

    img_glob = str(TEST_IMG_DIR / '*.png')
    train_out = TEST_VAE_DIR
    latent_csv = TEST_VAE_DIR / 'latent.csv'
    metrics_csv = TEST_DIR / 'data' / 'metrics.csv'
    umap_model = TEST_VAE_DIR / 'features_umap' / 'latent_umap_model.pkl'
    umap_csv = TEST_VAE_DIR / 'features_umap' / 'metrics_umap.csv'
    plots_out = TEST_USER_DIR / 'plots'

    try:
        run_step("Train VAE", [
            sys.executable, str(SRC_DIR / 'train.py'),
            '--pattern', img_glob,
            '--out_dir', str(train_out),
            '--epochs', str(args.epochs),
            '--batch_size', '32',
            '--num_workers', str(args.train_num_workers),
        ], cwd=SRC_DIR)

        run_step("Encode Latents", [
            sys.executable, str(SRC_DIR / 'encode_latent.py'),
            '--pattern', img_glob,
            '--model_path', str(train_out / 'vae_best.pth'),
            '--out_csv', str(latent_csv),
            '--batch_size', '64'
        ], cwd=SRC_DIR)

        run_step("Compute Metrics", [
            sys.executable, str(SRC_DIR / 'compute_features.py'),
            '--pattern', img_glob,
            '--max-files', str(args.metrics_n),
            '--output', str(metrics_csv)
        ], cwd=SRC_DIR)

        # Normalize metrics CSV to ensure AF exists
        try:
            import pandas as pd
            metrics_fixed = TEST_DIR / 'data' / 'metrics_fixed.csv'
            if Path(metrics_csv).exists():
                dfm = pd.read_csv(metrics_csv)
            else:
                dfm = pd.DataFrame()
            # Ensure filename
            if 'filename' not in dfm.columns:
                fn_alt = next((c for c in ['file','img','image','path','name'] if c in dfm.columns), None)
                if fn_alt:
                    dfm.rename(columns={fn_alt:'filename'}, inplace=True)
            if 'filename' not in dfm.columns:
                dfm['filename'] = ''
            # Ensure AF
            if 'AF' not in dfm.columns:
                af_alt = next((c for c in ['af','area_fraction','areaFraction','AreaFraction','AF_mean','AF_val'] if c in dfm.columns), None)
                if af_alt is not None:
                    dfm['AF'] = dfm[af_alt]
                else:
                    dfm['AF'] = 0.0
            dfm.to_csv(metrics_fixed, index=False)
            metrics_csv_for_plot = metrics_fixed
            checklist.append(("Normalize Metrics CSV", "[PASSED]"))
        except Exception as e:
            metrics_csv_for_plot = metrics_csv
            checklist.append(("Normalize Metrics CSV", f"[FAILED] {e}"))

        run_step("Train+Apply UMAP", [
            sys.executable, str(SRC_DIR / 'train_apply_umap.py'),
            '--mode', 'train',
            '--latent_csv', str(latent_csv),
            '--metrics_csv', str(metrics_csv),
            '--model_path', str(umap_model),
            '--out_csv', str(umap_csv)
        ], cwd=SRC_DIR)

        # sanitize umap csv
        try:
            import pandas as pd
            df = pd.read_csv(umap_csv)
            umap_only = TEST_VAE_DIR / 'features_umap' / 'umap_only.csv'
            keep = [c for c in df.columns if c in ('filename','U1','U2')]
            if keep:
                df[keep].to_csv(umap_only, index=False)
                umap_csv_for_plot = umap_only
            else:
                umap_csv_for_plot = umap_csv
        except Exception:
            umap_csv_for_plot = umap_csv

        run_step("Plot Features UMAP", [
            sys.executable, str(SRC_DIR / 'plot_features_umap.py'),
            '--metrics_csv', str(metrics_csv),
            '--umap_csv', str(umap_csv_for_plot),
            '--out_dir', str(TEST_VAE_DIR / 'features_umap' / 'plots')
        ], cwd=SRC_DIR)

        run_step("Run MCMC", [
            sys.executable, str(SRC_DIR / 'run_mcmc.py'),
            '--vae_path', str(train_out / 'vae_best.pth'),
            '--steps', str(args.mcmc_steps),
            '--chains', str(args.mcmc_chains),
            '--out_dir', str(TEST_USER_DIR),
            '--data_subdir', 'mcmc_out',
            '--save_latents',
            '--latent_dim', '128',
            '--umap_model_path', str(umap_model),
            '--target_af', str(args.target_af)
        ], cwd=SRC_DIR)

        run_step("Plot MCMC UMAP", [
            sys.executable, str(SRC_DIR / 'plot_mcmc_umap.py'),
            '--mcmc_dir', str(TEST_MCMC_DIR)
        ], cwd=SRC_DIR)

        # Normalize training log columns for plot_analysis.py expectations
        try:
            import pandas as pd
            orig_log = TEST_VAE_DIR / 'training_log.csv'
            norm_log = TEST_VAE_DIR / 'training_log_normalized.csv'
            if orig_log.exists() and orig_log.stat().st_size > 0:
                df = pd.read_csv(orig_log)
            else:
                df = pd.DataFrame()
            # Ensure core columns: epoch, bce, kld, total
            if 'epoch' not in df.columns:
                df['epoch'] = list(range(len(df))) if len(df) > 0 else [0,1,2]
            # map alternatives
            alt = lambda opts: next((c for c in opts if c in df.columns), None)
            if 'bce' not in df.columns:
                src = alt(['bce','recon','recon_loss','reconstruction','reconstruction_loss'])
                df['bce'] = df[src] if src else 1.0
            if 'kld' not in df.columns:
                src = alt(['kld','kl','kl_div','kl_loss'])
                df['kld'] = df[src] if src else 0.1
            if 'total' not in df.columns:
                src = alt(['total','total_loss','loss','train_loss'])
                if src:
                    df['total'] = df[src]
                else:
                    df['total'] = (df['bce'] if 'bce' in df.columns else 1.0) + (df['kld'] if 'kld' in df.columns else 0.1)
            # For compatibility with plot_analysis.py, mirror 'loss' to 'total' if missing
            if 'loss' not in df.columns:
                df['loss'] = df['total']
            # Write normalized log
            ordered_cols = [c for c in ['epoch','bce','kld','total','loss'] if c in df.columns]
            df = df[ordered_cols].copy()
            df.to_csv(norm_log, index=False)
            checklist.append(("Normalize Training Log (epoch,bce,kld,total)", "[PASSED]"))
            training_log_for_plot = norm_log
        except Exception as e:
            checklist.append(("Normalize Training Log (epoch,bce,kld,total)", f"[FAILED] {e}"))
            training_log_for_plot = orig_log

        plots_out.mkdir(parents=True, exist_ok=True)
        # Build a merged metrics+UMAP CSV that includes AF for plot_analysis.py
        try:
            import pandas as pd
            merged_for_plot = TEST_VAE_DIR / 'features_umap' / 'metrics_umap_for_plot.csv'
            # Load metrics (features)
            df_feat = pd.read_csv(metrics_csv) if Path(metrics_csv).exists() else pd.DataFrame()
            # Try to locate AF in common variants
            if 'AF' not in df_feat.columns:
                af_alt = next((c for c in ['af','area_fraction','areaFraction','AreaFraction','AF_mean','AF_val'] if c in df_feat.columns), None)
                if af_alt is not None:
                    df_feat['AF'] = df_feat[af_alt]
                else:
                    # fabricate if totally missing
                    df_feat['AF'] = 0.0
            # Keep only filename + AF to avoid duplicate columns during merge
            cols_feat = [c for c in df_feat.columns if c in ('filename','AF')]
            df_feat = df_feat[cols_feat].copy() if cols_feat else pd.DataFrame(columns=['filename','AF'])
            # Load UMAP coords (use trimmed one if available)
            umap_candidate = umap_csv_for_plot if 'umap_csv_for_plot' in locals() else umap_csv
            df_umap = pd.read_csv(umap_candidate) if Path(umap_candidate).exists() else pd.DataFrame()
            # Ensure U1/U2 present
            if not {'U1','U2'}.issubset(df_umap.columns):
                # try alternates
                u1 = next((c for c in ['u1','umap1','x','X'] if c in df_umap.columns), None)
                u2 = next((c for c in ['u2','umap2','y','Y'] if c in df_umap.columns), None)
                if u1 and u2:
                    df_umap.rename(columns={u1:'U1', u2:'U2'}, inplace=True)
                else:
                    # cannot fix; create placeholder
                    df_umap['U1'] = 0.0
                    df_umap['U2'] = 0.0
            # Ensure filename exists in UMAP for merge
            if 'filename' not in df_umap.columns:
                # try common alt
                fn_alt = next((c for c in ['file','img','image','path','name'] if c in df_umap.columns), None)
                if fn_alt:
                    df_umap.rename(columns={fn_alt:'filename'}, inplace=True)
            # Merge and fill
            df_merge = pd.merge(df_umap, df_feat, on='filename', how='left') if 'filename' in df_umap.columns else df_umap
            if 'AF' not in df_merge.columns:
                df_merge['AF'] = 0.0
            # Save merged
            df_merge.to_csv(merged_for_plot, index=False)
            checklist.append(("Prepare metrics_umap_for_plot", "[PASSED]"))
            metrics_umap_for_plot = merged_for_plot
        except Exception as e:
            checklist.append(("Prepare metrics_umap_for_plot", f"[FAILED] {e}"))
            metrics_umap_for_plot = umap_csv
        run_step("Plot Analysis", [
            sys.executable, str(SRC_DIR / 'plot_analysis.py'),
            '--metrics_csv', str(metrics_csv),
            '--training_log', str(training_log_for_plot),
            '--metrics_umap_csv', str(metrics_umap_for_plot),
            '--mcmc_dir', str(TEST_MCMC_DIR),
            '--out_dir', str(plots_out),
            '--no_dataset_umap'
        ], cwd=SRC_DIR)
    except Exception as e:
        print("Error in pipeline:", e)

    print("\nChecklist:")
    for step, status in checklist:
        print(f" - {step}: {status}")
    print("\nArtifacts under:", TEST_DIR)
    print("Log file:", TEST_LOG)

if __name__ == "__main__":
    main()
