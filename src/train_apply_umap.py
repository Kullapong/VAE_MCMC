#!/usr/bin/env python
import os
import argparse
import pandas as pd
import joblib
import umap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mode', choices=['train','apply'], required=True,
        help='Train a new UMAP or apply existing model'
    )
    parser.add_argument(
        '--latent_csv',
        default=os.path.join(os.path.dirname(__file__), '..', 'user_data', 'VAE', 'latent.csv'),
        help='Path to latent CSV (filename + z0..zN)'
    )
    parser.add_argument(
        '--metrics_csv',
        default=os.path.join(os.path.dirname(__file__), '..', 'data', 'metrics.csv'),
        help='Path to image features CSV (filename + AF, MI, BI, ORI, lambda)'
    )
    parser.add_argument(
        '--model_path',
        default=os.path.join(os.path.dirname(__file__), '..', 'user_data', 'VAE', 'features_umap', 'latent_umap_model.pkl'),
        help='Path to save (train) or load (apply) UMAP model (.pkl)'
    )
    parser.add_argument(
        '--out_csv',
        default=os.path.join(os.path.dirname(__file__), '..', 'user_data', 'VAE', 'features_umap', 'metrics_umap.csv'),
        help='Output CSV with filename, U1, U2, and image features'
    )
    parser.add_argument('--n_neighbors', type=int, default=120, help='UMAP n_neighbors')
    parser.add_argument('--min_dist', type=float, default=0.4, help='UMAP min_dist')
    parser.add_argument('--metric', default='cosine', help='UMAP metric')
    parser.add_argument('--random_state', type=int, default=0, help='Random seed')
    args = parser.parse_args()

        # Load latent and feature data
    df_latent = pd.read_csv(args.latent_csv)
    df_feat   = pd.read_csv(args.metrics_csv)
    # Merge on filename to attach features
    df = pd.merge(
        df_latent,
        df_feat[['filename', 'AF', 'MI', 'BI', 'ORI', 'lambda']],
        on='filename', how='left'
    )

    # Extract latent vectors for UMAP
    Z = df_latent.drop(columns=['filename']).values

    if args.mode == 'train':
        reducer = umap.UMAP(
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
            metric=args.metric,
            random_state=args.random_state
        )
        reducer.fit(Z)
        os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
        joblib.dump(reducer, args.model_path)
        print(f"Trained UMAP model saved to {args.model_path}")
    else:
        reducer = joblib.load(args.model_path)
        print(f"Loaded UMAP model from {args.model_path}")

    # Compute embedding and merge features
    emb = reducer.transform(Z)
    df['U1'], df['U2'] = emb[:, 0], emb[:, 1]

    # Save combined output
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"Saved UMAP+features to {args.out_csv}")

if __name__ == '__main__':
    main()