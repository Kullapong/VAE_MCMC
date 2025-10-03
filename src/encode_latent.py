#!/usr/bin/env python
import os
import sys
# OpenMP workaround on Windows
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import glob
import csv
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# Ensure src/ on path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
from vae import VAE

class ImageDataset(Dataset):
    """Dataset for loading grayscale images and preserving filenames."""
    def __init__(self, pattern, transform=None):
        self.paths = sorted(glob.glob(pattern))
        self.transform = transform

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert('L')
        if self.transform:
            img_tensor = self.transform(img)
        else:
            img_tensor = transforms.ToTensor()(img)
        return os.path.basename(path), img_tensor

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pattern', default=None, help='Glob for input images')
    parser.add_argument('--model_path', default=os.path.join(script_dir, '..', 'user_data', 'VAE', 'vae_best.pth'), help='Path to trained VAE (.pth)')
    parser.add_argument('--out_csv', default=os.path.join(script_dir,'..','user_data','VAE','latent.csv'),
                        help='Output CSV to save latent means')
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Encoding on device: {device}")

    # Prepare dataset
    if args.pattern is None:
        base = os.path.join(script_dir, '..', 'data', 'Img')
        args.pattern = os.path.join(base, '*.png')
    transform = transforms.Compose([transforms.Resize((60,60)), transforms.ToTensor()])
    ds = ImageDataset(args.pattern, transform)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    # Load model
    vae = VAE(latent_dim=128, img_channels=1, img_size=60).to(device)
    vae.load_state_dict(torch.load(args.model_path, map_location=device))
    vae.eval()

    # Encode and collect
    records = []
    with torch.no_grad():
        for fnames, imgs in dl:
            imgs = imgs.to(device)
            mu, logvar = vae.encode(imgs)
            mu = mu.cpu().numpy()
            for name, vec in zip(fnames, mu):
                records.append([name] + vec.tolist())

    # Save to CSV
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    header = ['filename'] + [f'z{i}' for i in range(records[0][1:].__len__())]
    with open(args.out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(records)

    print(f"Saved latent space to {args.out_csv}")