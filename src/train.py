#!/usr/bin/env python
import os
import sys
# OpenMP workaround on Windows
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import glob, csv, random, argparse, time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image

# Enable cuDNN auto-tuner
torch.backends.cudnn.benchmark = True

# Ensure `src/` on Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
from vae import VAE

# Loss functions
def compute_bce(recon_x, x):
    return nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
def compute_kld(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

class MicrostructureDataset(Dataset):
    """Loads grayscale images named numerically (1–3101)."""
    def __init__(self, pattern=None, transform=None):
        if pattern is None:
            base = os.path.join(script_dir, '..', 'data', 'Img')
            pattern = os.path.join(base, '*.png')
        paths = sorted(glob.glob(pattern))
        self.paths = [p for p in paths if os.path.splitext(os.path.basename(p))[0].isdigit()]
        self.transform = transform
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('L')
        return self.transform(img) if self.transform else img

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pattern', default=None, help='Input glob (default data/Img/*.png)')
    parser.add_argument('--out_dir', default=os.path.join(script_dir,'..','user_data','VAE'), help='Directory for outputs')
    parser.add_argument('--epochs', type=int, default=9500, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of DataLoader workers')
    args = parser.parse_args()

    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on device: {device}")
    # Check CPU availability
    available_cpus = os.cpu_count() or 1
    print(f"Detected {available_cpus} CPU cores; using {args.num_workers} DataLoader workers")
    os.makedirs(args.out_dir, exist_ok=True)

    start_time = time.time()
    log_file = os.path.join(args.out_dir, 'training_log.csv')
    with open(log_file, 'w', newline='') as cf:
        writer = csv.writer(cf)
        # Record training setup
        writer.writerow(['training_setup', f"pattern={args.pattern}", f"epochs={args.epochs}", f"batch_size={args.batch_size}"])
        # Write header for metrics
        writer.writerow(['epoch', 'bce', 'kld', 'total'])

        ds = MicrostructureDataset(args.pattern, transform=transforms.Compose([
            transforms.Resize((60,60)), transforms.ToTensor()
        ]))
        dl = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,  # parallel data loading
            pin_memory=True,
            persistent_workers=True
        )

        model = VAE(img_channels=1, img_size=60).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        best_loss = float('inf')

        for epoch in range(1, args.epochs+1):
            model.train()
            bce_sum = kld_sum = 0.0
            for batch in dl:
                batch = batch.to(device, non_blocking=True)
                optimizer.zero_grad()
                recon, mu, logvar = model(batch)
                bce = compute_bce(recon, batch)
                kld = compute_kld(mu, logvar)
                (bce + kld).backward()
                optimizer.step()
                bce_sum += bce.item()
                kld_sum += kld.item()

            n = len(ds)
            avg_bce = bce_sum / n
            avg_kld = kld_sum / n
            avg_total = avg_bce + avg_kld
            print(f"Epoch [{epoch}/{args.epochs}]  BCE: {avg_bce:.4f}, KLD: {avg_kld:.4f}, Total: {avg_total:.4f}")
            writer.writerow([epoch, f"{avg_bce:.6f}", f"{avg_kld:.6f}", f"{avg_total:.6f}"])

            if epoch % 10 == 0:
                idxs = random.sample(range(n), 49)
                batch = torch.stack([ds[i] for i in idxs]).to(device, non_blocking=True)
                with torch.no_grad():
                    recon_rand, _, _ = model(batch)
                vutils.save_image(recon_rand,
                                  os.path.join(args.out_dir, f"recon_epoch_{epoch}.png"),
                                  nrow=7)

            if avg_total < best_loss:
                best_loss = avg_total
                torch.save(model.state_dict(), os.path.join(args.out_dir, 'vae_best.pth'))

    total_time = time.time() - start_time
    with open(log_file, 'a', newline='') as cf:
        writer = csv.writer(cf)
        writer.writerow(['total_time', f"{total_time:.2f}"])


    print(f"Training complete in {total_time:.2f} seconds. Logs and model in {args.out_dir}")
