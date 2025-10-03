import torch
import torch.nn as nn
from torch import Tensor

class VAE(nn.Module):
    def __init__(self,
                 latent_dim: int = 128,
                 img_channels: int = 1,
                 img_size: int = 60):
        super().__init__()
        self.latent_dim = latent_dim
        self.img_size   = img_size
        # Encoder
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(img_channels, 32, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, 1, 0),
            nn.ReLU(inplace=True),
        )
        self.fc1       = nn.Linear(256 * 4 * 4, 512)
        self.fc_mu     = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)
        # Decoder
        self.fc2 = nn.Linear(latent_dim, 256 * 4 * 4)
        self.decoder_deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, img_channels, 4, 2, 1),
            nn.Sigmoid(),
        )

    def encode(self, x: Tensor) -> (Tensor, Tensor):
        h = self.encoder_conv(x).view(x.size(0), -1)
        h = torch.relu(self.fc1(h))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: Tensor) -> Tensor:
        h = torch.relu(self.fc2(z)).view(-1, 256, 4, 4)
        recon = self.decoder_deconv(h)
        return recon[:, :, :self.img_size, :self.img_size]

    def forward(self, x: Tensor) -> (Tensor, Tensor, Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Loss functions:

def compute_bce(recon_x: Tensor, x: Tensor) -> Tensor:
    return nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')

def compute_kld(mu: Tensor, logvar: Tensor) -> Tensor:
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())