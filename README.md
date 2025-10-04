# VAE_MCMC
This repository presents a β-Variational Autoencoder (β-VAE) combined with Markov Chain Monte Carlo (MCMC) sampling for inverse microstructure design and reconstruction. The workflow provides a physics-aware, probabilistic way to generate synthetic microstructures that meet user-defined feature constraints.

# Project overview
The VAE_MCMC project integrates deep generative modeling and probabilistic sampling to study and reconstruct microstructures:
  - Variational Autoencoder (VAE): Learns a compact latent representation of microstructure images, enabling dimensionality reduction and generative modeling.
  - Markov Chain Monte Carlo (MCMC): Explores the latent space probabilistically, producing diverse samples beyond deterministic reconstruction.
  - Goal: Provide a framework for analyzing microstructural variability, generating synthetic microstructures, and studying uncertainty in microstructure-property relations.

Project structure
VAE_MCMC/
│── data/                  # Deafault data using in this framework includes Microstructure image dataset, pretrained VAE model and pretrained UMAP 
│   └── Img/               # Microstructure images for VAE training
│   └── VAE/               # Pretrained VAE and UMAP model
│
│── src/                      # Source code
│   │── vae.py                # VAE architecture (encoder, decoder, reparam trick)
│   │── train.py              # Train VAE model, log losses, save best weights
│   │── metrics.py            # Compute microstructure metrics (AF, MI, BI, ORI, λ, Hu)
│   │── compute_features.py   # Extract metrics from dataset → CSV + pairplots
│   │── encode_latent.py      # Encode images into latent vectors → CSV
│   │── train_apply_umap.py   # Train/apply UMAP to latent vectors + features
│   │── plot_features_umap.py # Scatter plots of UMAP colored by features
│   │── run_mcmc.py           # Run MCMC in latent space with VAE decoding
│   │── plot_mcmc_umap.py     # Visualize MCMC samples in UMAP space
│   │── plot_analysis.py      # Combined analysis: loss curves, distributions, UMAPs
│
│── Installation/          # Setup guides (conda, pip)
│── requirements.txt       # List of Python dependencies
│── README.md              # User manual


# How to use.
1. After installation completed you can start generate 2 phases synthetic microstructure by predefine features via run_mcmc.py with user input argument. 
  Here is the deafault features target the code have provided:
  -  Area fraction
  -  Moran's index
  -  Martensite band index
  -  Martensite band orientation
  -  Local thickness parameter
Exmple python prompt: python run_mcmc.py --target_af 0.2 --steps 1000 --chains 2 (Please see the detail in run_mcmc.py)
2. Plot the analyis using plot_analysis.py for user data,training,umap and MCMC plot,  or plot_mcmc_umap.py for only result plot (Noted that please see plot option in the files)
3. If you would like to work with other features please add the feature computation to metrics.py and as well as add it to run_mcmc.py loss function.
4. If you use your own data set and pretrained VAE model please assign the VAE path in run_mcmc.py prompt. If you need to edit VAE structure please see VAE.py, If you would like to train VAE with your own dataset and/or change the training parameter and method please see train.py
5. If the VAE model using not deafault model that we have provided please train UMAP model again for accuracy visualization please see train_apply_umap.py.   
