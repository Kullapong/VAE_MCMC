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
│── user_data/             # Output from running MCMC, user trained VAE and UMAP model. (This directory will appear after running some module.)
│ 
│── Installation.txt       # Setup guides (conda, pip)
│── requirements.txt       # List of Python dependencies
│── README.md              # User manual

# Output type
1. From Training (train.py)
  - vae_best.pth → trained VAE weights
  - training_log.csv → BCE, KLD, Total loss per epoch
  - recon_epoch_X.png → sample reconstructions at checkpoints
2. From Feature Extraction (compute_features.py)
  - metrics.csv → per-image metrics table
  - metrics_pairplot.png → feature pairplot
3. From Encoding + UMAP
  - latent.csv → latent vectors from VAE
  - latent_umap_model.pkl → trained UMAP reducer
  - metrics_umap.csv → latent UMAP + features merged
  - umap_AF.png, umap_MI.png, … → scatter plots colored by features
4. From MCMC Sampling
  - chain_i/chain.csv → latent samples, losses, metrics per iteration
  - chain_i/best_reconstruction.png → best sample image
  - mcmc_umap_metrics.csv → merged UMAP coords + metrics
  - mcmc_umap_scatter.png, mcmc_umap_kde.png, mcmc_best_point.png → visualizations
5. From Comprehensive Analysis (plot_analysis.py)
  - dataset_distributions.png → histograms of AF, MI, BI, ORI, λ
  - training_loss.png → BCE, KLD, Total loss curves
  - dataset_umap_AF.png → dataset UMAP colored by AF
  - mcmc_umap_* → scatter, KDE, and best-sample plots

# How to use.
1. Generate Synthetic Microstructures
After completing the installation, you can generate two-phase synthetic microstructures by specifying target features and running run_mcmc.py.
The following features are supported by default:
  - Area Fraction (AF)
  - Moran’s Index (MI)
  - Martensite Band Index (BI)
  - Martensite Band Orientation (Angle)
  - Local Thickness Parameter (λ)
If you only want to use this workflow with default settings (VAE model, UMAP model, dataset and mirostrcuture features), you don’t need to modify any code — simply run run_mcmc.py with your desired targets.
Example command: python run_mcmc.py --target_af 0.2 --steps 1000 --chains 2
2. Plot Results and Analysis
  - Use plot_analysis.py for a comprehensive set of plots (training loss, feature distributions, dataset UMAP, MCMC scatter/KDE, etc.).
  - Use plot_mcmc_umap.py if you only need MCMC result plots. (Each script has options you can configure inside the file.)
3. Extend with New Features
If you want to add new microstructure descriptors to the workflow, follow these steps:
  3.1 Define the feature in metrics.py
  - Write a new function (e.g., compute_new_feature(img)) that takes an image and returns a numeric value
  3.2 Integrate it into run_mcmc.py
  - Add the new feature into the metrics dictionary inside run_chain().
  - Extend the loss function (loss_fn) with a weight and the new feature.
  - (Optional) Add a command-line argument like --target_newfeature so users can set the target from the terminal.
  3.3 Update visualizations (optional)
  - If you want your new feature to appear in distribution plots or UMAP coloring, extend the relevant plotting functions in plot_analysis.py or plot_features_umap.py.
You do not need to change the VAE (vae.py) or training script (train.py) to add a new feature — those only handle encoding/decoding of images.
4. Using Custom Data or VAE Models
  - To use your own dataset and/or pretrained VAE model, specify the model path in the run_mcmc.py arguments.
  - To modify the VAE architecture, see vae.py.
  - To retrain the VAE with your dataset or change training parameters, see train.py.
  - Please extract the Image_dataset.zip before running. 
5. Re-training UMAP for Visualization
  - If you train a non-default VAE model, you should also retrain the UMAP projection to ensure accurate latent-space visualization.
  - See train_apply_umap.py for details.
