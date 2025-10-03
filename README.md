# VAE_MCMC
This repository presents a β-Variational Autoencoder (β-VAE) combined with Markov Chain Monte Carlo (MCMC) sampling for inverse microstructure design and reconstruction. The workflow provides a physics-aware, probabilistic way to generate synthetic microstructures that meet user-defined feature constraints.

# How to use.
1. After installation completed you can start generate 2 phases synthetic microstructure by predefine features via run_mcmc.py with user input argument. 
  Here is the deafault features target the code have provided:
  -  Area fraction
  -  Moran's index
  -  Martensite band index
  -  Martensite band orientation
  -  Local thickness parameter
Exmple python prompt: python run_mcmc.py --target_af 0.2 --steps 1000 --chains 2 (Please see the detail in run_mcmc.py)
2. Plot the analyis using plot_analysis.py or plot_mcmc_umap.py
2. If you would like to work with other features please add the feature computation to metrics.py and as well as add it to run_mcmc.py loss function.
3. If you use your own data set and pretrained VAE model please assign the VAE path in prompth. If you need to edit VAE structure please see VAE.py, If you would like to train VAE with your own dataset and/or change the training parameter and method please see train.py
4. If the VAE model using not deafault model we have provided please train UMAP model again for accuracy visualization please see train_apply_umap.py.   
