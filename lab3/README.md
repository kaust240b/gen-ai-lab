This project implements a Variational Autoencoder (VAE) using PyTorch to learn compact latent representations of handwritten digits from the MNIST dataset. The notebook covers the complete workflow required to train, evaluate, and use a generative VAE model, including model definition, loss formulation, training, checkpointing, and image generation.

The VAE architecture consists of an encoder network that maps input images to a latent Gaussian distribution (mean and log-variance), followed by the reparameterization trick to sample a latent vector. A decoder network then reconstructs the original input image from this latent representation. Training is performed using a combined objective of reconstruction loss (Binary Cross Entropy) and KL divergence regularization, encouraging the latent space to remain smooth and well-structured for generative sampling.

The training pipeline includes:

Loading and preprocessing MNIST images using PyTorch DataLoader

Training the VAE for multiple epochs

Saving model checkpoints at regular intervals

Logging loss values during training

Saving reconstruction results (input vs reconstructed images) for qualitative evaluation

In addition to reconstruction, the project demonstrates the generative ability of VAEs by sampling random latent vectors from a standard normal distribution and decoding them to produce new digit images. The notebook also supports resuming training from saved checkpoints, allowing long training runs to be continued without restarting and enabling further fine-tuning for improved output quality.


