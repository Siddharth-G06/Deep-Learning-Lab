# Lab 7 — End-to-End Study of Autoencoders, Convolutional Autoencoders, Denoising Autoencoders and Variational Autoencoders

> **Shiv Nadar University Chennai | CS3807 — Deep Learning Laboratory | Academic Year 2026–27**

---

## Experiment Overview

This experiment develops an end-to-end understanding of autoencoders and their major variants for image representation, reconstruction, denoising, and generative modelling. Starting from a fully connected autoencoder, the study progressively introduces spatial structure (Convolutional Autoencoder), controlled image corruption (Denoising Autoencoder), and probabilistic latent modelling (Variational Autoencoder). The MNIST handwritten digit dataset is used as a common benchmark throughout all four studies.

---

## Objective

The objective is to:

- Understand the encoder–latent–decoder structure common to all autoencoder variants.
- Implement and compare a Fully Connected Autoencoder (FC-AE), a Convolutional Autoencoder (CAE), a Denoising Convolutional Autoencoder (DCAE), and a Variational Autoencoder (VAE).
- Evaluate reconstruction quality using MSE, MAE, and SSIM.
- Visualise and interpret the latent space learned by each model.
- Explore the generative capability of the VAE through sampling and latent-space interpolation.

---

## Learning Outcomes

After completing this experiment, students will be able to:

- Explain the encoder–latent–decoder structure of an autoencoder.
- Prepare image data for reconstruction-based learning.
- Implement a fully connected autoencoder.
- Implement a Convolutional Autoencoder for image reconstruction.
- Introduce controlled noise and build a denoising autoencoder.
- Evaluate reconstruction using MSE, MAE and SSIM.
- Visualise and interpret latent representations.
- Explain the probabilistic latent representation of a VAE.
- Implement the reparameterisation trick.
- Calculate and interpret reconstruction and KL-divergence losses.
- Generate new images by sampling from a learned latent space.
- Compare deterministic and variational latent representations.

---

## Dataset

### MNIST Handwritten Digit Dataset

| Property | Value |
|---|---|
| **Image Size** | 28 × 28 × 1 (grayscale) |
| **Classes** | 10 (digits 0–9) |
| **Training Samples used** | 10,000 |
| **Test Samples used** | 2,000 |
| **Source** | `tf.keras.datasets.mnist` |

Pixel values are normalised from `[0, 255]` to `[0, 1]`. The labels are **not** used during training; the original image itself serves as the reconstruction target.

---

## Overall Experimental Pipeline

```
MNIST Image → Normalisation → Encoder → Latent Representation → Decoder → Reconstructed Image → Evaluation
```

The experiment is divided into four related studies:

1. Fully Connected Autoencoder (FC-AE)
2. Convolutional Autoencoder (CAE)
3. Denoising Convolutional Autoencoder (DCAE)
4. Variational Autoencoder (VAE)

---

## Study 1 — Fully Connected Autoencoder

Each 28×28 image is flattened to a vector of length 784 and passed through a symmetric encoder–decoder network.

### Architecture

```
Input (784) → Dense(128) → Dense(32) → Latent(16) → Dense(32) → Dense(128) → Output (784)
```

Hidden layers use ReLU activation; the output layer uses sigmoid activation.

### Training Configuration

| Parameter | Value |
|---|---|
| Latent Dimension | 16 |
| Optimiser | Adam |
| Learning Rate | 1 × 10⁻³ |
| Batch Size | 128 |
| Epochs | 20 |
| Loss | Binary Cross-Entropy |

---

## Study 2 — Convolutional Autoencoder (CAE)

The CAE preserves spatial structure by applying convolutional operations in the encoder and transposed convolutions or upsampling in the decoder.

### Architecture

```
Input (28×28×1)
→ Conv2D(32, 3, relu, same) → MaxPooling2D(2)
→ Conv2D(64, 3, relu, same) → MaxPooling2D(2)
→ Conv2D(64, 3, relu, same) → UpSampling2D(2)
→ Conv2D(32, 3, relu, same) → UpSampling2D(2)
→ Conv2D(1, 3, sigmoid, same)
→ Output (28×28×1)
```

---

## Study 3 — Denoising Convolutional Autoencoder (DCAE)

The DCAE is trained to reconstruct a **clean** image from a **corrupted** input. The clean image always remains the reconstruction target.

```
Clean Image (x) → Add Noise → Noisy Image (x̃) → Encoder → Latent → Decoder → Denoised Output (x̂)
```

### Corruption Types

| Type | Parameters |
|---|---|
| Gaussian Noise | σ ∈ {0.1, 0.2, 0.3} |
| Salt-and-Pepper | p ∈ {0.05, 0.10, 0.20} |

---

## Study 4 — Variational Autoencoder (VAE)

The VAE learns a probability distribution over the latent space rather than a single deterministic vector.

```
q_φ(z|x) = N(μ(x), diag(σ²(x)))
z = μ + σ ⊙ ε,   ε ~ N(0, I)
```

### VAE Loss Function

```
L_VAE = L_rec + L_KL
```

where:
- **Reconstruction Loss** measures pixel-level fidelity (binary cross-entropy).
- **KL Divergence** regularises the latent space towards a standard normal prior:

```
D_KL = -½ Σ_j (1 + log σ_j² − μ_j² − σ_j²)
```

### Architecture (Latent Dimension = 2)

```
Input (28×28×1) → Conv Layers → Flatten → (μ, log σ²) → z ∈ R² → Dense → ConvTranspose → Output (28×28×1)
```

A two-dimensional latent space is used so that the learned representation can be directly visualised on a 2D scatter plot.

---

## Experimental Procedure

The full implementation is documented in `Lab_7.ipynb` and is divided into the following tasks:

1. **Dataset Loading and Preprocessing:** Loading MNIST, normalising pixel values to [0, 1], and splitting into training and test subsets.
2. **Fully Connected Autoencoder:** Building, training, and evaluating the FC-AE. Plotting original vs. reconstructed images (Plot 1) and the training/validation reconstruction loss curve (Plot 2).
3. **Convolutional Autoencoder:** Building and training the CAE. Comparing FC-AE and CAE reconstructions side-by-side (Plot 3) and computing MSE, MAE, SSIM and parameter counts for both.
4. **Denoising Autoencoder:** Applying Gaussian and salt-and-pepper noise, training the DCAE, and displaying clean vs. noisy vs. denoised triplets (Plot 4). Evaluating reconstruction quality across noise levels (Plot 5).
5. **VAE Training:** Building and training the VAE. Plotting the VAE training/validation reconstruction loss (Plot 9).
6. **Latent Space Visualisation:** Encoding all test images and plotting their latent coordinates coloured by digit label (Plot 6).
7. **Generative Sampling:** Sampling random latent vectors from N(0, I) and decoding to generate new images (Plot 7).
8. **Latent Space Interpolation:** Interpolating between two latent vectors and decoding each step to observe smooth morphing (Plot 8).
9. **Reconstruction Error Distribution:** Computing per-image MSE on the test set and plotting the histogram. Identifying and displaying the five highest-error samples (Plot 10).
10. **Consolidated Model Comparison:** Reporting MSE, MAE, SSIM, parameter count and training time for all four models.
11. **Latent Dimension Study:** Repeating the FC-AE with dz ∈ {2, 8, 16, 32} and plotting reconstruction MSE vs. latent dimension.

---

## Required Plots

| Plot | Description |
|---|---|
| Plot 1 | Original vs. FC-AE reconstructed images (grid of ≥ 10 images) |
| Plot 2 | FC-AE training & validation reconstruction loss curve |
| Plot 3 | FC-AE vs. CAE reconstruction comparison |
| Plot 4 | Clean vs. noisy vs. denoised images (grid of ≥ 10 images) |
| Plot 5 | Noise level vs. MSE / MAE / SSIM |
| Plot 6 | VAE 2D latent space scatter (coloured by digit label) |
| Plot 7 | Grid of randomly generated VAE images (≥ 25 samples) |
| Plot 8 | Latent space interpolation sequence |
| Plot 9 | VAE training & validation reconstruction loss |
| Plot 10 | Reconstruction error distribution histogram + high-error samples |

---

## Repository Contents

```
Lab 7 - End-to-End Study of Autoencoders, Convolutional Autoencoders, Denoising Autoencoders and Variational Autoencoders/
+-- README.md                       # This document
+-- Lab_7.ipynb                     # Jupyter Notebook with the full implementation
+-- Experiment_7.tex                # Lab manual (LaTeX source)
+-- lab7_plots/
    +-- plot1.png                   # Plot 1 - Original vs. reconstructed (FC-AE)
    +-- plot3.png                   # Plot 3 - FC-AE vs. CAE comparison
    +-- plot4.png                   # Plot 4 - Clean vs. noisy vs. denoised
    +-- plot6.png                   # Plot 6 - VAE latent space visualisation
    +-- plot7.png                   # Plot 7 - VAE generated samples
    +-- plot8.png                   # Plot 8 - Latent space interpolation
    +-- plot10_he.png               # Plot 10 - Highest reconstruction error samples
    +-- plot_1_original_vs_reconstructed.eps
    +-- plot_2_fc_training_validation_loss.eps
    +-- plot_3_original_fc_ae_vs_cae.eps
    +-- plot_4_clean_noisy_denoised.eps
    +-- plot_5_noise_level_vs_metrics.eps
    +-- plot_6_vae_latent_space.eps
    +-- plot_7_vae_generated_samples.eps
    +-- plot_8_vae_latent_interpolation.eps
    +-- plot_9_vae_training_validation_reconstruction_loss.eps
    +-- plot_10_reconstruction_error_distribution.eps
    +-- plot_10_highest_error_images.eps
```

---

## Reconstruction Metrics

| Metric | Description |
|---|---|
| **MSE** | Mean Squared Error — pixel-level squared deviation |
| **MAE** | Mean Absolute Error — pixel-level absolute deviation |
| **SSIM** | Structural Similarity Index — perceptual similarity measure |

SSIM is computed using [`skimage.metrics.structural_similarity`](https://scikit-image.org/docs/stable/api/skimage.metrics.html).

---

## Consolidated Model Results

| Model | MSE | MAE | SSIM | Parameters | Training Time |
|---|---|---|---|---|---|
| FC Autoencoder | — | — | — | — | — |
| Convolutional AE | — | — | — | — | — |
| Denoising CAE | — | — | — | — | — |
| VAE | — | — | — | — | — |

*(Values are to be filled in from the student's own execution of `Lab_7.ipynb`.)*

---

## Key Observations

- **FC-AE Limitations:** The fully connected autoencoder treats each pixel independently and loses spatial locality, leading to blurred reconstructions especially for digits with fine strokes.
- **CAE Advantage:** Convolutional layers share weights across spatial positions and preserve local structure, yielding sharper reconstructions with fewer parameters compared to a fully connected model of similar capacity.
- **Denoising Capability:** The denoising autoencoder successfully recovers major digit structure even under moderate corruption (σ = 0.2 or p = 0.10). Reconstruction quality degrades gradually as noise level increases, with SSIM being the most sensitive indicator.
- **VAE Latent Space:** The VAE latent space forms smooth, overlapping clusters for each digit class. The 2D scatter plot reveals that semantically similar digits (e.g., 3 and 8, 4 and 9) tend to occupy neighbouring regions.
- **Latent Space Interpolation:** Interpolating between two latent vectors produces a visually smooth morphing sequence, confirming that the VAE latent space is continuous and well-structured.
- **Generative Quality:** Randomly sampled VAE images generally resemble plausible handwritten digits, though some samples near decision boundaries appear ambiguous or blurred due to the trade-off imposed by the KL-divergence regulariser.
- **Latent Dimension Trade-off:** Smaller latent dimensions compress information more aggressively, increasing reconstruction error. Larger latent dimensions reduce error but can approach the capacity of an identity mapping with negligible compression.

---

## Dependencies

The following Python libraries are required to run the notebook:

- `tensorflow` (≥ 2.x)
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `scikit-image` *(for SSIM)*
- `jupyter`

Install all dependencies with:

```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn scikit-image jupyter
```

---

## Execution Instructions

1. Navigate to the experiment folder:
   ```bash
   cd "Lab 7 - End-to-End Study of Autoencoders, Convolutional Autoencoders, Denoising Autoencoders and Variational Autoencoders"
   ```
2. Install the required libraries:
   ```bash
   pip install tensorflow numpy pandas matplotlib seaborn scikit-learn scikit-image jupyter
   ```
3. Launch Jupyter Notebook:
   ```bash
   jupyter notebook Lab_7.ipynb
   ```
4. Execute the cells sequentially from top to bottom. Generated plots will be saved automatically to the `lab7_plots/` subdirectory.

> **Note:** Training the VAE with a 2D latent space is fast on a CPU. If you use a larger latent dimension or a deeper architecture, consider switching to a GPU-enabled environment for faster execution.

---

## References

1. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
2. Kingma, D. P., & Welling, M. (2014). Auto-Encoding Variational Bayes. *ICLR 2014*.
3. Vincent, P., et al. (2010). Stacked Denoising Autoencoders: Learning Useful Representations in a Deep Network with a Local Denoising Criterion. *Journal of Machine Learning Research*, 11, 3371–3408.
4. LeCun, Y., Cortes, C., & Burges, C. J. C. MNIST Handwritten Digit Database.
5. TensorFlow Documentation: https://www.tensorflow.org
6. Keras Documentation: https://keras.io
7. scikit-image Documentation: https://scikit-image.org
