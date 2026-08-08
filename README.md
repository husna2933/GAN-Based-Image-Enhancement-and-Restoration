# GAN-Based Image Enhancement and Restoration

This repository contains a deep learning project for **image enhancement and restoration using a Generative Adversarial Network (GAN)**.

The system learns to reconstruct cleaner images from artificially degraded inputs. Training data is created by applying several types of degradation to clean images, including noise, blur, JPEG compression, geometric distortion, and erosion artifacts. A residual CNN generator then learns to restore these degraded images, while a discriminator encourages the generated outputs to appear similar to clean images.

A pretrained **VGG19** network is also used to provide feature-based supervision during training.

---

## Project Overview

The project is organized into three Jupyter notebooks:

```text
.
├── Image_data_generation.ipynb
├── SRDCGAN.ipynb
├── OUTPUT_GAN.ipynb
└── README.md
```

Each notebook represents a different stage of the pipeline:

| Notebook | Purpose |
|---|---|
| `Image_data_generation.ipynb` | Creates paired clean and artificially degraded images |
| `SRDCGAN.ipynb` | Builds and trains the GAN-based restoration model |
| `OUTPUT_GAN.ipynb` | Loads the trained generator and evaluates generated outputs |

---

## Pipeline

```text
Clean Images
     │
     ▼
Artificial Degradation
     │
     ├── Gaussian Noise
     ├── Gaussian Blur
     ├── JPEG Compression
     ├── Geometric Distortion
     └── Erosion Artifacts
     │
     ▼
Degraded Images
     │
     ▼
Residual CNN Generator
     │
     ▼
Restored Images
     │
     ├──────────────► Discriminator
     │                  │
     │               Real / Fake
     │
     └──────────────► VGG19
                        │
                 Feature Similarity
                        │
                        ▼
                  Generator Update
```

---

## 1. Data Generation

The `Image_data_generation.ipynb` notebook creates the paired data used for training.

For every clean input image, the notebook creates degraded versions using five transformations:

### Gaussian Noise

Random Gaussian noise is added to the original image.

### Gaussian Blur

Images are blurred using randomly selected Gaussian kernel sizes.

### JPEG Compression

JPEG compression is applied with varying quality levels to simulate compression-related degradation.

### Geometric Distortion

A sinusoidal transformation is applied to distort the spatial structure of the image.

### Erosion Artifacts

Morphological erosion is used to introduce additional image degradation.

The clean image is saved as the target image, while the corresponding modified image is saved as the degraded input.

Conceptually:

```text
Original Image ─────────────► Clean / Target Image
       │
       ▼
Image Degradation
       │
       ▼
Degraded / Input Image
```

The notebook also contains a function for reducing image dimensions to 25% of their original size for potential 4× super-resolution experiments. However, this downsampling operation is not used in the active data-generation pipeline.

---

## 2. GAN Architecture

The main model is implemented in `SRDCGAN.ipynb`.

It consists of three major components:

- Residual CNN Generator
- CNN Discriminator
- Pretrained VGG19 Feature Extractor

Both the degraded input images and clean target images are resized to:

```text
128 × 128 × 3
```

before training.

---

## Generator

The generator receives a degraded image and attempts to reconstruct a cleaner version.

Its architecture contains:

- Initial `Conv2D` layer
- ReLU activation
- **16 residual blocks**
- Batch Normalization
- Skip connections
- Two convolution-based reconstruction blocks
- Final `Conv2D` layer with `tanh` activation

A simplified view is:

```text
Degraded Image
      │
      ▼
   Conv2D
      │
      ▼
16 Residual Blocks
      │
      ▼
Skip Connection
      │
      ▼
Convolution Blocks
      │
      ▼
Final Conv2D
      │
      ▼
Restored Image
```

Residual connections help the network learn image corrections while retaining useful information from the original input.

---

## Discriminator

The discriminator learns to distinguish between:

```text
Clean Target Image  → Real
Generated Image     → Fake
```

It uses a sequence of:

- `Conv2D`
- LeakyReLU
- Dropout
- Batch Normalization
- Flatten
- Dense output layer with sigmoid activation

The discriminator provides adversarial feedback that encourages the generator to create more realistic outputs.

---

## VGG19 Feature Extractor

A pretrained **VGG19** network with ImageNet weights is used as a fixed feature extractor.

During training:

```text
Clean Image ─────────► VGG19 ─────► Target Features

Generated Image ─────► VGG19 ─────► Generated Features
                                      │
                                      ▼
                                Feature MSE
```

The generator is therefore encouraged not only to fool the discriminator, but also to produce images whose deep feature representation resembles that of the clean target.

---

## Training Objective

The combined generator model uses two losses:

1. **Adversarial loss** — encourages generated images to be classified as real.
2. **Feature reconstruction loss** — minimizes the mean squared error between VGG19 features of generated and clean images.

The implementation uses:

```python
loss = ['binary_crossentropy', 'mse']
loss_weights = [1e-3, 1]
```

The overall objective can be represented approximately as:

```text
Generator Loss =
VGG Feature Reconstruction Loss
+ 0.001 × Adversarial Loss
```

The discriminator itself is trained using mean squared error.

---

## Training Configuration

The main training notebook uses:

| Parameter | Value |
|---|---:|
| Input size | 128 × 128 × 3 |
| Residual blocks | 16 |
| Generator filters | 256 |
| Discriminator filters | 128 |
| Optimizer | Adam |
| Learning rate | 0.0002 |
| Adam β1 | 0.5 |
| Batch size | 1 |
| Training iterations | 2001 |
| Sample interval | 10 |

During each training iteration:

1. A clean/degraded image pair is loaded.
2. The generator reconstructs the degraded image.
3. The discriminator is trained on clean and generated images.
4. VGG19 features are extracted from the clean target.
5. The generator is updated using adversarial and feature reconstruction losses.
6. Generated samples are periodically saved for visual comparison.

---

## Model Evaluation

The project evaluates reconstruction quality both visually and quantitatively.

### Visual Comparison

Generated images are displayed alongside their corresponding clean target images:

```text
Generated | Original
Generated | Original
```

This makes it possible to visually inspect how closely the model reconstructs the original image.

### Peak Signal-to-Noise Ratio (PSNR)

PSNR is calculated from the mean squared error between the original and generated images.

The implementation follows:

```text
PSNR = 20 × log10(255 / √MSE)
```

Higher PSNR generally indicates that the generated image is closer to the reference image.

### Mean Squared Error (MSE)

MSE is also calculated directly between the generated and original images.

Lower MSE indicates smaller pixel-level differences.

---

## Output / Inference Notebook

`OUTPUT_GAN.ipynb` is used to test the trained model.

The notebook:

1. Loads a previously trained Keras model.
2. Reconstructs the generator architecture.
3. Transfers the trained model weights.
4. Loads degraded test images.
5. Generates restored images.
6. Displays the generated outputs.
7. Compares generated images with their original clean counterparts.
8. Calculates PSNR and MSE for evaluation.

The notebook contains saved outputs, so the example generated images can be viewed directly on GitHub without retraining the model.

---

## Technologies Used

- Python
- TensorFlow
- Keras
- OpenCV
- NumPy
- SciPy
- Matplotlib
- VGG19
- Google Colab

---

## Key Features

- Automatic generation of degraded training images
- Multiple degradation types
- Residual CNN generator
- GAN-based adversarial training
- VGG19 feature-based reconstruction loss
- Clean-vs-generated visual comparisons
- PSNR evaluation
- MSE evaluation
- Separate notebooks for preprocessing, training, and inference

---

## Authors

This project was developed collaboratively as an academic machine learning project.

---

## Repository Contents

### `Image_data_generation.ipynb`

Creates paired clean and degraded images using noise, blur, compression, distortion, and erosion.

### `SRDCGAN.ipynb`

Defines and trains the generator, discriminator, VGG19 feature extractor, adversarial model, and PSNR monitoring workflow.

### `OUTPUT_GAN.ipynb`

Loads the trained model and demonstrates image reconstruction and quantitative evaluation using PSNR and MSE.

---

## Disclaimer

This repository is an academic implementation intended for learning, experimentation, and research. The current architecture should be described primarily as a **GAN-based image enhancement and restoration model** rather than a complete production-ready super-resolution system.
