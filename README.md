# PWD: Prior-aware Wavelet Diffusion for Efficient Dental Limited-angle CT Reconstruction

Official PyTorch implementation of the paper:

> **PWD: Prior-aware Wavelet Diffusion for Efficient Dental Limited-angle CT Reconstruction**


## 🏗️ Project Structure
---
├── CT_rec_lib/ # CT reconstruction library (FP / BP operators)
├── guided_diffusion/ # Diffusion model implementation
├── limited_IMG_train.py # Training script
├── limited_IMG_sample.py # Sampling / inference script

### 📂 Module Description

- **CT_rec_lib**  
  Provides forward projection (FP) and back projection (BP) tools for CT reconstruction.

- **guided_diffusion**  
  Core implementation of the diffusion model, including network architecture, training, and sampling logic.

- **limited_IMG_train.py**  
  Main training entry point.

- **limited_IMG_sample.py**
  Used for reconstruction (sampling) from trained models.
  
-  **guided_diffusion/image_datasets.py**
  Used for reconstruction (sampling) from trained models.
  Prepare your CT projection or reconstruction dataset
  Modify dataset configuration in:
---
