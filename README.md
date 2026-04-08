# Photo-to-sketch (CycleGAN)

An implementation of Cycle-Consistent Generative Adversarial Networks (CycleGAN) for automated photo-to-sketch translation, trained on the Sketchy dataset.

## Overview
CycleGAN allows for learning image-to-image translation without paired examples. This project specifically focuses on translating realistic photos into sketches and vice-versa, maintaining structural consistency through cycle-consistency loss.

## Project Structure
- `cyclegan_model.py`: Definitions for the ResNet-based Generator and PatchGAN Discriminator.
- `data_utils.py`: Custom dataset loader for unpaired image domains.
- `train_cyclegan.py`: Trainer class encapsulating the training step and loss calculations.
- `train.py`: Main entry point for training with automated checkpointing and resume support.
- `CycleGAN_Sketch_to_Photo.ipynb`: Comprehensive notebook for experimentation and visualization.

## Architecture
- **Generator**: ResNet-based architecture with 9 residual blocks for high-capacity feature transformation.
- **Discriminator**: 70x70 PatchGAN for local texture and style validation.

## Training Stability
- **Identity Loss**: Ensures that the generator preserves the basic color/content when presented with target domain images.
- **Cycle-Consistency Loss**: Enforces that $G_{BA}(G_{AB}(A)) \approx A$, preventing mode collapse and ensuring structural mapping correctly.
- **Mixed Precision**: Uses `torch.cuda.amp` for faster training and lower memory footprint on NVIDIA GPUs.

## Usage
1. Download the Sketchy dataset.
2. Run the training script:
```python
python train.py
```
Checkpoints are saved automatically to manage session timeout on platforms like Kaggle.

## Acknowledgments
Based on the original [CycleGAN Paper](https://arxiv.org/abs/1703.10593).
