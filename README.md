# Unet4Retina: Segmentation of Retinal Blood Vessels using U-Net

This repository presents a U-Net model specifically adapted to the challenging task of retinal vessel segmentation, demonstrating robustness and improved segmentation of thin and branching structures using the STARE dataset.


## Overview

Retinal blood vessel segmentation is crucial for diagnosing ophthalmological conditions. This project leverages the well-established **U-Net architecture**, enhancing its performance by introducing a specialized **Skeleton Recall Loss**, ensuring fine vascular details are preserved in segmentation tasks.


## Motivation

Traditional binary cross-entropy (BCE) loss functions often inadequately segment fine, filament-like structures, crucial in medical imagery such as retinal scans. This work overcomes this limitation by combining a **Skeleton Recall Loss** with a weighted BCE loss, achieving superior segmentation performance, particularly relevant in clinical diagnostics.


## U-Net Architecture

The implemented model follows the classical U-Net structure:

- **Encoder (Contracting Path)**:
  - Extracts hierarchical spatial features through convolutional layers and max pooling.
  
- **Decoder (Expanding Path)**:
  - Recovers image details via upsampling and convolution, utilizing skip connections for precise reconstruction of fine features.

- **Output Layer**:
  - Per-pixel classification ensuring accurate segmentation.

<img src="https://raw.githubusercontent.com/MariusDragic/Unet4Retina/main/images/unet.png" alt="U-Net Architecture" width="75%"/>

*Figure 1 — U-Net architecture scheme*

## Loss Function: Skeleton Recall + Weighted BCE

The custom loss function integrates:

- **Skeleton Recall Loss**: prioritizes the detection of thin structures by comparing predictions against a skeletonized mask.
- **Weighted BCE Loss**: manages class imbalance, crucial for detailed and accurate segmentation.

Mathematically, the loss function is expressed as:

```math
\mathcal{L}_{\text{total}} = 1 - \frac{1}{|C|}\sum_{c \in C}\frac{\sum_i Y_{\text{skel},i,c} \cdot \hat{Y}_{i,c}}{\sum_i Y_{\text{skel},i,c}} + \lambda_{\text{BCE}} \cdot \mathbb{E}[w(x)\cdot\text{BCE}(x)]
```

## Dataset: STARE

This project utilizes the STARE dataset, specifically aimed at retinal imagery. Data augmentation techniques, including rotations, flips, and slight deformations, significantly improve model generalization and robustness.

- **Initial Training Set**: 80 images  
- **Post-Augmentation Training Set**: 240 images (3 augmentations per image)  
- **Test Set**: 20 images  

## Results

The proposed loss function significantly improves segmentation accuracy:

| Metric              | Skeleton Recall + Weighted BCE (%) | BCE Only (%) |
|---------------------|------------------------------------|--------------|
| **Jaccard Index**   | **64.35**                          | 63.43        |
| **F1-score**        | **78.27**                          | 77.48        |
| **Recall**          | **91.12**                          | 74.54        |
| **Precision**       | 68.92                              | **82.14**    |
| **Accuracy**        | 95.59                              | **96.28**    |


<img src="https://raw.githubusercontent.com/MariusDragic/Unet4Retina/main/images/comparison.png" alt="Segmentation Comparison" width="75%"/>

*Figure 2 — Visual comparison of segmentation results. Top: BCE Loss alone. Bottom: Skeleton Recall + Weighted BCE Loss.*

Compared to the BCE-only configuration, the hybrid loss preserves vascular continuity and reveals fine capillary details often missed in standard segmentation.  


**Qualitative Improvement**:

- Enhanced segmentation of finer blood vessels.
- Reduced fragmentation, preserving anatomical consistency.
- Improved recall metric essential for clinical diagnosis.

## Usage

### Requirements

- Python 3.10+  
- TensorFlow / Keras  
- NumPy, Pandas, OpenCV, scikit-image  
- Matplotlib, Jupyter Notebook  

Install dependencies:

```bash
    pip install -r requirements.txt
```

### Training

Run the provided Jupyter notebook to reproduce training steps and evaluation:

```bash
    jupyter notebook notebooks/training_evaluation.ipynb
```

## References

Here are the two main references i was based on to make my researches:

- U-Net: [Olaf Ronneberger et al., 2015](https://arxiv.org/abs/1505.04597)  
- Skeleton Recall Loss: [Yannick Kirchhoff et al., 2024](https://arxiv.org/abs/2404.03010)  

## Author

- **Marius Dragic** - [marius.dragic@student-cs.fr](mailto:marius.dragic@student-cs.fr)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


*This README was created to rigorously present and document the methodology, architecture, and findings associated with U-Net applied to retinal vessel segmentation.*
