
# Brain Tumor Detection using CNNs

Brain tumors are among the most dangerous forms of cancer, often requiring early detection and treatment. **MRI (Magnetic Resonance Imaging)** is a key technique used by radiologisits to visualize the brain in high 3 Dimensional detail.
In this project, I aim to leverage a Deep CNN architecture to carry out segmentation of brain tumors usig multi-modal MRi scans.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Model Architecture](#model-architecture)
- [Evaluation](#evaluation)
- [Results and Conclusions Drawn](#results-and-conclusions-drawn)

## Project Overview

The focus of this project is to develop an efficient CNN to segment brain tumors by training on the **BRATS 2020 dataset** using a **W-Net** [(more on this later)](#model-architecture) Convolutional Neural Network. The architecture takes 3D volumes (stacked along the z-xis) of T1, T1ce, T2, and FLAIR images and outputs #D segmentation maps labeling tumor sub-regions.

### Objectives: 

- Preprocess 3D MRI slices and create complete volumes.
- Train the model on the volumetric data
- Evaluate model performance nd visualize predictions

## Dataset

For the purpose of this project, I am using the BRATS 2020 dataset containing:
- Muti-modal MRI scans: T1, T1ce, T2, and FLAIR,
- Ground truth (y) segmentation masks.

Each subject contains 3D volumes of shape (240,240,155) - 240 X 240 height and widh, 155 slices per modality.

## Data Preprocessing

## Model Architecture

## Evaluation

## Results and Conclusions Drawn



