
# Brain Tumor Detection using CNNs

Brain tumors are among the most dangerous forms of cancer, often requiring early detection and treatment. **MRI (Magnetic Resonance Imaging)** is a key technique used by radiologists to visualize the brain in high 3-dimensional detail.
In this project, I aim to leverage a deep CNN architecture to carry out segmentation of brain tumors using multi-modal MRI scans.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Model Architecture](#model-architecture)
- [Evaluation](#evaluation)
- [Results and Conclusions Drawn](#results-and-conclusions-drawn)

## Project Overview

The focus of this project is to develop an efficient CNN to segment brain tumors by training on the **BRATS 2020 dataset** using a **W-Net** [(more on this later)](#model-architecture) Convolutional Neural Network. The architecture takes 3D volumes (stacked along the z-axis) of T1, T1ce, T2, and FLAIR images and outputs 3D segmentation maps labeling tumor sub-regions.

### Objectives: 

- Preprocess 3D MRI slices and create complete volumes.
- Train the model on the volumetric data.
- Evaluate model performance and visualize predictions.

## Dataset

For the purpose of this project, I am using the BRATS 2020 dataset containing:
- Multi-modal MRI scans: T1, T1ce, T2, and FLAIR
- Ground truth segmentation masks (labels).

Each subject contains 3D volumes of shape (240,240,155) - 240 X 240 pixels (height and width), with 155 slices per modality.

## Data Preprocessing

## Model Architecture

## Evaluation

## Results and Conclusions Drawn



