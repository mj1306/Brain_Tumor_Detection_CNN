import nibabel as nib
import numpy as np
import os
import tensorflow as tf


def load_brats_data(data_dir, split='train', num_samples=100, img_size=(128, 128)):
    images = []
    masks = []
    
    subdir = '' if split == 'train' else ''
    
    for i in range(1, num_samples + 1):
      
        flair = nib.load(f"{data_dir}/{subdir}/MICCAI_BraTS2020_TrainingData/BraTS20_Training_{i:03d}/BraTS20_Training_{i:03d}_flair.nii").get_fdata()
        t1 = nib.load(f"{data_dir}/{subdir}/MICCAI_BraTS2020_TrainingData/BraTS20_Training_{i:03d}/BraTS20_Training_{i:03d}_t1.nii").get_fdata()
        t1ce = nib.load(f"{data_dir}/{subdir}/MICCAI_BraTS2020_TrainingData/BraTS20_Training_{i:03d}/BraTS20_Training_{i:03d}_t1ce.nii").get_fdata()
        t2 = nib.load(f"{data_dir}/{subdir}/MICCAI_BraTS2020_TrainingData/BraTS20_Training_{i:03d}/BraTS20_Training_{i:03d}_t2.nii").get_fdata()
        seg = nib.load(f"{data_dir}/{subdir}/MICCAI_BraTS2020_TrainingData/BraTS20_Training_{i:03d}/BraTS20_Training_{i:03d}_seg.nii").get_fdata()

      
        flair = tf.image.resize(flair, img_size)
        t1 = tf.image.resize(t1, img_size)
        t1ce = tf.image.resize(t1ce, img_size)
        t2 = tf.image.resize(t2, img_size)
        seg = tf.image.resize(seg, img_size)

        flair = (flair - np.min(flair)) / (np.max(flair) - np.min(flair))
        t1 = (t1 - np.min(t1)) / (np.max(t1) - np.min(t1))
        t1ce = (t1ce - np.min(t1ce)) / (np.max(t1ce) - np.min(t1ce))
        t2 = (t2 - np.min(t2)) / (np.max(t2) - np.min(t2))

        image = np.stack([flair, t1, t1ce, t2], axis=-1)  #Stacks all modalities (image.shape = (128,128,4))

        seg = np.round(seg)  

        images.append(image)
        masks.append(seg)
    

    images = np.array(images)  
    masks = np.array(masks) 

    return images, masks

data_dir = ""
X_train, y_train = load_brats_data(data_dir, split='train', num_samples=369)

X_val, y_val = load_brats_data(data_dir, split='val', num_samples=125)
