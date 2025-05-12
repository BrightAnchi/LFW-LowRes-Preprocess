# LFW Low-Resolution Preprocessing

This repository contains preprocessing scripts and datasets for generating low-resolution versions of the Labeled Faces in the Wild (LFW) dataset.  
It is designed for experiments in face recognition with degraded input quality, especially for use with super-resolution models such as StarSRGAN.

## 📁 Folder Structure

├── preprocess_lfw_lowres.py # Main preprocessing script
├── datasets/
│ ├── LFW_original/ # Original high-resolution LFW images
│ ├── LFW_lowres/ # Downsampled low-resolution LFW images
│ └── lfw-deepfunneled -> ... # Symlink to original dataset
└── .gitignore


## 🚀 How to Use

To run the preprocessing script and generate low-resolution images:

```bash
python preprocess_lfw_lowres.py

Make sure the original LFW_original dataset exists under datasets/.
