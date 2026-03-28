#!/bin/bash

# Download REN DINO ViT-B/8
echo "Downloading REN DINO ViT-B/8..."
mkdir -p logs/ren-dino-vitb8/
wget -O logs/ren-dino-vitb8/checkpoint.pth "https://huggingface.co/savyak2/ren-dino-vitb8/resolve/main/checkpoint.pth"

# Download REN DINOv2 ViT-L/14
echo "Downloading REN DINOv2 ViT-L/14..."
mkdir -p logs/ren-dinov2-vitl14/
wget -O logs/ren-dinov2-vitl14/checkpoint.pth "https://huggingface.co/savyak2/ren-dinov2-vitl14/resolve/main/checkpoint.pth"

# Download REN OpenCLIP ViT-g/14
echo "Downloading REN OpenCLIP ViT-g/14..."
mkdir -p logs/ren-openclip-vitg14/
wget -O logs/ren-openclip-vitg14/checkpoint.pth "https://huggingface.co/savyak2/ren-openclip-vitg14/resolve/main/checkpoint.pth"

echo "Download complete."
