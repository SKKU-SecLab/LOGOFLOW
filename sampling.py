# This file includes code from the rectified-flow-pytorch project:
# https://github.com/lucidrains/rectified-flow-pytorch
# 
# Original author: Phil Wang (lucidrains)
# Licensed under the MIT License
#
# Modifications by: SKKU-SecLab
# Date: 2025-11-21

import torch
import os
from rectified_flow_pytorch import RectifiedFlow, ImageDataset, Unet, Trainer, load_siamese_and_ocr_models

model = Unet(dim = 64)

siamese_model = load_siamese_and_ocr_models()

rectified_flow = RectifiedFlow(model, siamese_model)

img_dataset = ImageDataset(
    folder = './datasets/AT&T',
    image_size = 256
)

trainer = Trainer(
    rectified_flow,
    dataset = img_dataset,
    num_samples=16,
    checkpoint_path='./checkpoints/AT&T/checkpoint.70000.pt',
    result_path ='./sampling/AT&T'
)

trainer.sample_and_save_individual_images(checkpoint_path='./checkpoints/AT&T/checkpoint.70000.pt', result_path ='./sampling/AT&T', num_samples=16)


