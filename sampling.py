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
    num_samples=81,
    checkpoint_path='./checkpoints/clean/xai/hyper/5/AT&T/checkpoint.70000.pt',
    result_path ='./sampling/clean/xai/hyper/5/AT&T/70000pt'
)

trainer.sample_and_save_individual_images(checkpoint_path='./checkpoints/clean/xai/hyper/5/AT&T/checkpoint.70000.pt', result_path ='./sampling/clean/xai/hyper/5/AT&T/70000pt', num_samples=8)
