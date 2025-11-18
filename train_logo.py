import torch
from rectified_flow_pytorch import RectifiedFlow, ImageDataset, Unet, Trainer, load_siamese_and_ocr_models

import datetime
import json
import logging
import os
import sys
import time
from pathlib import Path

logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )



logger = logging.getLogger(__name__)


model = Unet(dim = 64)


siamese_model = load_siamese_and_ocr_models()


'''logger.info(str(siamese_model.siamese_model))
print("------------------------------------------------------------------------------------")
logger.info(str(siamese_model.ocr_model))'''


rectified_flow = RectifiedFlow(model, siamese_model)


img_dataset = ImageDataset(
    folder = './datasets/AT&T',
    image_size = 256
)

trainer = Trainer(
    rectified_flow,
    dataset = img_dataset,
    num_train_steps = 100000,
    results_folder = './results/clean/siam/hyper/5/AT&T', 
    checkpoints_folder = './checkpoints/clean/siam/hyper/5/AT&T',
    save_results_every = 1000,
    checkpoint_every = 10000
)


'''checkpoint_path = './checkpoints/clean/xai/hyper/0.1/instagram/checkpoint.10000.pt'
trainer.load(checkpoint_path)
trainer.forward(start_step = 10000)'''
trainer()