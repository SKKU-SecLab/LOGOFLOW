# import os
# import numpy as np
import os
from collections import OrderedDict
import inspect

import torch
import torch.nn as nn

# import matplotlib.pyplot as plt
# from torch import nn
import torch.nn.functional as F

# from skimage.io import imread
# from PIL import Image
import torchvision.transforms as T
from PIL import Image, ImageOps
from torchvision import transforms
import numpy as np 

# import torch
# from collections import OrderedDict
# import matplotlib.pyplot as plt
# from torch.backends import cudnn
# from tqdm import tqdm
# from .phishpedia_siamese.inference import siamese_inference, pred_siamese
# from .OCR_siamese_utils.inference import siamese_inference_OCR, pred_siamese_OCR
# from .OCR_siamese_utils.demo import ocr_model_config
# import yaml
# import subprocess
from .OCR_siamese_utils.demo import ocr_main, ocr_main2, ocr_model_config
from captum.attr import LayerGradCam
import matplotlib.pyplot as plt
import cv2


class SiameseModel:
    def __init__(
        self,
        model,
        ocr_model,
    ):
        super().__init__()
        self.siamese_model = model
        self.ocr_model = ocr_model

    def l2_norm(self, x):
        if len(x.shape):
            x = x.reshape((x.shape[0], -1))
        return F.normalize(x, p=2, dim=1)

    def pred_siamese_OCR(
        self, img, model, ocr_model, imshow=False, title=None, grayscale=False, require_grad=False, return_cam=False
    ):
        """
        Inference for a single image with OCR enhanced model
        :param img_path: image path in str or image in PIL.Image
        :param model: Siamese model to make inference
        :param ocr_model: pretrained OCR model
        :param imshow: enable display of image or not
        :param title: title of displayed image
        :param grayscale: convert image to grayscale or not
        :return feature embedding of shape (2048,)
        """

        """for params in model.parameters():
            params.requires_grad = False"""
        
        #print(f"pred_siamese_OCR input img.requires_grad: {img.requires_grad}")
        
        if require_grad:
            for p in model.parameters():
                p.requires_grad = True
        else:
            for p in model.parameters():
                p.requires_grad = False

        img_size = 224
        mean = torch.tensor([0.5, 0.5, 0.5], device=img.device).view(1, 3, 1, 1)
        std = torch.tensor([0.5, 0.5, 0.5], device=img.device).view(1, 3, 1, 1)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # print("what is device ", device)
        img_transforms = transforms.Compose(
            [
                transforms.Normalize(mean=mean, std=std),
            ]
        )

        img = F.interpolate(img, size=img_size, mode="bicubic")
        
        
        ocr_emb = ocr_main2(image_path=img, model=ocr_model, height=None, width=None)
        ocr_emb = ocr_emb.to(device).requires_grad_()
        
        
        img = (img - mean) / std
        img = img.to(device).requires_grad_()
        #print("img.requires_grad after transforms:", img.requires_grad)
        
        #print(model.features3)
        #print(inspect.getsource(model.features3))
        logo_feat = model.features3(img, ocr_emb)
        logo_feat.requires_grad_()

        logo_feat = self.l2_norm(logo_feat)
        logo_feat.requires_grad_()
        
        if not return_cam:
            return logo_feat
        
        
        target_layer = model.body.block4.unit03
        grad_cam = LayerGradCam(model.forward, target_layer)
        
        output_feat = model.forward(img, ocr_emb)
        #print(output_feat.shape)
        
        target_index = output_feat[0].argmax().item()
        #print("target_index.shape:", target_index)
        
        attributions = grad_cam.attribute(img, target=target_index, additional_forward_args=(ocr_emb,))
        
        attributions_upsampled = F.interpolate(attributions, size = (256, 256), mode='bicubic', align_corners=False)
         
        cam = attributions_upsampled.detach().cpu().squeeze(0).numpy()
        if cam.ndim == 3:
            cam = cam.mean(axis=0)  # (C, H, W) -> (H, W)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-10)
        
        cam_tensor = attributions_upsampled.detach()
        cam_tensor = (cam_tensor - cam_tensor.min()) / (cam_tensor.max() - cam_tensor.min() + 1e-10)
        
        return logo_feat, cam_tensor

    def siamese_loss(self, x1, x2):
        # print("start in siamese x1 grad_fn ", x1)
        # print("start in siamese x2 grad_fn ", x2)
        # print("img_feat1")
        # print("type of x1: ", x1.dtype)
        img_feat1 = self.pred_siamese_OCR(x1, self.siamese_model, self.ocr_model)

        # print("img_feat2")
        # print("type of x2: ", x2.dtype)
        img_feat2 = self.pred_siamese_OCR(x2, self.siamese_model, self.ocr_model)

        sim_list2 = []
        for i in range(img_feat1.shape[0]):
            sim_list2.append(img_feat1[i] @ img_feat2[i].T)

        sim_list2 = torch.stack(sim_list2)

        return sim_list2
        
    
    def get_attention_map(self, x1, x2):
        """
        x1: ground truth image
        x2: generated image
        """
        self.siamese_model.eval()  
        device = x2.device
        
        x2 = x2.clone().detach().requires_grad_(True)
        
        
        logo_feat, cam = self.pred_siamese_OCR(x2, self.siamese_model, self.ocr_model, return_cam=True)
        
        sim_score = self.siamese_loss(x1, x2)
        
        if isinstance(cam, np.ndarray):
            cam = torch.from_numpy(cam).to(device)
        return cam, sim_score.detach()
    


    
    def visualize_attention_map(self, attention_map, save_path=None, idx=0, title="Attention Map"):
        """
        Visualize and optionally save the attention map.
        attention_map: numpy array (H, W)
        """
        plt.figure(figsize=(4, 4))

        if isinstance(attention_map, torch.Tensor):
            if attention_map.dim() == 4:
                att_map = attention_map[idx].cpu().detach().numpy()  # (C, H, W)
            elif attention_map.dim() == 3:
                att_map = attention_map.cpu().detach().numpy()
            elif attention_map.dim() == 2:
                att_map = attention_map.cpu().detach().numpy()
            else:
                raise ValueError(f"Unsupported attention_map dimension: {attention_map.shape}")
        elif isinstance(attention_map, np.ndarray):
            if attention_map.ndim == 4:
                att_map = attention_map[idx]  # (C, H, W)
            elif attention_map.ndim == 3:
                att_map = attention_map
            elif attention_map.ndim == 2:
                att_map = attention_map
            else:
                raise ValueError(f"Unsupported attention_map ndim: {attention_map.shape}")
        else:
            raise TypeError("attention_map should be torch.Tensor or numpy.ndarray")
    
        if att_map.ndim == 3:
            C, H, W = att_map.shape
            if C == 1:
                
                plt.imshow(att_map[0], cmap='jet')
            elif C == 3:
                
                img_rgb = np.transpose(att_map, (1, 2, 0))
                
                img_rgb = (img_rgb - img_rgb.min()) / (img_rgb.max() - img_rgb.min() + 1e-10)
                plt.imshow(img_rgb)
            else:
                raise ValueError(f"Unsupported channel size for visualization: {C}")
        elif att_map.ndim == 2:
            plt.imshow(att_map, cmap='jet')
        else:
            raise ValueError(f"Unexpected shape for visualization: {att_map.shape}")
    
        plt.colorbar()
        plt.title(title)
        plt.axis('off')
    
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
