import torch
import torch.nn as nn
import torch.nn.functional as F
from captum.attr import LayerGradCam
import numpy as np
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os

from .OCR_siamese_utils.demo import ocr_main2, ocr_model_config

class SiameseModel(nn.Module):
    def __init__(self, model: nn.Module, ocr_model: nn.Module):
        super().__init__()
        self.siamese_model = model   
        self.ocr_model = ocr_model


    def l2_norm(self, x):
        if len(x.shape):
            x = x.reshape((x.shape[0], -1))
        return F.normalize(x, p=2, dim=1)

    def named_parameters(self, prefix: str = '', recurse: bool = True):
        return self.siamese_model.named_parameters(prefix=prefix, recurse=recurse)

    def pred_siamese_OCR(
        self, img, model, ocr_model, imshow=False, title=None, grayscale=False, require_grad=False, return_cam=False
    ):
        model.train()
        for params in model.parameters():
            params.requires_grad = False

        device = "cuda" if torch.cuda.is_available() else "cpu"
        img_size = 224
        mean = torch.tensor([0.5, 0.5, 0.5], device=img.device).view(1, 3, 1, 1)
        std = torch.tensor([0.5, 0.5, 0.5], device=img.device).view(1, 3, 1, 1)

        img = F.interpolate(img, size=img_size, mode="bicubic")
        ocr_emb = ocr_main2(image_path=img, model=ocr_model, height=None, width=None).to(device)
        img = (img - mean) / std
        img = img.to(device)

        logo_feat = model.features3(img, ocr_emb)
        logo_feat = self.l2_norm(logo_feat)

        if not return_cam:
            return logo_feat

        target_layer = model.body.block4.unit03
        grad_cam = LayerGradCam(model.forward, target_layer)
        output_feat = model.forward(img, ocr_emb)
        target_indices = output_feat.argmax(dim=1).tolist()
        
        #target_index = output_feat[0].argmax().item()
        attributions = grad_cam.attribute(
            img, target=target_indices, additional_forward_args=(ocr_emb,)
        )
        
        attributions_upsampled = F.interpolate(attributions, size=(256, 256), mode='bicubic', align_corners=False)
        cam_tensor = (attributions_upsampled - attributions_upsampled.min()) / (attributions_upsampled.max() - attributions_upsampled.min() + 1e-10)
        return logo_feat, cam_tensor

    def siamese_loss(self, x1, x2):
        img_feat1 = self.pred_siamese_OCR(x1, self.siamese_model, self.ocr_model)
        img_feat2 = self.pred_siamese_OCR(x2, self.siamese_model, self.ocr_model)
        sim_list2 = torch.stack([img_feat1[i] @ img_feat2[i].T for i in range(img_feat1.shape[0])])
        return sim_list2

    def get_attention_map(self, x1, x2):
        self.siamese_model.eval()
        device = x2.device
        #x2 = x2.clone().detach().requires_grad_(True)
        
        logo_feat, cam = self.pred_siamese_OCR(
            x2, self.siamese_model, self.ocr_model, return_cam=True
        )
        
        sim_score = self.siamese_loss(x1, x2)
        
        if isinstance(cam, np.ndarray):
            cam = torch.from_numpy(cam).to(device)
        return cam.to(device), sim_score.detach()


    def visualize_attention_map(self, attention_map, save_path=None, idx=0, title="Attention Map"):
    
        plt.figure(figsize=(4, 4))
        
        if isinstance(attention_map, torch.Tensor):
            if attention_map.dim() == 4:  # (B, 1, H, W)
                att_map = attention_map[idx, 0].cpu().detach().numpy()
            elif attention_map.dim() == 3 and attention_map.shape[0] == 1:  # (1, H, W)
                att_map = attention_map[0].cpu().detach().numpy()
            elif attention_map.dim() == 2:  # (H, W)
                att_map = attention_map.cpu().detach().numpy()
            else:
                raise ValueError(f"Unsupported tensor shape for attention map: {attention_map.shape}")
        elif isinstance(attention_map, np.ndarray):
            att_map = attention_map
        else:
            raise TypeError("attention_map should be torch.Tensor or numpy.ndarray")
            
        att_map = (att_map - att_map.min()) / (att_map.max() - att_map.min() + 1e-10)
        plt.imshow(att_map, cmap='jet')
        plt.colorbar()
        plt.title(title)
        plt.axis('off')
    
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
            
        '''    
        if isinstance(attention_map, torch.Tensor):
            att_map = attention_map[idx, 0].cpu().detach().numpy() if attention_map.dim() == 4 else attention_map.cpu().detach().numpy()
        elif isinstance(attention_map, np.ndarray):
            att_map = attention_map
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
        plt.colorbar()
        plt.title(title)
        plt.axis('off')
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()'''
