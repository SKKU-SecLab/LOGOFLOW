import sys
import os
import torch 

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

def load_siamese_and_ocr_models(siamese_path="./OCR_siamese_utils/output/targetlist_lr0.01/bit.pth.tar", ocr_path="./OCR_siamese_utils/demo_downgrade.pth.tar"):
    
    from guided_diffusion.script_util import create_siamese_model
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    siamese_model = create_siamese_model(siamese_path, ocr_path)
    
    siamese_model.siamese_model.to(device)
    siamese_model.ocr_model.to(device)
    
    
    return siamese_model