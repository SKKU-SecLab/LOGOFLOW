import argparse
import inspect
import os
from collections import OrderedDict

import torch
import torch.optim as optim

from . import gaussian_diffusion as gd

# from .siamese_model4_mr import SiameseModel
from .respace import space_timesteps, SpacedDiffusion

# from .import gaussian_diffusion_mr as gd
from .siamese_model4 import SiameseModel
from .unet import EncoderUNetModel, SuperResModel, UNetModel

# NUM_CLASSES = 1000
NUM_CLASSES = 14
import os

from .OCR_siamese_utils.demo import ocr_model_config

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def siamese_defaults():
    return dict(siamese_path="", ocr_path="")


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def phishpedia_config_OCR_easy2(
    num_classes: int, weights_path: str, ocr_weights_path: str
):
    # load OCR model

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ocr_model = ocr_model_config(checkpoint=ocr_weights_path)

    # from .phishpedia_siamese.siamese_retrain.bit_pytorch.models import KNOWN_MODELS

    from .OCR_siamese_utils.siamese_unified.bit_pytorch.models import KNOWN_MODELS

    model = KNOWN_MODELS["BiT-M-R50x1"](head_size=num_classes, zero_head=True)

    # Load weights
    weights = torch.load(weights_path, map_location=device, weights_only=True)
    weights = weights["model"] if "model" in weights.keys() else weights
    new_state_dict = OrderedDict()
    for k, v in weights.items():
        if k.startswith("module"):
            name = k.split("module.")[1]
        else:
            name = k
        new_state_dict[name] = v

    # for k, v in weights.items():
    #     name = k.split('module.')[1]
    #     new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()

    return model, ocr_model


def create_siamese_model(siamese_path, ocr_path):

    # original model
    SIAMESE_MODEL, OCR_MODEL = phishpedia_config_OCR_easy2(
        num_classes=277,
        weights_path=os.path.join(
            os.path.dirname(__file__), siamese_path.replace("/", os.sep)
        ),
        ocr_weights_path=os.path.join(
            os.path.dirname(__file__), ocr_path.replace("/", os.sep)
        ),
    )

    return SiameseModel(SIAMESE_MODEL, OCR_MODEL)


def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")
