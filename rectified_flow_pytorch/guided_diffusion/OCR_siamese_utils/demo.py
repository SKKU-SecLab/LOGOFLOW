from __future__ import absolute_import

# sys.path.append('./')

import argparse
import math
import os
import os.path as osp
import sys
import time

import numpy as np

import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
from torch.backends import cudnn
from torchvision import transforms

from .lib.models.model_builder import ModelBuilder
from .lib.utils.labelmaps import get_vocabulary, labels2strs


def image_process(image_path, imgH=32, imgW=100, keep_ratio=False, min_ratio=1):

    # img = Image.open(image_path).convert('RGB') if isinstance(image_path, str) else image_path.convert('RGB')
    # Now the image_path = tensor_image
    img = image_path
    

    if keep_ratio:
        w, h = img.shape[1], img.shape[2]
        ratio = w / float(h)
        imgW = int(np.floor(ratio * imgH))
        imgW = max(imgH * min_ratio, imgW)
    target_size = [imgH, imgW]

    img = F.interpolate(img, size=target_size, mode="bilinear")

    # img = img.resize((imgW, imgH), Image.BILINEAR)
    # img = transforms.ToTensor()(img)
    img = torch.squeeze(img)
    #print("img after squeeze ", img.shape)
    img.sub_(0.5).div_(0.5)

    return img


def image_process2(image, imgH=32, imgW=100, keep_ratio=False, min_ratio=1):

    if keep_ratio:
        w, h = image.shape[1], image.shape[2]
        print("w ", w)
        print("h ", h)
        ratio = w / float(h)
        imgW = int(np.floor(ratio * imgH))
        imgW = max(imgH * min_ratio, imgW)

    target_size = [imgH, imgW]

    img1 = F.interpolate(image, size=target_size, mode="bilinear")
    
    # print("img1 gradient ", img1)

    img1.sub_(0.5).div_(0.5)

    return img1.requires_grad_()


class DataInfo(object):
    """
    Save the info about the dataset.
    This a code snippet from dataset.py
    """

    def __init__(self, voc_type):
        super(DataInfo, self).__init__()
        self.voc_type = voc_type

        assert voc_type in ["LOWERCASE", "ALLCASES", "ALLCASES_SYMBOLS"]
        self.EOS = "EOS"
        self.PADDING = "PADDING"
        self.UNKNOWN = "UNKNOWN"
        self.voc = get_vocabulary(
            voc_type, EOS=self.EOS, PADDING=self.PADDING, UNKNOWN=self.UNKNOWN
        )
        self.char2id = dict(zip(self.voc, range(len(self.voc))))
        self.id2char = dict(zip(range(len(self.voc)), self.voc))

        self.rec_num_classes = len(self.voc)


def ocr_model_config(checkpoint, height=None, width=None):

    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    #if device == "cuda":
    #    print("using cuda.")
    #    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    #else:
    #    torch.set_default_tensor_type("torch.FloatTensor")

    # Create data loaders
    if height is None or width is None:
        height, width = (32, 100)

    dataset_info = DataInfo("ALLCASES_SYMBOLS")

    # Create model
    model = ModelBuilder(
        arch="ResNet_ASTER",
        rec_num_classes=dataset_info.rec_num_classes,
        sDim=512,
        attDim=512,
        max_len_labels=100,
        eos=dataset_info.char2id[dataset_info.EOS],
        STN_ON=True,
    )

    # Load from checkpoint
    checkpoint = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])

    if device == "cuda":
        model = model.to(device)

    return model


# def ocr_main(image_path, model, height=None, width=None):

#     # device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     device = 'cuda'
#     # Evaluation
#     model.eval()
#     # for params in model.parameters():
#     #     params.requires_grad = False

#     img = image_process(image_path)
#     print("img shape what the hell ", img.shape)
#     print("does it have gradient in ocr_main ", img)
#     # with torch.no_grad():
#     #     img = img.to(device)
#     input_dict = {}
#     # input_dict['images'] = img.unsqueeze(0)
#     input_dict['images'] = img
#     # TODO: testing should be more clean. to be compatible with the lmdb-based testing, need to construct some meaningless variables.
#     dataset_info = DataInfo('ALLCASES_SYMBOLS')
#     rec_targets = torch.IntTensor(1, 100).fill_(1)
#     rec_targets[:,100-1] = dataset_info.char2id[dataset_info.EOS]
#     input_dict['rec_targets'] = rec_targets.to(device)
#     input_dict['rec_lengths'] = [100]
#     # with torch.no_grad():
#     print("input dict ", input_dict)
#     features, decoder_feat = model.features(input_dict)
#     print("features grad in ocr ", features)
#     #features = features.detach().cpu()
#     decoder_feat = decoder_feat.detach().cpu()
#     features = torch.mean(features, dim=1)
#     # print("features grad in ocr ", features)

#     return features


def ocr_main(image_path, model, height=None, width=None):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Evaluation
    model.eval()

    img = image_process(image_path)
    with torch.no_grad():
        img = img.to(device)

    input_dict = {}
    input_dict["images"] = img.unsqueeze(0)
    #print("oriignal unsqueeze ", input_dict["images"].shape)
    # TODO: testing should be more clean. to be compatible with the lmdb-based testing, need to construct some meaningless variables.
    dataset_info = DataInfo("ALLCASES_SYMBOLS")
    rec_targets = torch.IntTensor(1, 100).fill_(1)
    rec_targets[:, 100 - 1] = dataset_info.char2id[dataset_info.EOS]
    input_dict["rec_targets"] = rec_targets.to(device)
    input_dict["rec_lengths"] = [100]

    with torch.no_grad():
        features, decoder_feat = model.features(input_dict)
    features = features.detach().cpu()
    decoder_feat = decoder_feat.detach().cpu()
    features = torch.mean(features, dim=1)

    return features


def ocr_main2(image_path, model, height=None, width=None):
    model.eval()  # for batch normalization layer
    for params in model.parameters():
        params.requires_grad = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Evaluation
    if not image_path.requires_grad:
        image_path.requires_grad_()
        
    #print("ocr_main2 image_path.requires_grad:", image_path.requires_grad)
    img = image_process2(image_path)
    img = img.to(device).requires_grad_()
    #print("ocr_main2 img.requires_grad:", img.requires_grad)
    # img = img.to(torch.float32)
    # print("----------------------------------------")
    # print("type of img: ", img.dtype)
    # print("----------------------------------------")

    # model = model.float()

    features = model.features2(img)
    # print("features gradient? ", features)
    features = torch.mean(features, dim=1)

    return features
