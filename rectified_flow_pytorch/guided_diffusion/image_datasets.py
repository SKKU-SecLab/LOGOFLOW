import math
import random

import blobfile as bf
import numpy as np
import torch

from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset


def load_data(
    *,
    data_dir,
    batch_size,
    image_size,
    class_cond=False,
    deterministic=False,
    random_crop=False,
    random_flip=False,
    use_distributed=False,
    num_tasks=1,
    global_rank=0,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_files = _list_image_files_recursively(data_dir)
    # print("all files ", all_files)
    
    classes = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        class_names = [bf.basename(path).split("_")[0] for path in all_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]
        
    dataset = ImageDataset(
        image_size,
        all_files,
        classes=classes,
        random_crop=random_crop,
        random_flip=random_flip,
        shard=global_rank if use_distributed else 0,
        num_shards=num_tasks if use_distributed else 1,
    )
    
    if use_distributed:
        generator = torch.Generator(device="cpu")
        sampler = torch.utils.data.DistributedSampler(
            dataset,
            num_replicas=num_tasks,
            rank=global_rank,
            shuffle=not deterministic,
            generator=generator,
        )
    else:
        sampler = None
    
    generator = torch.Generator(device="cpu")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None) and not deterministic,
        sampler = sampler,
        pin_memory=False,
        num_workers=0,
        drop_last=True,
        generator=generator,
        )
    
    # else:
    #     g = torch.Generator(device="cuda")
    #     loader = DataLoader(
    #         dataset,
    #         batch_size=batch_size,
    #         shuffle=True,
    #         generator=g,
    #         pin_memory=False,
    #         num_workers=0,
    #         drop_last=True,
    #     )

    return loader


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        classes=None,
        shard=0,
        num_shards=1,
        random_crop=False,
        random_flip=True,
    ):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.random_crop = random_crop
        self.random_flip = random_flip

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        # print("path ", path)
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")

        if self.random_crop:
            arr = random_crop_arr(pil_image, self.resolution)
        else:
            arr = center_crop_arr(pil_image, self.resolution)

        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]
        arr = arr.astype(np.float32) / 127.5 - 1
        # out_dict = {}
        # out_path = {}
        # if self.local_classes is not None:
        #     out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        # out_path["path"] = path
        # # print("array shape ", arr.shape)
        # return np.transpose(arr, [2, 0, 1]), out_dict["y"]
        
        image_tensor = torch.tensor(np.transpose(arr, [2,0,1]), dtype=torch.float32)
        mask_tensor = self._generate_random_mask(self.resolution)
        
        return image_tensor, mask_tensor
    
    def _generate_random_mask(self, size):
        """다양한 타입의 마스크를 섞어서 생성"""
        # mask_type = random.choice(["box", "circle", "irregular"])
        mask_type = 'box'
        mask = torch.zeros((1, size, size), dtype=torch.float32)

        if mask_type == "box":
            # x = random.randint(0, size // 2)
            # y = random.randint(size // 2, max(size - h, size // 2))            
            # w = random.randint(size // 4, size // 2)
            # h = random.randint(size // 4, size // 2)
            # mask[:, y:y+h, x:x+w] = 1.0
            h = size
            w = size
            y_start = size // 2  # 이미지 세로의 중간부터 시작
            mask[:, y_start:, :] = 1.0
            

        elif mask_type == "circle":
            center_x = random.randint(size // 4, size * 3 // 4)
            center_y = random.randint(size // 4, size * 3 // 4)
            radius = random.randint(size // 8, size // 4)
            yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
            dist = ((xx - center_x) ** 2 + (yy - center_y) ** 2).sqrt()
            mask[0] = (dist < radius).float()

        elif mask_type == "irregular":
            num_vertices = random.randint(3, 6)
            points = [(random.randint(0, size), random.randint(0, size)) for _ in range(num_vertices)]
            mask_img = Image.new("L", (size, size), 0)
            ImageDraw.Draw(mask_img).polygon(points, outline=1, fill=1)
            mask[0] = torch.tensor(np.array(mask_img), dtype=torch.float32)

        return mask


def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2

    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
