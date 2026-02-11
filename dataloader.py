import os
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset


class ImageMaskTransform:
    def __init__(self, crop_size=224, interpolation=T.InterpolationMode.BICUBIC, hflip_prob=0.5):
        self.crop_size = crop_size
        self.hflip_prob = hflip_prob
        self.random_resized_crop = T.RandomResizedCrop(crop_size, interpolation=interpolation)
        self.random_horizontal_flip = T.RandomHorizontalFlip(hflip_prob)

    def __call__(self, image, mask, augment):
        if augment:
            i, j, h, w = self.random_resized_crop.get_params(image, (0.08, 1.0), (3/4, 4/3))
            image = T.functional.crop(image, i, j, h, w)
            mask = T.functional.crop(mask, i, j, h, w)
            image = T.functional.resize(image, (self.crop_size, self.crop_size), interpolation=T.InterpolationMode.BICUBIC)
            mask = T.functional.resize(mask, (self.crop_size, self.crop_size), interpolation=T.InterpolationMode.NEAREST)
            if torch.rand(1) < self.hflip_prob:
                image = T.functional.hflip(image)
                mask = T.functional.hflip(mask)
        else:
            image = T.functional.resize(image, (self.crop_size, self.crop_size), interpolation=T.InterpolationMode.BICUBIC)
            mask = T.functional.resize(mask, (self.crop_size, self.crop_size), interpolation=T.InterpolationMode.NEAREST)

        image = T.functional.to_tensor(image)
        mask = torch.tensor(np.array(mask), dtype=torch.uint8)
        return image, mask


class VOCSegmentation(Dataset):
    def __init__(self, config, image_set='trainval', augment=True):
        self.root = os.path.join(config['data']['voc_root_dir'], f'VOC2012')
        self.image_set = image_set

        self.image_dir = os.path.join(self.root, 'JPEGImages')
        self.mask_dir = os.path.join(self.root, 'SegmentationClass')

        image_set_path = os.path.join(self.root, 'ImageSets', 'Segmentation', f'{image_set}.txt')
        with open(image_set_path) as f:
            self.image_ids = [line.strip() for line in f.readlines()]

        self.augment = augment
        image_resolution = config['ren']['parameters']['image_resolution']
        self.transform = ImageMaskTransform(crop_size=image_resolution)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]

        # Load image
        image_path = os.path.join(self.image_dir, f'{image_id}.jpg')
        image = Image.open(image_path).convert('RGB')

        # Load segmentation mask
        mask_path = os.path.join(self.mask_dir, f'{image_id}.png')
        if self.image_set == 'test':
            mask_width, mask_height = image.size
            mask = Image.fromarray(np.full((mask_height, mask_width), 255, dtype=np.uint8), mode='L')
        else:
            mask = Image.open(mask_path)
            mask_width, mask_height = image.size
        image, mask = self.transform(image, mask, self.augment)
        return {
            'image': image,
            'mask': mask,
            'image_id': image_id,
            'mask_shape': (mask_height, mask_width),
        }
    

class ADESegmentation(Dataset):
    def __init__(self, config, split='training', augment=True):
        self.root_dir = config['data']['ade_root_dir']
        self.split = split
        
        self.image_dir = os.path.join(self.root_dir, 'images', self.split)
        self.mask_dir = os.path.join(self.root_dir, 'annotations', self.split)
        
        self.image_list = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.jpg')])

        self.augment = augment
        image_resolution = config['ren']['parameters']['image_resolution']
        self.transform = ImageMaskTransform(crop_size=image_resolution)

    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        # Load image
        image_name = self.image_list[idx]
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert('RGB')
        
        # Load segmentation mask
        mask_name = os.path.splitext(image_name)[0] + '.png'
        mask_path = os.path.join(self.mask_dir, mask_name)
        mask = Image.open(mask_path)
        mask_width, mask_height = mask.size

        # Apply transformations to the image and the mask
        image, mask = self.transform(image, mask, self.augment)
        mask -= 1
        return {
            'image': image,
            'mask': mask,
            'image_id': image_name[:-4],
            'mask_shape': (mask_height, mask_width),
        }