import os
import sys
import json
import cv2
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
import random
import string
from matplotlib import pyplot as plt
import mmap
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, WeightedRandomSampler
import torchvision.transforms as T
from pycocotools import mask
from pycocotools.coco import COCO
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

sys.path.append('segment_anything/')
from segment_anything.sam2.build_sam import build_sam2
from segment_anything.sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from task_utils import deduplicate_masks


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def collate_fn(batch):
    v1 = {}
    v1['images'] = torch.stack([item[0]['image'] for item in batch])
    v1['regions'] = [torch.tensor(np.array(item[0]['regions'])) for item in batch]
    v1['region_ids'] = torch.stack([item[0]['region_ids'] for item in batch])
    v1['loss_mask'] = torch.stack([item[0]['loss_mask'] for item in batch])
    v1['grid_points'] = [torch.tensor(np.array(item[0]['grid_points'])) for item in batch]
    v2 = {}
    v2['images'] = torch.stack([item[1]['image'] for item in batch])
    v2['regions'] = [torch.tensor(np.array(item[1]['regions'])) for item in batch]
    v2['region_ids'] = torch.stack([item[1]['region_ids'] for item in batch])
    v2['loss_mask'] = torch.stack([item[1]['loss_mask'] for item in batch])
    v2['grid_points'] = [torch.tensor(np.array(item[1]['grid_points'])) for item in batch]
    return v1, v2


class COCODataset(Dataset):
    def __init__(self, config, split):
        self.coco = COCO(config['data'][f'coco_{split}_annotations_path'])
        self.image_ids = self.coco.getImgIds()
        self.image_dir = config['data'][f'coco_{split}_images_dir']
        self.split = split

        # Define the image transforms
        image_resolution = config['parameters']['image_resolution']
        self.transform = T.Compose([
            T.ToTensor(),
            T.Resize((image_resolution, image_resolution), antialias=True),
        ])

        # Cache region masks on to the disk as RLE
        self.rle_cache_dir = config['data']['coco_regions_rle_cache_dir']
        sam2 = build_sam2(config['pretrained']['sam2_hieral_config'], config['pretrained']['sam2_hieral_ckpt'],
                          device=device, apply_postprocessing=True)
        self.mask_generator = SAM2AutomaticMaskGenerator(sam2, output_mode='coco_rle', stability_score_thresh=0.9)
        if not config['data']['are_coco_rles_cached']:
            self._cache_mask_rles()

        # Cache regions masks as binaries for faster processing
        self.binary_cache_dir = Path(config['data']['coco_regions_binary_cache_dir'])
        os.makedirs(self.binary_cache_dir, exist_ok=True)
        self.cache_metadata_path = os.path.join(self.binary_cache_dir, f'{split}_cache_metadata.pkl')
        self.binary_cache_path = os.path.join(self.binary_cache_dir, f'{split}_binary_cache.bin')
        self.num_workers = config['parameters']['num_workers']
        if not (os.path.exists(self.cache_metadata_path) and os.path.exists(self.binary_cache_path)):
            self._cache_mask_binaries()

        # Initialize the binary cache by loading the meta-data and performing memory mapping of the cache file
        with open(self.cache_metadata_path, 'rb') as f:
            self.cache_metadata = pickle.load(f)
        self.binary_file = open(self.binary_cache_path, 'rb')
        self.mm = mmap.mmap(self.binary_file.fileno(), 0, access=mmap.ACCESS_READ)

    def _cache_mask_rles(self):
        os.makedirs(self.rle_cache_dir, exist_ok=True)
        transform = T.Compose([
            T.Resize((1024, 1024), antialias=True),
            T.ToTensor()
        ])

        # Run SAM on all images in the data pool and save mask RLEs in a json file
        for image_idx in tqdm(range(len(self)), desc=f'Saving mask RLEs for coco2017 {self.split}'):
            image_info = self.coco.loadImgs(self.image_ids[image_idx])[0]
            image_path = os.path.join(self.image_dir, image_info['file_name'])
            image = Image.open(image_path).convert('RGB')
            image = transform(image).permute(1, 2, 0).numpy()
            save_path = os.path.join(self.rle_cache_dir, f'coco2017-{self.split}-{image_idx}.json')
            if os.path.exists(save_path):
                continue
            
            regions = self.mask_generator.generate(np.array(image))
            with open(save_path, 'w') as f:
                json.dump(regions, f)
    
    def _cache_mask_binaries(self):
        metadata = {}
        offset = 0
        with open(self.binary_cache_path, 'wb') as binary_file:
            for image_idx in tqdm(range(len(self)), desc=f'Saving mask binary for coco2017 {self.split}'):
                json_path = os.path.join(self.rle_cache_dir, f'coco2017-{self.split}-{image_idx}.json')
                
                # Load masks and convert them to binary format
                masks = []
                total_size = 0
                with open(json_path, 'r') as f:
                    regions_rle = json.load(f)
                    for rle in regions_rle:
                        mask_array = mask.decode(rle['segmentation'])
                        packed_mask = np.packbits(mask_array.astype(np.uint8))
                        mask_bytes = packed_mask.tobytes()
                        masks.append(mask_bytes)
                        total_size += len(mask_bytes)
                
                # Save metadata for the masks
                metadata[image_idx] = {
                    'offset': offset,
                    'sizes': [len(m) for m in masks],
                    'shape': mask_array.shape
                }
                
                # Write masks to binary file
                for mask_bytes in masks:
                    binary_file.write(mask_bytes)
                    offset += len(mask_bytes)
        
        with open(self.cache_metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_info = self.coco.loadImgs(self.image_ids[idx])[0]
        image_path = os.path.join(self.image_dir, image_info['file_name'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        
        # Read mask from the memory-mapped file
        regions = []
        metadata = self.cache_metadata[idx]
        offset = metadata['offset']
        shape = metadata['shape']
        for size in metadata['sizes']:
            self.mm.seek(offset)
            mask_bytes = self.mm.read(size)
            packed_mask = np.frombuffer(mask_bytes, dtype=np.uint8)
            unpacked_mask = np.unpackbits(packed_mask)[:np.prod(shape)].reshape(shape)
            resized_mask = cv2.resize(unpacked_mask, (image.shape[1], image.shape[2]))
            regions.append(resized_mask)
            offset += size
        return image, regions

    def __del__(self):
        self.mm.close()
        self.binary_file.close()


class Ego4DDataset(Dataset):
    def __init__(self, config, split, num_videos=3000, frame_duration=15):
        image_dir = config['data'][f'ego4d_{split}_images_dir']
        mask_dir = config['data'][f'ego4d_{split}_regions_rle_cache_dir']
        image_files = [os.path.join(image_dir, filename) for filename in sorted(os.listdir(image_dir))][:num_videos]
        mask_files = [os.path.join(mask_dir, filename) for filename in sorted(os.listdir(mask_dir))][:num_videos]
        self.split = split

        # Define the image transforms
        image_resolution = config['parameters']['image_resolution']
        self.transform = T.Compose([
            T.ToTensor(),
            T.Resize((image_resolution, image_resolution), antialias=True),
        ])

        # Create a look up table for accessing the data
        self.dataset_idx = {}
        idx = 0
        def process_file(args):
            image_file, mask_file = args
            entries = []
            with open(image_file, 'rb') as f:
                record = pickle.load(f)
                num_images = record['num_frames']
                for image_idx in range(0, num_images, 5 * frame_duration):
                    entries.append((image_file, mask_file, image_idx))
            return entries
        with ThreadPoolExecutor() as executor:
            results = executor.map(process_file, zip(image_files, mask_files))
            for file_entries in results:
                for entry in file_entries:
                    self.dataset_idx[idx] = entry
                    idx += 1

        # Cache regions masks as binaries for faster processing
        self.binary_cache_dir = Path(config['data']['ego4d_regions_binary_cache_dir'])
        os.makedirs(self.binary_cache_dir, exist_ok=True)
        self.cache_metadata_path = os.path.join(self.binary_cache_dir, f'{split}_cache_metadata.pkl')
        self.binary_cache_path = os.path.join(self.binary_cache_dir, f'{split}_binary_cache.bin')
        self.num_workers = config['parameters']['num_workers']
        if not (os.path.exists(self.cache_metadata_path) and os.path.exists(self.binary_cache_path)):
            self._cache_mask_binaries()

        # Initialize the binary cache by loading the meta-data and performing memory mapping of the cache file
        with open(self.cache_metadata_path, 'rb') as f:
            self.cache_metadata = pickle.load(f)
        self.binary_file = open(self.binary_cache_path, 'rb')
        self.mm = mmap.mmap(self.binary_file.fileno(), 0, access=mmap.ACCESS_READ)

    def _cache_mask_binaries(self):
        metadata = {}
        offset = 0
        with open(self.binary_cache_path, 'wb') as binary_file:
            for _, mask_file, image_idx in tqdm(self.dataset_idx.values(), desc=f'Saving mask binary for ego4d {self.split}'):
                # Load masks and convert them to binary format
                masks = []
                total_size = 0
                with open(mask_file, 'r') as f:
                    regions_rles = json.load(f)
                    image_key = f'frame-{image_idx}'
                    for rle in regions_rles[image_key]:
                        mask_array = mask.decode(rle['segmentation'])
                        packed_mask = np.packbits(mask_array.astype(np.uint8))
                        mask_bytes = packed_mask.tobytes()
                        masks.append(mask_bytes)
                        total_size += len(mask_bytes)
                
                # Save metadata for the masks
                metadata[(mask_file, image_idx)] = {
                    'offset': offset,
                    'sizes': [len(m) for m in masks],
                    'shape': mask_array.shape
                }
                
                # Write masks to binary file
                for mask_bytes in masks:
                    binary_file.write(mask_bytes)
                    offset += len(mask_bytes)
        
        with open(self.cache_metadata_path, 'wb') as f:
            pickle.dump(metadata, f)

    def __len__(self):
        return len(self.dataset_idx)

    def __getitem__(self, idx):
        image_file, mask_file, image_idx = self.dataset_idx[idx]
        with open(image_file, 'rb') as f:
            record = pickle.load(f)
            image = record['frames'][image_idx]
            image = self.transform(image)

        # Read mask from the memory-mapped file
        regions = []
        mask_file_key = mask_file
        metadata = self.cache_metadata[(mask_file_key, image_idx)]
        offset = metadata['offset']
        shape = metadata['shape']
        for size in metadata['sizes']:
            self.mm.seek(offset)
            mask_bytes = self.mm.read(size)
            packed_mask = np.frombuffer(mask_bytes, dtype=np.uint8)
            unpacked_mask = np.unpackbits(packed_mask)[:np.prod(shape)].reshape(shape)
            resized_mask = cv2.resize(unpacked_mask, (image.shape[1], image.shape[2]))
            regions.append(resized_mask)
            offset += size
        return image, regions
    
    def __del__(self):
        self.mm.close()
        self.binary_file.close()
    

class RENDataset(Dataset):
    def __init__(self, config, split):
        image_datasets = {}
        image_datasets = {}
        if f'coco_{split}' in config['data'][f'{split}_datasets']:
            image_datasets[f'coco_{split}'] = COCODataset(config, split)
        if f'ego4d_{split}' in config['data'][f'{split}_datasets']:
            image_datasets[f'ego4d_{split}'] = Ego4DDataset(config, split)
        
        # Create a list of all datasets and a look-up table to index images in the cumulative data pool
        self.split = split
        self.datasets = []
        self.dataset_idxs = {}
        idx = 0
        for dataset_idx, dataset_name in enumerate(config['data'][f'{split}_datasets']):
            dataset = image_datasets[dataset_name]
            self.datasets.append(dataset)
            for image_idx in range(len(dataset)):
                self.dataset_idxs[idx] = (dataset_idx, image_idx)
                idx += 1
        assert len(self.datasets) > 0, 'No dataset is specified'

        # Define weights for our data sampler
        dataset_weights = config['data']['weights']
        dataset_weights = [w / sum(dataset_weights) for w in dataset_weights]
        sample_weights = []
        for dataset_idx, dataset in enumerate(self.datasets):
            dataset_weight = dataset_weights[dataset_idx]
            dataset_samples = len(dataset)
            sample_weights.extend([dataset_weight / dataset_samples] * dataset_samples)
        self.sample_weights = torch.tensor(sample_weights)

        # Additional parameters
        self.image_resolution = config['parameters']['image_resolution']
        self.deduplicate = config['parameters']['deduplicate_masks']
        self.grid_size = config['architecture']['grid_size']
        x_coords = np.linspace(0, self.image_resolution - 1, self.grid_size, dtype=int)
        y_coords = np.linspace(0, self.image_resolution - 1, self.grid_size, dtype=int)
        self.grid_points = np.array([(y, x) for y in y_coords for x in x_coords])
        self.max_prompts = config['parameters']['max_prompts']
        self.upsample_features = config['parameters']['upsample_features']
        self.patch_size = config['pretrained']['patch_sizes'][0]

    def __len__(self):
        return len(self.dataset_idxs)
    
    def __getitem__(self, idx):
        dataset_idx, image_idx = self.dataset_idxs[idx]
        dataset = self.datasets[dataset_idx]
        image, regions = dataset[image_idx]
        if self.deduplicate:
            regions = deduplicate_masks(regions)

        # Get two views of the image and  regions
        image_v1, regions_v1 = self.apply_transforms(image, regions)
        image_v2, regions_v2 = self.apply_transforms(image, regions)

        # Subsample prompts for training
        subsampled_grid_idxs_v1 = random.sample(range(len(self.grid_points)), self.max_prompts)
        subsampled_grid_idxs_v2 = random.sample(range(len(self.grid_points)), self.max_prompts)
        grid_points_v1 = self.grid_points[subsampled_grid_idxs_v1]
        grid_points_v2 = self.grid_points[subsampled_grid_idxs_v2]

        # Arrange the regions in prompt order
        regions_v1, region_ids_v1, loss_mask_v1 = self.arrange_regions(regions_v1, grid_points_v1, null_region_id=-1)
        regions_v2, region_ids_v2, loss_mask_v2 = self.arrange_regions(regions_v2, grid_points_v2, null_region_id=-2)
        
        # Downsample the region masks for attention supervision if we are not upsampling features
        if not self.upsample_features:
            masks_v1 = torch.tensor(np.array(regions_v1)).unsqueeze(1).float()
            masks_v2 = torch.tensor(np.array(regions_v2)).unsqueeze(1).float()
            mask_size = (image.shape[1] // self.patch_size, image.shape[2] // self.patch_size)
            masks_v1_resized = F.interpolate(masks_v1, size=mask_size, mode='area').squeeze(1)  
            masks_v2_resized = F.interpolate(masks_v2, size=mask_size, mode='area').squeeze(1) 
            regions_v1 = masks_v1_resized.numpy()
            regions_v2 = masks_v2_resized.numpy()

        # Collate info for the two views
        v1 = {
            'image': image_v1,
            'regions': regions_v1,
            'region_ids': region_ids_v1,
            'loss_mask': loss_mask_v1,
            'grid_points': grid_points_v1,
        }
        v2 = {
            'image': image_v2,
            'regions': regions_v2,
            'region_ids': region_ids_v2,
            'loss_mask': loss_mask_v2,
            'grid_points': grid_points_v2,
        }
        return v1, v2
    
    def apply_transforms(self, image, regions):
        image_pil = T.ToPILImage()(image)
        regions_pil = [Image.fromarray(mask.astype(np.uint8) * 255) for mask in regions]

        do_flip = np.random.rand() > 0.5
        brightness_param = 0.4
        contrast_param = 0.4
        saturation_param = 0.4
        sharpness_param = np.random.uniform(0, 2)
        rotation_angle = np.random.uniform(-45, 45)
        shear_x = np.random.uniform(0, 15)
        shear_y = np.random.uniform(0, 15)
        crop_size = np.random.randint(int(0.3 * image.shape[2]), image.shape[2])
        crop_params = T.RandomCrop.get_params(image_pil, output_size=(crop_size, crop_size))

        def synchronized_transform(img, blur_and_jitter=False):
            if do_flip:
                img = T.functional.hflip(img)
            if blur_and_jitter:
                img = T.ColorJitter(brightness=brightness_param, contrast=contrast_param, saturation=saturation_param)(img)
                img = T.RandomAdjustSharpness(sharpness_factor=sharpness_param, p=1.0)(img)
            img = T.functional.affine(img, translate=(0.0, 0.0), scale=1.0, angle=rotation_angle, shear=(shear_x, shear_y))
            img = T.functional.crop(img, *crop_params)
            img = T.functional.resize(img, (image.shape[1], image.shape[2]), interpolation=Image.BICUBIC)
            return img

        transformed_image = T.ToTensor()(synchronized_transform(image_pil, blur_and_jitter=True))
        transformed_regions = [np.array(synchronized_transform(mask_pil)) // 255 for mask_pil in regions_pil]
        return transformed_image, transformed_regions
    
    def arrange_regions(self, regions, grid_points, null_region_id=-1):
        arranged_regions, region_ids, loss_mask = [], [], []
        for point in grid_points:
            y, x = point
            regions_on_point = []
            for region_id, region in enumerate(regions):
                if region[y, x]:
                    regions_on_point.append((np.sum(region), region, region_id))
            regions_on_point.sort(key=lambda x: x[0])
            if len(regions_on_point):
                selected_region_idx = len(regions_on_point) // 2
                arranged_regions.append(regions_on_point[selected_region_idx][1])
                region_ids.append(regions_on_point[selected_region_idx][2])
                loss_mask.append(1)
            else:
                arranged_regions.append(np.zeros_like(regions[0]))
                region_ids.append(null_region_id)
                loss_mask.append(0)
        region_ids = torch.tensor(np.array(region_ids))
        loss_mask = torch.tensor(np.array(loss_mask))
        return arranged_regions, region_ids, loss_mask
    
    def visualize_regions(self, image, regions, region_ids, grid_points, save_dir='vis'):
        image_dir = ''.join(random.choices(string.ascii_letters + string.digits, k=5))
        os.makedirs(f'{save_dir}/{image_dir}', exist_ok=True)
        plt.imshow(np.clip(image.permute(1, 2, 0).numpy(), 0.0, 1.0))
        plt.savefig(f'{save_dir}/{image_dir}/image.jpg')
        plt.clf()
        for region_idx, (region, region_id, grid_point) in enumerate(zip(regions, region_ids, grid_points)):
            plt.imshow(np.clip(image.permute(1, 2, 0).numpy() * np.clip(region[..., None] + 0.2, 0.0, 1.0), 0.0, 1.0))
            plt.scatter([grid_point[1]], [grid_point[0]], marker='o', s=10, c='red')
            plt.savefig(f'{save_dir}/{image_dir}/region-q{region_idx}-i{region_id}.jpg')
            plt.clf()

    def get_weighted_sampler(self):
        return WeightedRandomSampler(weights=self.sample_weights, num_samples=len(self), replacement=True)