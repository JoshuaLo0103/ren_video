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
from huggingface_hub import hf_hub_download
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from task_utils import print_log
import gzip


def collate_fn(batch):
    v1 = {}
    v1['images'] = torch.stack([item[0]['image'] for item in batch])                # [B,T,C,H,W]
    v1['regions'] = torch.stack([torch.from_numpy(np.array(item[0]['regions'])).to(torch.uint8)
                                 for item in batch])                                # [B,T,P,H,W]
    v1['region_ids'] = torch.stack([item[0]['region_ids'] for item in batch])       # [B,T,P]
    v1['loss_mask'] = torch.stack([item[0]['loss_mask'] for item in batch])         # [B,T,P]
    v1['grid_points'] = torch.stack([torch.from_numpy(np.array(item[0]['grid_points'])).to(torch.long)
                                     for item in batch])                             # [B,P,2]

    v2 = {}
    v2['images'] = torch.stack([item[1]['image'] for item in batch])
    v2['regions'] = torch.stack([torch.from_numpy(np.array(item[1]['regions'])).to(torch.uint8)
                                 for item in batch])
    v2['region_ids'] = torch.stack([item[1]['region_ids'] for item in batch])
    v2['loss_mask'] = torch.stack([item[1]['loss_mask'] for item in batch])
    v2['grid_points'] = torch.stack([torch.from_numpy(np.array(item[1]['grid_points'])).to(torch.long)
                                     for item in batch])
    return v1, v2



class SAVClipDataset(Dataset):
    def __init__(self, config, split="train"):
        self.split = split
        self.clip_len = int(config["parameters"].get("clip_len", 5))
        self.clip_stride = int(config["parameters"].get("clip_stride", 1))
        self.annot_to_video_stride = int(config["parameters"].get("sav_annot_stride", 4))
        self.patch_size = int(config["pretrained"]["patch_sizes"][0])
        self.root = Path(config["data"][f"sav_{split}_root"])
        self.cache_root = Path(config["data"][f"sav_{split}_cache_root"])  # NEW
        assert self.cache_root.exists(), f"Missing cache root: {self.cache_root}"

        image_resolution = int(config["parameters"]["image_resolution"])
        self.transform = T.Compose([
            T.ToTensor(),
            T.Resize((image_resolution, image_resolution), antialias=True),
        ])
        self.items = []  # {video_id, video_path, video_h, video_w, num_annot_frames, cache_dir}
        # cache dirs are keyed by mp4 stem
        for meta_path in sorted(self.cache_root.rglob("meta.json")):
            meta = json.loads(meta_path.read_text())
            n = int(meta["num_annot_frames"])
            if n < self.clip_len:
                continue
            self.items.append({
                "video_id": meta["video_id"],
                "video_path": meta["video_path"],
                "video_h": int(meta["video_height"]),
                "video_w": int(meta["video_width"]),
                "num_annot_frames": n,
                "cache_dir": str(meta_path.parent),
            })
        assert self.items, f"No cached SAV videos found under {self.cache_root}"
        self.samples = []
        for item_idx, it in enumerate(self.items):
            n = it["num_annot_frames"]
            for s in range(0, n - self.clip_len + 1, self.clip_stride):
                self.samples.append((item_idx, s))
    def __len__(self):
        return len(self.samples)

    def _load_frame_jpg(self, cache_dir: Path, annot_idx: int) -> np.ndarray:
        img_path = cache_dir / "frames" / f"{annot_idx:06d}.jpg"
        frame_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if frame_bgr is None:
            raise RuntimeError(f"Failed to read cached jpg: {img_path}")
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return frame_rgb

    def _read_frame_cv2(self, cap: cv2.VideoCapture, frame_idx: int) -> np.ndarray:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame_bgr = cap.read()
        if not ok:
            raise RuntimeError(f"Failed to read frame {frame_idx}")
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    def _load_rles_frame(self, cache_dir: Path, annot_idx: int):
        p = cache_dir / f"rles_{annot_idx:06d}.pkl.gz"
        with open(p, "rb") as f:
            blob = f.read()
        return pickle.loads(gzip.decompress(blob))  # List[Dict]

    def __getitem__(self, idx: int):
        item_idx, start_annot = self.samples[idx]
        it = self.items[item_idx]
        cache_dir = Path(it["cache_dir"])

        images_t = []
        regions_per_t = []

        for t in range(self.clip_len):
            annot_idx = start_annot + t

            frame_rgb = self._load_frame_jpg(cache_dir, annot_idx)
            img = self.transform(Image.fromarray(frame_rgb))
            images_t.append(img)

            rles = self._load_rles_frame(cache_dir, annot_idx)

            Ht, Wt = img.shape[1], img.shape[2]
            #h2, w2 = Ht // self.patch_size, Wt // self.patch_size
            resized_masks = []
            for rle in rles:
                m = mask.decode(rle)
                if m.ndim == 3:
                    m = m[..., 0]
                m = m.astype(np.uint8)
                m_ds = cv2.resize(m, (Ht, Wt))
                m_ds = (m_ds > 0).astype(np.uint8)
                resized_masks.append(m_ds)

            regions_per_t.append(resized_masks)
        images = torch.stack(images_t, dim=0)
        meta = {"video_id": it["video_id"], "start_annot_frame": start_annot}
        return images, regions_per_t, meta

class SAVValCachedClipDataset(Dataset):
    def __init__(self, config, split="val"):
        assert split in ["val", "test"]
        self.split = split

        self.clip_len = int(config["parameters"].get("clip_len", 5))
        self.clip_stride = int(config["parameters"].get("clip_stride", 1))
        self.annot_to_video_stride = int(config["parameters"].get("sav_annot_stride", 4))
        assert self.annot_to_video_stride == 4

        self.sav_root = Path(config["data"][f"sav_{split}_root"])
        self.cache_root = Path(config["data"][f"sav_{split}_cache_root"])  # <-- add this key
        self.list_path = self.sav_root / f"sav_{split}.txt"

        self.jpeg_root = self.sav_root / "JPEGImages_24fps"
        assert self.list_path.exists()
        assert self.jpeg_root.exists()
        assert self.cache_root.exists(), f"Missing cache_root_npz: {self.cache_root}"

        image_resolution = int(config["parameters"]["image_resolution"])
        self.transform = T.Compose([
            T.ToTensor(),
            T.Resize((image_resolution, image_resolution), antialias=True),
        ])

        self.video_ids = [ln.strip() for ln in self.list_path.read_text().splitlines() if ln.strip()]

        self.items = []
        self.samples = []

        for vid in self.video_ids:
            meta_path = self.cache_root / vid / "meta.json"
            masks_dir = self.cache_root / vid / "masks"
            frame_dir = self.jpeg_root / vid
            if not meta_path.exists() or not masks_dir.exists() or not frame_dir.exists():
                continue
            meta = json.loads(meta_path.read_text())
            n = int(meta["num_annot_frames"])
            if n < self.clip_len:
                continue

            item_idx = len(self.items)
            self.items.append({
                "video_id": vid,
                "num_annot_frames": n,
                "masks_dir": masks_dir,
                "frame_dir": frame_dir,
            })
            for s in range(0, n - self.clip_len + 1, self.clip_stride):
                self.samples.append((item_idx, s))

        assert self.samples, "No cached val samples found"

        try:
            cv2.setNumThreads(0)
        except:
            pass

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item_idx, start_annot = self.samples[idx]
        it = self.items[item_idx]

        images_t = []
        regions_per_t = []

        for t in range(self.clip_len):
            annot_idx = start_annot + t
            frame_idx_24 = annot_idx * self.annot_to_video_stride

            frame_path = it["frame_dir"] / f"{frame_idx_24:05d}.jpg"
            frame_bgr = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if frame_bgr is None:
                raise RuntimeError(f"Failed to read frame: {frame_path}")
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            img = self.transform(Image.fromarray(frame_rgb))
            images_t.append(img)

            npz_path = it["masks_dir"] / f"{annot_idx:06d}.npz"
            z = np.load(npz_path, allow_pickle=False)
            masks = z["masks"]  # [K,H,W] uint8

            # convert to List[H,W] uint8 like your pipeline expects
            regs_t = [masks[k] for k in range(masks.shape[0])]
            regions_per_t.append(regs_t)

        images = torch.stack(images_t, dim=0)
        meta = {"video_id": it["video_id"], "start_annot_frame": start_annot}
        return images, regions_per_t, meta


class RENDataset(Dataset):
    def __init__(self, config, split):
        self.split = split
        # Build dataset objects only for those requested
        datasets_map = {}

        if split == "train":
            datasets_map["sav_train"] = SAVClipDataset(config, split="train")  # your cached/RLE train
        else:
            datasets_map[f"sav_{split}"] = SAVValCachedClipDataset(config, split=split)  # folder-based val/test

        # Create list of datasets and a global index -> (dataset_idx, local_idx) table
        self.datasets = []
        self.dataset_idxs = {}
        global_idx = 0

        requested = config['data'][f'{split}_datasets']
        for dataset_idx, dataset_name in enumerate(requested):
            assert dataset_name in datasets_map, f"Dataset {dataset_name} not constructed. Check config paths/keys."
            ds = datasets_map[dataset_name]
            self.datasets.append(ds)

            for local_idx in range(len(ds)):
                self.dataset_idxs[global_idx] = (dataset_idx, local_idx)
                global_idx += 1

        assert len(self.datasets) > 0, "No dataset is specified"

        # Weighted sampler support
        dataset_weights = config['data'].get('weights', [1.0] * len(self.datasets))
        assert len(dataset_weights) == len(self.datasets), \
            f"weights length {len(dataset_weights)} must match number of datasets {len(self.datasets)}"

        s = sum(dataset_weights)
        dataset_weights = [w / s for w in dataset_weights]

        sample_weights = []
        for dataset_idx, ds in enumerate(self.datasets):
            w = dataset_weights[dataset_idx]
            n = len(ds)
            sample_weights.extend([w / n] * n)

        self.sample_weights = torch.tensor(sample_weights, dtype=torch.float32)

        # Additional parameters (same as before)
        self.image_resolution = config['parameters']['image_resolution']
        self.deduplicate = config['parameters']['deduplicate_masks']
        self.grid_size = config['architecture']['grid_size']

        x_coords = np.linspace(0, self.image_resolution - 1, self.grid_size, dtype=int)
        y_coords = np.linspace(0, self.image_resolution - 1, self.grid_size, dtype=int)
        self.grid_points = np.array([(y, x) for y in y_coords for x in x_coords])
        self.patch_size = config['pretrained']['patch_sizes'][0]

        self.max_prompts = config['parameters']['max_prompts']
        self.upsample_features = config['parameters']['upsample_features']
        

    def set_epoch(self, epoch: int):
        for d in self.datasets:
            if hasattr(d, "set_epoch"):
                d.set_epoch(epoch)

    def __len__(self):
        return len(self.dataset_idxs)
    
    def __getitem__(self, idx):
        
        dataset_idx, image_idx = self.dataset_idxs[idx]
      
        dataset = self.datasets[dataset_idx]
        image, regions, meta = dataset[image_idx]
     

        if self.deduplicate:
            regions = [deduplicate_masks(r) for r in regions]

        image_v1, regions_v1 = self.apply_transforms_clip(image, regions)
   
        image_v2, regions_v2 = self.apply_transforms_clip(image, regions)
        
    
        subsampled_grid_idxs_v1 = random.sample(range(len(self.grid_points)), self.max_prompts)
        
        subsampled_grid_idxs_v2 = random.sample(range(len(self.grid_points)), self.max_prompts)
   
        grid_points_v1 = self.grid_points[subsampled_grid_idxs_v1]
        grid_points_v2 = self.grid_points[subsampled_grid_idxs_v2]

        regions_v1, region_ids_v1, loss_mask_v1 = self.arrange_regions_clip(regions_v1, grid_points_v1, null_region_id=-1)
   
        regions_v2, region_ids_v2, loss_mask_v2 = self.arrange_regions_clip(regions_v2, grid_points_v2, null_region_id=-2)
    
        
        v1 = {
            "image": image_v1,               # [T,C,H,W]
            "regions": regions_v1,           # List[T][P][H,W] OR np [T,P,h',w']
            "region_ids": region_ids_v1,     # [T,P]
            "loss_mask": loss_mask_v1,       # [T,P]
            "grid_points": grid_points_v1,   # [P,2]
        }
        v2 = {
            "image": image_v2,
            "regions": regions_v2,
            "region_ids": region_ids_v2,
            "loss_mask": loss_mask_v2,
            "grid_points": grid_points_v2,
        }
        return v1, v2
    
    def apply_transforms_clip(self, images, regions):
        T_ = images.shape[0]
        H, W = images.shape[2], images.shape[3]

        do_flip = np.random.rand() > 0.5

        # Image-only jitter/affine/crop
        brightness_param = 0.4
        contrast_param = 0.4
        saturation_param = 0.4
        sharpness_param = np.random.uniform(0, 2)
        rotation_angle = np.random.uniform(-45, 45)
        shear_x = np.random.uniform(0, 15)
        shear_y = np.random.uniform(0, 15)
        crop_size = np.random.randint(int(0.3 * W), W)

        first_pil = T.ToPILImage()(images[0])
        crop_params = T.RandomCrop.get_params(first_pil, output_size=(crop_size, crop_size))

        def synced_img(pil_img):
            if do_flip:
                pil_img = T.functional.hflip(pil_img)
            pil_img = T.ColorJitter(brightness=brightness_param, contrast=contrast_param, saturation=saturation_param)(pil_img)
            pil_img = T.RandomAdjustSharpness(sharpness_factor=sharpness_param, p=1.0)(pil_img)
            pil_img = T.functional.affine(
                pil_img, translate=(0.0, 0.0), scale=1.0, angle=rotation_angle, shear=(shear_x, shear_y)
            )
            pil_img = T.functional.crop(pil_img, *crop_params)
            pil_img = T.functional.resize(pil_img, (H, W), interpolation=Image.BICUBIC)
            return pil_img

        def synced_mask(mask_np: np.ndarray):
            # Keep mask binary after geometry transforms (use nearest-neighbor interpolation).
            mask_np = (mask_np > 0).astype(np.uint8)
            mask_pil = Image.fromarray(mask_np * 255, mode="L")

            if do_flip:
                mask_pil = T.functional.hflip(mask_pil)

            mask_pil = T.functional.affine(
                mask_pil,
                translate=(0.0, 0.0),
                scale=1.0,
                angle=rotation_angle,
                shear=(shear_x, shear_y),
                interpolation=T.InterpolationMode.NEAREST,
            )
            mask_pil = T.functional.crop(mask_pil, *crop_params)
            mask_pil = T.functional.resize(mask_pil, (H, W), interpolation=T.InterpolationMode.NEAREST)

            out = np.array(mask_pil)
            return (out > 0).astype(np.uint8)

        out_frames = []
        out_regions = []

        for t in range(T_):
            # images
            img_pil = T.ToPILImage()(images[t])
            img_tf = T.ToTensor()(synced_img(img_pil))
            out_frames.append(img_tf)

            # Apply the same geometry transforms to masks
            regs_t = []
            for m in regions[t]:   # m is [h2,w2]
                regs_t.append(synced_mask(m))
            out_regions.append(regs_t)

        return torch.stack(out_frames, dim=0), out_regions

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

    def arrange_regions_clip(self, regions_per_t, grid_points, null_region_id=-1):
        arranged_all, region_ids_all, loss_mask_all = [], [], []
        for t in range(len(regions_per_t)):
            arranged_t, ids_t, lm_t = self.arrange_regions(regions_per_t[t], grid_points, null_region_id)
            arranged_all.append(arranged_t)      # List[P][h2,w2]
            region_ids_all.append(ids_t)         # [P]
            loss_mask_all.append(lm_t)           # [P]
        return arranged_all, torch.stack(region_ids_all), torch.stack(loss_mask_all)
