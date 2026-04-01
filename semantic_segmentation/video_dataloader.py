import os
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from collections import defaultdict
import csv
import re

def load_events(csv_path: str):
    """
    Returns: dict[video] -> list of (frame_t, dt)
    frame_t is 0-based frame index in the sorted mask list used by the scan.
    """
    events = defaultdict(list)
    count = 0
    with open(csv_path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            v = row["video"]
            ft = int(row["frame_t"])
            dt = int(row.get("dt", 1))
            events[v].append((ft, dt))
            count += 1
            if count > 1000:
                break
    return events

def load_camvid_class_dict(csv_path):
    class_names = []
    color_to_idx = {}

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            name = row["name"]
            r = int(row["r"])
            g = int(row["g"])
            b = int(row["b"])

            class_names.append(name)
            color_to_idx[(r, g, b)] = idx

    return class_names, color_to_idx

def camvid_color_mask_to_index_mask(mask_pil, color_to_idx, ignore_index=255):
    mask_np = np.array(mask_pil, dtype=np.uint8)   # [H, W, 3]
    h, w, _ = mask_np.shape

    index_mask = np.full((h, w), ignore_index, dtype=np.uint8)

    for color, idx in color_to_idx.items():
        matches = np.all(mask_np == np.array(color, dtype=np.uint8), axis=-1)
        index_mask[matches] = idx

    return Image.fromarray(index_mask, mode="L")

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


class VSPWClipDataset(Dataset):
    def __init__(self, config, split, events_csv, event_sampling, frames_per_video = 5, sampling: str = "window", augment=True, seed = 42):
        self.root = config['data']['vspw_root_dir']

        self.image_dir = os.path.join(self.root, 'JPEGImages')
        self.mask_dir = os.path.join(self.root, 'SegmentationClass')

        self.augment = augment
        image_resolution = config['ren']['parameters']['image_resolution']
        self.transform = ImageMaskTransform(crop_size=image_resolution)

        self.data_dir = os.path.join(self.root, "data")
        split_path = os.path.join(self.root, f"{split}.txt")
        with open(split_path, "r") as f:
            self.video_names = [line.strip() for line in f if line.strip()]
        self.seed = seed
        self.frames_per_video = frames_per_video
        self.sampling = sampling
        self.rng = np.random.default_rng(seed)
        self.split = split
        self.base_seed = seed
        
        self.events_csv = events_csv
        self.event_sampling = event_sampling
        self.events = load_events(events_csv) if (events_csv and event_sampling) else None
        self.samples = self._build_samples()
        if self.event_sampling and len(self.samples) == 0:
            raise ValueError(
                f"VSPWClipDataset(split={split}) produced 0 samples with event_sampling=True. "
                f"Check events_csv path/content and that CSV video IDs overlap split {split}.txt."
            )

    def _list_frame_stems(self, video_name: str):
        origin_dir = os.path.join(self.data_dir, video_name, "origin")
        if not os.path.isdir(origin_dir):
            raise FileNotFoundError(f"Missing origin dir: {origin_dir}")

        frames = [fn for fn in os.listdir(origin_dir) if fn.lower().endswith(".jpg") and not fn.lower().startswith("._")]
        frames.sort()
        stems = [os.path.splitext(fn)[0] for fn in frames]
        return stems

    def _pick_k(self, stems):
        k = min(self.frames_per_video, len(stems))
        if k == 0:
            return []

        if self.sampling == "first":
            return stems[:k]

        if self.sampling == "window":

            if k == 0:
                return []
            start = int(self.rng.integers(0, len(stems) - k + 1))
            window = stems[start:start + k]
            return window
        raise ValueError(f"Unknown sampling='{self.sampling}'")

    def _pick_k_centered(self, stems, center_idx: int):
        k = min(self.frames_per_video, len(stems))
        if k == 0:
            return []

        # Choose a window [start, start+k) that contains center_idx
        # Clamp start to valid range
        start_min = max(0, center_idx - k + 1)
        start_max = min(center_idx, len(stems) - k)
        if start_max < start_min:
            start = max(0, min(center_idx, len(stems) - k))
        else:
            start = int(self.rng.integers(start_min, start_max + 1))

        return stems[start:start + k]

    def set_epoch(self, epoch):
        self.rng = np.random.default_rng(self.base_seed + epoch)
        self.samples = self._build_samples()

    def _build_samples(self):
        samples = []

        # EVENT-DRIVEN sampling: each sample corresponds to one event (or a few per video)
        if self.event_sampling and self.events is not None:
            for v in self.video_names:
                if v not in self.events:
                    continue
                stems = self._list_frame_stems(v)
                if len(stems) == 0:
                    continue

                # For each event, create one sample window centered around frame_t
                for (frame_t, dt) in self.events[v]:
                    center = min(max(frame_t, 0), len(stems) - 1)
                    picked = self._pick_k_centered(stems, center_idx=center)
                    if len(picked) > 0:
                        samples.append((v, picked))
            return samples

        # ORIGINAL behavior
        for v in self.video_names:
            stems = self._list_frame_stems(v)
            picked = self._pick_k(stems)
            if len(picked) > 0:
                samples.append((v, picked))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_name, stems = self.samples[idx]

        images = []
        masks = []

        for stem in stems:
            origin_path = os.path.join(self.data_dir, video_name, "origin", f"{stem}.jpg")
            mask_path   = os.path.join(self.data_dir, video_name, "mask",   f"{stem}.png")

            image = Image.open(origin_path).convert("RGB")
            w, h = image.size

            if self.split == "test" or (not os.path.exists(mask_path)):
                mask = Image.fromarray(np.full((h, w), 255, dtype=np.uint8), mode="L")
            else:
                mask = Image.open(mask_path).convert("L")

            image_t, mask_t = self.transform(image, mask, self.augment)


            mask_t = mask_t.long()
            mask_t[mask_t == 0] = 255
            mask_t = mask_t - 1
            mask_t[mask_t == 254] = 255
            mask_t[(mask_t < 0) | (mask_t >= 124)] = 255
            images.append(image_t)
            masks.append(mask_t)

        images = torch.stack(images, dim=0)
        masks  = torch.stack(masks,  dim=0)

        return {
            "image": images,
            "mask": masks,
            "video": video_name,
            "frames": stems,
            "mask_shape": (h, w),
            "image_id": f"{video_name}/{stems[0]}",
        }


class CamVidClipDataset(Dataset):
    def __init__(self, config, split, frames_per_video=5, sampling="window", augment=True, seed=42):
        self.root = config["data"]["camvid_root_dir"]
        self.split = split
        self.frames_per_video = frames_per_video
        self.sampling = sampling
        self.augment = augment
        self.base_seed = seed
        self.rng = np.random.default_rng(seed)

        self.class_dict_path = os.path.join(self.root, "class_dict.csv")
        self.class_names, self.color_to_idx = load_camvid_class_dict(self.class_dict_path)
        self.num_classes = len(self.class_names)
        print(f"Loaded CamVid classes: {self.num_classes}")

        image_resolution = config["ren"]["parameters"]["image_resolution"]
        self.transform = ImageMaskTransform(crop_size=image_resolution)

        # Adjust folder names if needed
        self.image_dir = os.path.join(self.root, split)
        self.mask_dir = os.path.join(self.root, f"{split}_labels")

        if not os.path.isdir(self.image_dir):
            raise FileNotFoundError(f"Missing image dir: {self.image_dir}")
        if split != "test" and not os.path.isdir(self.mask_dir):
            raise FileNotFoundError(f"Missing mask dir: {self.mask_dir}")

        self.samples = self._build_samples()

    def set_epoch(self, epoch):
        self.rng = np.random.default_rng(self.base_seed + epoch)
        self.samples = self._build_samples()

    def _parse_filename(self, fname):
        """
        Supports both:
            0006R0_f00930.png
            0001TP_009210.png
        """
        stem = os.path.splitext(fname)[0]

        m = re.match(r"(.+?)_f?(\d+)$", stem)
        if m is None:
            raise ValueError(f"Filename does not match expected pattern: {fname}")

        prefix = m.group(1)
        frame_idx = int(m.group(2))
        return prefix, frame_idx

    def _list_and_group_frames(self):
        grouped = defaultdict(list)

        files = [
            f for f in os.listdir(self.image_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg")) and not f.startswith("._")
        ]

        for fname in files:
            prefix, frame_idx = self._parse_filename(fname)
            grouped[prefix].append((frame_idx, fname))

        for prefix in grouped:
            grouped[prefix].sort(key=lambda x: x[0])  # sort by frame number

        return grouped

    def _pick_k(self, file_list):
        """
        file_list: already sorted list of filenames from the same prefix/video
        """
        k = min(self.frames_per_video, len(file_list))
        if k == 0:
            return []

        if self.sampling == "first":
            return file_list[:k]

        if self.sampling == "window":
            start = int(self.rng.integers(0, len(file_list) - k + 1))
            return file_list[start:start + k]

        raise ValueError(f"Unknown sampling='{self.sampling}'")

    def _build_samples(self):
        samples = []
        grouped = self._list_and_group_frames()

        for prefix, items in grouped.items():
            file_list = [fname for _, fname in items]
            if len(file_list) < self.frames_per_video:
                continue

            for start in range(0, len(file_list) - self.frames_per_video + 1):
                picked = file_list[start:start + self.frames_per_video]
                samples.append((prefix, picked))

        return samples

    def __len__(self):
        return len(self.samples)

    def _mask_name_from_image_name(self, image_name):
        """
        Common CamVid naming:
            image: 0001TP_006690.png
            mask : 0001TP_006690_L.png
        Change this if your labels use a different naming rule.
        """
        stem, _ = os.path.splitext(image_name)
        return f"{stem}_L.png"

    def __getitem__(self, idx):
        video_name, frame_files = self.samples[idx]

        images = []
        masks = []

        h = w = None

        for fname in frame_files:
            image_path = os.path.join(self.image_dir, fname)
            image = Image.open(image_path).convert("RGB")
            w, h = image.size

            if self.split == "test":
                mask = Image.fromarray(np.full((h, w), 255, dtype=np.uint8), mode="L")
            else:
                mask_name = self._mask_name_from_image_name(fname)
                mask_path = os.path.join(self.mask_dir, mask_name)

                if not os.path.exists(mask_path):   
                    raise FileNotFoundError(f"Missing mask file: {mask_path}")

                mask_rgb = Image.open(mask_path).convert("RGB")
                mask = camvid_color_mask_to_index_mask(mask_rgb, self.color_to_idx)

            image_t, mask_t = self.transform(image, mask, self.augment)
            mask_t = mask_t.long()

            # DO NOT use VSPW remapping unless your labels require it
            # mask_t[mask_t == 255] = 255

            images.append(image_t)
            masks.append(mask_t)

        images = torch.stack(images, dim=0)   # [T, C, H, W]
        masks = torch.stack(masks, dim=0)     # [T, H, W]

        return {
            "image": images,
            "mask": masks,
            "video": video_name,       # here "video" is just the shared prefix
            "frames": frame_files,
            "mask_shape": (h, w),
            "image_id": f"{video_name}/{os.path.splitext(frame_files[0])[0]}",
        }