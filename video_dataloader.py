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


class VSPWClipDataset(Dataset):
    def __init__(self, config, split, frames_per_video = 5, sampling: str = "window", augment=True, seed = 42):
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
        self.samples = self._build_samples()

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

    def set_epoch(self, epoch):
        self.rng = np.random.default_rng(self.base_seed + epoch)
        self.samples = self._build_samples()

    def _build_samples(self):
        samples = []
        for v in self.video_names:
            stems = self._list_frame_stems(v)
            picked = self._pick_k(stems)
            if len(picked) > 0:
                samples.append((v, picked))
            #for stem in picked:
            #    samples.append((v, stem))
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