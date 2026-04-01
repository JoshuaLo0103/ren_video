import os, json
import yaml
import numpy as np
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as T

from segment_anything.sam2.build_sam import build_sam2, build_sam2_hf
from segment_anything.sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from tqdm import tqdm

device = "cuda"

def build_sam2_from_config(cfg):
    ckpt = cfg["pretrained"]["sam2_hieral_ckpt"]
    config_file = cfg["pretrained"]["sam2_hieral_config"]
    if ckpt.startswith("facebook/") or not os.path.isfile(ckpt):
        model_id = ckpt if ckpt.startswith("facebook/") else "facebook/sam2.1-hiera-large"
        return build_sam2_hf(model_id, device=device, apply_postprocessing=True)
    return build_sam2(config_file, ckpt, device=device, apply_postprocessing=True)

def list_frame_stems(data_dir, video_name):
    origin_dir = os.path.join(data_dir, video_name, "origin")
    frames = [fn for fn in os.listdir(origin_dir) if fn.lower().endswith(".jpg") and not fn.lower().startswith("._")]
    frames.sort()
    return [os.path.splitext(fn)[0] for fn in frames]

def rle_cache_path(rle_dir, split, video, stem):
    return os.path.join(rle_dir, f"vspw-{split}-{video}-{stem}.json")



def precache_split(cfg, split):
    root = cfg["data"]["vspw_root_dir"]
    data_dir = os.path.join(root, "data")
    rle_dir = cfg["data"]["vspw_regions_rle_cache_dir"]
    os.makedirs(rle_dir, exist_ok=True)

    split_path = os.path.join(root, f"{split}.txt")
    with open(split_path, "r") as f:
        videos = [line.strip() for line in f if line.strip()]

    sam2 = build_sam2_from_config(cfg)
    mask_gen = SAM2AutomaticMaskGenerator(sam2, output_mode="coco_rle", stability_score_thresh=0.9)

    tf = T.Compose([
        T.Resize((1024, 1024), antialias=True),
        T.ToTensor()
    ])

    # Count total frames first (for accurate progress)
    total_frames = 0
    for video in videos:
        stems = list_frame_stems(data_dir, video)
        total_frames += len(stems)

    print(f"Total frames to process for {split}: {total_frames}")

    pbar = tqdm(total=total_frames, desc=f"Precaching {split}", dynamic_ncols=True)

    for video in videos:
        stems = list_frame_stems(data_dir, video)
        for stem in stems:
            out_path = rle_cache_path(rle_dir, split, video, stem)

            if not os.path.exists(out_path):
                img_path = os.path.join(data_dir, video, "origin", f"{stem}.jpg")
                img = Image.open(img_path).convert("RGB")
                img_np = tf(img).permute(1, 2, 0).numpy()

                regions = mask_gen.generate(np.array(img_np))

                with open(out_path, "w") as f:
                    json.dump(regions, f)

            pbar.update(1)

    pbar.close()

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    if args.split in ("train", "all"):
        precache_split(cfg, "train")
    if args.split in ("val", "all"):
        precache_split(cfg, "val")
    if args.split in ("test", "all"):
        precache_split(cfg, "test")
