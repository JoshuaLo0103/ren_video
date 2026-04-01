import os, json
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

def atomic_write_bytes(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)

def _process_one_video_val(args):
    """
    Cache format per video_id:
      cache_root/video_id/meta.json
      cache_root/video_id/masks/000000.npz  # masks for annot_idx=0 (frame_idx_24=0)
      cache_root/video_id/masks/000001.npz  # annot_idx=1 (frame_idx_24=4)
      ...
    Each npz stores:
      masks: uint8 [K,H,W] (binary)
      obj_ids: string array [K]
      frame_idx_24: int
    """
    (video_id, sav_root_str, cache_root_str, image_resolution, annot_stride, overwrite) = args
    sav_root = Path(sav_root_str)
    cache_root = Path(cache_root_str)

    jpeg_dir = sav_root / "JPEGImages_24fps" / video_id
    anno_dir = sav_root / "Annotations_6fps" / video_id
    if not jpeg_dir.exists() or not anno_dir.exists():
        return (video_id, "skipped", "missing jpeg/anno dir")

    obj_dirs = sorted([p for p in anno_dir.iterdir() if p.is_dir()])
    if not obj_dirs:
        return (video_id, "skipped", "no obj dirs")

    out_dir = cache_root / video_id
    meta_path = out_dir / "meta.json"
    masks_dir = out_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    # Determine how many annot frames exist by scanning png indices
    max_frame_idx_24 = -1
    for od in obj_dirs:
        for p in od.glob("*.png"):
            try:
                fi = int(p.stem)  # 00000, 00004, ...
                max_frame_idx_24 = max(max_frame_idx_24, fi)
            except:
                pass

    if max_frame_idx_24 < 0:
        return (video_id, "skipped", "no png frames")

    # since masks are at 24fps indices spaced by annot_stride (should be 4)
    num_annot = max_frame_idx_24 // annot_stride + 1

    # write meta (always OK to overwrite meta)
    meta = {
        "video_id": video_id,
        "sav_root": str(sav_root),
        "jpeg_dir": str(jpeg_dir),
        "anno_dir": str(anno_dir),
        "num_annot_frames": int(num_annot),
        "annot_stride": int(annot_stride),
        "image_resolution": int(image_resolution),
    }
    atomic_write_bytes(meta_path, json.dumps(meta).encode("utf-8"))

    # Make OpenCV not spawn extra threads per process
    try:
        cv2.setNumThreads(0)
    except:
        pass

    cached = 0
    skipped = 0

    for aidx in range(num_annot):
        frame_idx_24 = aidx * annot_stride
        out_npz = masks_dir / f"{aidx:06d}.npz"
        if out_npz.exists() and not overwrite:
            skipped += 1
            continue

        # Load frame just to get size & verify it exists (optional but helpful)
        frame_path = jpeg_dir / f"{frame_idx_24:05d}.jpg"
        if not frame_path.exists():
            # some videos may be shorter / missing a frame
            skipped += 1
            continue

        # You can avoid decoding the jpg if you trust resolution; decoding is cheap relative to PNG storm.
        frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if frame is None:
            skipped += 1
            continue

        # Resize target (matches your ToTensor+Resize in dataset)
        H = W = int(image_resolution)

        masks = []
        obj_ids = []

        fname = f"{frame_idx_24:05d}.png"
        for od in obj_dirs:
            p = od / fname
            if not p.exists():
                continue
            m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if m is None:
                continue
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
            m = (m > 0).astype(np.uint8)
            masks.append(m)
            obj_ids.append(od.name)

        if len(masks) == 0:
            masks_arr = np.zeros((0, H, W), dtype=np.uint8)
            obj_ids_arr = np.array([], dtype="<U1")
        else:
            masks_arr = np.stack(masks, axis=0).astype(np.uint8)  # [K,H,W]
            obj_ids_arr = np.array(obj_ids)

        # Save compressed
        np.savez_compressed(
            out_npz,
            masks=masks_arr,
            obj_ids=obj_ids_arr,
            frame_idx_24=np.int32(frame_idx_24),
        )
        cached += 1

    return (video_id, "ok", f"cached={cached} skipped={skipped} num_annot={num_annot}")

def preprocess_sav_val_cache(
    sav_root: str,
    cache_root: str,
    split: str = "val",
    image_resolution: int = 224,
    annot_stride: int = 4,
    overwrite: bool = False,
    num_workers: int = 12,
):
    sav_root = Path(sav_root)
    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)

    list_path = sav_root / f"sav_{split}.txt"
    assert list_path.exists(), f"Missing split list: {list_path}"

    video_ids = [ln.strip() for ln in list_path.read_text().splitlines() if ln.strip()]
    assert video_ids, "Empty split list"

    work_items = [
        (vid, str(sav_root), str(cache_root), int(image_resolution), int(annot_stride), bool(overwrite))
        for vid in video_ids
    ]

    ok = skipped = err = 0
    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        futures = [ex.submit(_process_one_video_val, args) for args in work_items]
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Cache SAV {split}"):
            vid, status, msg = fut.result()
            if status == "ok":
                ok += 1
            elif status == "skipped":
                skipped += 1
            else:
                err += 1
                print(f"[ERROR] {vid}: {msg}")

    print(f"Done. ok={ok} skipped={skipped} err={err} total={len(video_ids)}")

if __name__ == "__main__":
    preprocess_sav_val_cache(
        sav_root="/scratch-nvme/joshua68/SAV/sav_val",
        cache_root="/scratch-nvme/joshua68/SAV/sav_cache_val",
        split="val",
        image_resolution=518,   # set to your config['parameters']['image_resolution']
        annot_stride=4,
        overwrite=False,
        num_workers=12,
    )