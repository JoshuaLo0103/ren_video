import os, json, gzip, pickle, cv2
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

def choose_ann_path(mp4: Path, ann_source: str) -> Path | None:
    stem = mp4.stem
    auto_json = mp4.with_name(stem + "_auto.json")
    manual_json = mp4.with_name(stem + "_manual.json")
    if ann_source == "manual":
        return manual_json if manual_json.exists() else None
    if ann_source == "auto":
        return auto_json if auto_json.exists() else None
    # prefer_manual
    if manual_json.exists():
        return manual_json
    if auto_json.exists():
        return auto_json
    return None

def atomic_write_bytes(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)

def _process_one_video(args):
    """
    Worker function: process exactly one MP4 + its chosen annotation JSON.
    Returns (video_id, status, message).
    """
    (mp4_str, cache_root_str, ann_source, annot_stride, jpg_quality, overwrite) = args
    mp4 = Path(mp4_str)
    cache_root = Path(cache_root_str)

    video_id = mp4.stem
    ann_path = choose_ann_path(mp4, ann_source)
    if ann_path is None:
        return (video_id, "skipped", "no annotation json found")

    out_dir = cache_root / video_id
    meta_path = out_dir / "meta.json"
    frames_dir = out_dir / "frames"

    if meta_path.exists() and frames_dir.exists() and not overwrite:
        return (video_id, "skipped", "already cached")

    # load annotation once (offline)
    with open(ann_path, "r") as f:
        ann = json.load(f)

    masklet = ann.get("masklet", [])
    num_annot = len(masklet)
    if num_annot == 0:
        return (video_id, "skipped", "empty masklet")

    meta = {
        "video_id": video_id,
        "video_path": str(mp4),
        "ann_path": str(ann_path),
        "video_height": int(ann["video_height"]),
        "video_width": int(ann["video_width"]),
        "num_annot_frames": num_annot,
        "annot_stride": int(annot_stride),
        "ann_source": ann_source,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(meta_path, json.dumps(meta).encode("utf-8"))

    cap = cv2.VideoCapture(str(mp4))
    if not cap.isOpened():
        return (video_id, "error", f"cannot open video {mp4}")

    try:
        for aidx in range(num_annot):
            # 1) save rles for this annotated frame
            rles_path = out_dir / f"rles_{aidx:06d}.pkl.gz"
            if overwrite or (not rles_path.exists()):
                blob = gzip.compress(pickle.dumps(masklet[aidx], protocol=pickle.HIGHEST_PROTOCOL))
                atomic_write_bytes(rles_path, blob)

            # 2) save corresponding decoded frame (annotated frame only)
            img_path = frames_dir / f"{aidx:06d}.jpg"
            if overwrite or (not img_path.exists()):
                frame_idx = aidx * annot_stride
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ok, frame_bgr = cap.read()
                if not ok:
                    return (video_id, "error", f"failed to read frame {frame_idx}")
                cv2.imwrite(
                    str(img_path),
                    frame_bgr,
                    [int(cv2.IMWRITE_JPEG_QUALITY), int(jpg_quality)]
                )
    finally:
        cap.release()

    return (video_id, "ok", f"cached {num_annot} annotated frames")

def preprocess_sav_cache(
    sav_root: str,
    cache_root: str,
    ann_source: str = "prefer_manual",
    annot_stride: int = 4,
    jpg_quality: int = 90,
    overwrite: bool = False,
    num_workers: int = 12,   
):
    sav_root = Path(sav_root)
    cache_root = Path(cache_root)
    mp4_paths = sorted(sav_root.rglob("*.mp4"))
    assert mp4_paths, f"No mp4 found under {sav_root}"

    # Pack args for each worker
    work_items = [
        (str(mp4), str(cache_root), ann_source, annot_stride, jpg_quality, overwrite)
        for mp4 in mp4_paths
    ]

    errors = 0
    skipped = 0

    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        futures = [ex.submit(_process_one_video, args) for args in work_items]

        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Preprocessing SAV (workers={num_workers})"):
            video_id, status, msg = fut.result()
            if status == "error":
                errors += 1
                print(f"[ERROR] {video_id}: {msg}")
            elif status == "skipped":
                skipped += 1

    print(f"Done. skipped={skipped} errors={errors} total={len(mp4_paths)}")

if __name__ == '__main__':
    preprocess_sav_cache(
        sav_root="/scratch-nvme/joshua68/SAV/sav_train/sav_023",
        cache_root="/scratch-nvme/joshua68/SAV/sav_cache_train",
        ann_source="prefer_manual",
        annot_stride=4,
        num_workers=12,  
        overwrite=False,
    )
