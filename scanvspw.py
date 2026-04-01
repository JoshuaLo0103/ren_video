import os
import csv
import argparse
from glob import glob

import numpy as np
from PIL import Image

# Optional: faster CC labeling if scipy exists
try:
    from scipy.ndimage import label as cc_label
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False


def read_mask(path: str) -> np.ndarray:
    return np.array(Image.open(path))


def touches_border(component_mask: np.ndarray) -> bool:
    return (
        component_mask[0, :].any() or component_mask[-1, :].any() or
        component_mask[:, 0].any() or component_mask[:, -1].any()
    )


def connected_components(binary_mask: np.ndarray):
    """
    Returns list[bool(H,W)] for each 4-connected component in binary_mask.
    Uses scipy if available; else BFS fallback.
    """
    if not binary_mask.any():
        return []

    if SCIPY_OK:
        structure = np.array([[0, 1, 0],
                              [1, 1, 1],
                              [0, 1, 0]], dtype=np.int8)
        labeled, n = cc_label(binary_mask.astype(np.uint8), structure=structure)
        return [(labeled == k) for k in range(1, n + 1)]

    # BFS fallback
    H, W = binary_mask.shape
    visited = np.zeros_like(binary_mask, dtype=bool)
    comps = []
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    ys, xs = np.where(binary_mask)
    for y0, x0 in zip(ys, xs):
        if visited[y0, x0]:
            continue
        stack = [(y0, x0)]
        visited[y0, x0] = True
        coords = []
        while stack:
            y, x = stack.pop()
            coords.append((y, x))
            for dy, dx in dirs:
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W and binary_mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    stack.append((ny, nx))
        comp = np.zeros_like(binary_mask, dtype=bool)
        yy, xx = zip(*coords)
        comp[np.array(yy), np.array(xx)] = True
        comps.append(comp)

    return comps


def scan_clip(mask_paths,
              ignore_label=255,
              min_area=300,
              shrink_ratio_thresh=0.25,
              vanish_ratio_thresh=0.10,
              lookahead=2):
    """
    Flag events where a border-touching CC for class c in frame t loses most pixels
    *within the same spatial support* in future frame(s).
    """
    events = []
    if len(mask_paths) < 2:
        return events

    masks = [read_mask(p) for p in mask_paths]

    for t in range(len(masks) - 1):
        m0 = masks[t]
        classes = np.unique(m0)
        classes = classes[classes != ignore_label]

        for c in classes:
            bin0 = (m0 == c)
            comps0 = connected_components(bin0)

            for comp0 in comps0:
                a0 = int(comp0.sum())
                if a0 < min_area:
                    continue
                if not touches_border(comp0):
                    continue

                best_ratio = 1.0
                best_dt = None

                for dt in range(1, lookahead + 1):
                    if t + dt >= len(masks):
                        break
                    m1 = masks[t + dt]
                    remaining = int(((m1 == c) & comp0).sum())
                    ratio = remaining / max(a0, 1)
                    if ratio < best_ratio:
                        best_ratio = ratio
                        best_dt = dt

                if best_dt is None:
                    continue

                if best_ratio <= vanish_ratio_thresh:
                    etype = "vanish"
                elif best_ratio <= shrink_ratio_thresh:
                    etype = "shrink"
                else:
                    continue

                events.append({
                    "frame_t": t,
                    "dt": best_dt,
                    "class_id": int(c),
                    "area_t": a0,
                    "remain_ratio": float(best_ratio),
                    "type": etype
                })

    return events


def load_video_names(test_txt: str):
    names = []
    with open(test_txt, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            # common formats: "video_name" or "video_name something"
            names.append(s.split()[0])
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_txt", type=str, required=True,
                    help="Path to VSPW test.txt listing video names.")
    ap.add_argument("--data_root", type=str, required=True,
                    help="Root directory containing per-video folders (e.g. /scratch-nvme/joshua68/VSPW/data).")
    ap.add_argument("--mask_subdir", type=str, default="mask",
                    help="Subdirectory inside each video folder that contains masks (default: mask).")
    ap.add_argument("--mask_ext", type=str, default="png",
                    help="Mask filename extension (default: png).")
    ap.add_argument("--out_csv", type=str, default="vspw_test_leaving.csv")
    ap.add_argument("--ignore_label", type=int, default=255)
    ap.add_argument("--min_area", type=int, default=300)
    ap.add_argument("--shrink_ratio", type=float, default=0.25)
    ap.add_argument("--vanish_ratio", type=float, default=0.10)
    ap.add_argument("--lookahead", type=int, default=2)
    ap.add_argument("--max_videos", type=int, default=0,
                    help="0 = all; else only scan first N for quick tests.")
    args = ap.parse_args()

    video_names = load_video_names(args.test_txt)
    if args.max_videos and args.max_videos > 0:
        video_names = video_names[:args.max_videos]

    rows = []
    scanned = 0
    missing = 0
    flagged_videos = 0
    total_events = 0

    for vid in video_names:
        mask_dir = os.path.join(args.data_root, vid, args.mask_subdir)
        if not os.path.isdir(mask_dir):
            missing += 1
            continue

        mask_paths = sorted(glob(os.path.join(mask_dir, f"*.{args.mask_ext}")))
        if len(mask_paths) < 2:
            scanned += 1
            continue

        events = scan_clip(
            mask_paths,
            ignore_label=args.ignore_label,
            min_area=args.min_area,
            shrink_ratio_thresh=args.shrink_ratio,
            vanish_ratio_thresh=args.vanish_ratio,
            lookahead=args.lookahead
        )

        scanned += 1
        if events:
            flagged_videos += 1
            total_events += len(events)
            for e in events:
                rows.append({
                    "video": vid,
                    "frame_t": e["frame_t"],
                    "dt": e["dt"],
                    "class_id": e["class_id"],
                    "area_t": e["area_t"],
                    "remain_ratio": f"{e['remain_ratio']:.4f}",
                    "type": e["type"],
                })

    # Write CSV
    fieldnames = ["video", "frame_t", "dt", "class_id", "area_t", "remain_ratio", "type"]
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print("=== VSPW leaving scan ===")
    print(f"Videos listed in test.txt: {len(load_video_names(args.test_txt))}")
    print(f"Videos attempted (after max_videos): {len(video_names)}")
    print(f"Videos scanned (mask dir exists): {scanned}")
    print(f"Videos missing mask dir: {missing}")
    print(f"Flagged videos (>=1 event): {flagged_videos}")
    print(f"Total events: {total_events}")
    print(f"CSV written to: {args.out_csv}")


if __name__ == "__main__":
    main()