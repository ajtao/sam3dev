"""Filter SAM3 text-prompted masks using heur12 bboxes for validation.

Generates 3 filtered videos:
  Method 1: Hard mean-IoU threshold per SAM3 track
  Method 2: Per-frame IoU gate
  Method 3: Track assignment with dedup (best SAM3 track per heur12 tid)

Usage:
  python filter_sam3_with_heur12.py --example-dir examples/20240507-jazzy-hedgehog_play3
  python filter_sam3_with_heur12.py --video V --heur12 H --masks-dir M --output-dir O
"""

import os
import argparse
import cv2
import numpy as np
import pandas as pd
from collections import Counter, defaultdict

parser = argparse.ArgumentParser(description="Filter SAM3 masks using heur12 bboxes")
parser.add_argument("--example-dir", type=str, help="Example directory containing video, heur12 csv, and sam3_output/masks")
parser.add_argument("--video", type=str, help="Path to video file")
parser.add_argument("--heur12", type=str, help="Path to heur12 CSV")
parser.add_argument("--masks-dir", type=str, help="Path to SAM3 masks directory")
parser.add_argument("--output-dir", type=str, help="Output directory for filtered videos")
args = parser.parse_args()

if args.example_dir:
    edir = args.example_dir
    VIDEO = os.path.join(edir, "second_half.mp4")
    HEUR12 = os.path.join(edir, "second_half_heur12.csv")
    MASKS_DIR = os.path.join(edir, "sam3_output", "masks")
    OUTPUT_DIR = os.path.join(edir, "output")
elif args.video and args.heur12 and args.masks_dir:
    VIDEO = args.video
    HEUR12 = args.heur12
    MASKS_DIR = args.masks_dir
    OUTPUT_DIR = args.output_dir or os.path.dirname(args.video)
else:
    parser.error("Provide --example-dir OR --video/--heur12/--masks-dir")

os.makedirs(OUTPUT_DIR, exist_ok=True)

W, H = 1920, 1080

# Load heur12
df = pd.read_csv(HEUR12)
grouped = df.groupby("frame_sh")

# Stable colors per heur12 tid
np.random.seed(0)
tid_colors = {}
for tid in range(13):
    tid_colors[tid] = tuple(int(c) for c in np.random.randint(60, 255, 3))

cap_test = cv2.VideoCapture(VIDEO)
num_frames = int(cap_test.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap_test.get(cv2.CAP_PROP_FPS)
cap_test.release()


def get_mask_bbox(mask):
    """Get bounding box from a binary mask at original resolution."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return xs.min(), ys.min(), xs.max(), ys.max()


def compute_iou(box1, box2):
    """IoU between two (x1,y1,x2,y2) boxes."""
    ix1 = max(box1[0], box2[0]); iy1 = max(box1[1], box2[1])
    ix2 = min(box1[2], box2[2]); iy2 = min(box1[3], box2[3])
    if ix1 >= ix2 or iy1 >= iy2:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (a1 + a2 - inter)


def load_frame_masks(fi):
    """Load SAM3 masks and obj_ids for a frame, upscale to original res."""
    mask_path = os.path.join(MASKS_DIR, f"frame_{fi:05d}.npy")
    ids_path = os.path.join(MASKS_DIR, f"obj_ids_{fi:05d}.npy")
    if not os.path.exists(mask_path):
        return None, None
    masks_np = np.load(mask_path)
    obj_ids = np.load(ids_path) if os.path.exists(ids_path) else np.arange(masks_np.shape[0])
    if masks_np.ndim == 4:
        masks_np = masks_np[:, 0]
    # Upscale each mask
    masks_full = []
    for i in range(masks_np.shape[0]):
        m = cv2.resize(masks_np[i].astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
        masks_full.append(m)
    return masks_full, obj_ids


def get_heur12_boxes(fi):
    """Get heur12 bboxes for a frame as dict {tid: (x1,y1,x2,y2)}."""
    if fi not in grouped.groups:
        return {}
    boxes = {}
    for _, row in grouped.get_group(fi).iterrows():
        x1, y1 = int(row.x1), int(row.y1)
        boxes[int(row.tid)] = (x1, y1, x1 + int(row.w), y1 + int(row.h))
    return boxes


# ============================================================
# PASS 1: Compute per-frame IoU for each SAM3 object
# ============================================================
print("Pass 1: Computing IoU statistics...")
# {sam3_id: [(frame, best_iou, best_tid), ...]}
sam3_stats = defaultdict(list)

for fi in range(num_frames):
    masks_full, obj_ids = load_frame_masks(fi)
    if masks_full is None:
        continue
    hboxes = get_heur12_boxes(fi)

    for idx, sid in enumerate(obj_ids):
        sid = int(sid)
        mbox = get_mask_bbox(masks_full[idx])
        if mbox is None:
            sam3_stats[sid].append((fi, 0.0, -1))
            continue

        best_iou = 0.0
        best_tid = -1
        for tid, hbox in hboxes.items():
            iou = compute_iou(mbox, hbox)
            if iou > best_iou:
                best_iou = iou
                best_tid = tid

        sam3_stats[sid].append((fi, best_iou, best_tid))

# Print summary
print(f"\n{'SAM3':>5} | {'frames':>6} | {'mean_iou':>8} | {'pct>0.3':>7} | {'best_tid':>8}")
print("-" * 50)
for sid in sorted(sam3_stats.keys()):
    records = sam3_stats[sid]
    ious = [r[1] for r in records]
    tids = [r[2] for r in records if r[1] > 0.2]
    tid_mode = Counter(tids).most_common(1)[0][0] if tids else -1
    n = len(records)
    mean_iou = np.mean(ious)
    pct = sum(1 for i in ious if i > 0.3) / n * 100
    print(f"  S{sid:<3} | {n:>6} | {mean_iou:>8.3f} | {pct:>6.1f}% | T{tid_mode}")

# ============================================================
# METHOD 1: Hard mean-IoU threshold
# ============================================================
MEAN_IOU_THRESH = 0.3
method1_keep = set()
for sid, records in sam3_stats.items():
    mean_iou = np.mean([r[1] for r in records])
    if mean_iou >= MEAN_IOU_THRESH:
        method1_keep.add(sid)
print(f"\nMethod 1 (mean IoU >= {MEAN_IOU_THRESH}): keep {sorted(method1_keep)}")

# ============================================================
# METHOD 2: Per-frame IoU gate
# ============================================================
FRAME_IOU_THRESH = 0.2
# Precompute which (frame, sid) pairs pass
method2_keep = set()  # (frame, sid)
for sid, records in sam3_stats.items():
    for fi, iou, tid in records:
        if iou >= FRAME_IOU_THRESH:
            method2_keep.add((fi, sid))
n_total = sum(len(v) for v in sam3_stats.values())
print(f"Method 2 (per-frame IoU >= {FRAME_IOU_THRESH}): {len(method2_keep)}/{n_total} frame-obj pairs kept")

# ============================================================
# METHOD 3: Track assignment + dedup (best SAM3 per heur12 tid)
# ============================================================
# Assign each SAM3 track to its best heur12 tid
sam3_to_tid = {}
for sid, records in sam3_stats.items():
    tids = [r[2] for r in records if r[1] > 0.2]
    if not tids:
        continue
    tid_mode = Counter(tids).most_common(1)[0][0]
    mean_iou = np.mean([r[1] for r in records])
    pct_good = sum(1 for r in records if r[1] > 0.3) / len(records)
    if pct_good >= 0.5:  # must match >50% of frames
        sam3_to_tid[sid] = (tid_mode, mean_iou)

# For each heur12 tid, pick the best SAM3 track
tid_to_best_sam3 = {}
for sid, (tid, mean_iou) in sam3_to_tid.items():
    if tid not in tid_to_best_sam3 or mean_iou > tid_to_best_sam3[tid][1]:
        tid_to_best_sam3[tid] = (sid, mean_iou)

method3_keep = {sid for sid, _ in tid_to_best_sam3.values()}
method3_relabel = {sid: tid for tid, (sid, _) in tid_to_best_sam3.items()}
print(f"Method 3 (assign+dedup): keep {sorted(method3_keep)}")
print(f"  Relabeling: " + ", ".join(f"S{s}->T{t}" for s, t in sorted(method3_relabel.items())))


# ============================================================
# RENDERING
# ============================================================
def add_frame_number(frame, fi):
    cv2.putText(frame, f"Frame {fi}", (W - 200, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"Frame {fi}", (W - 200, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)


def render_video(out_path, keep_fn, label_fn, title):
    """Render filtered video.
    keep_fn(fi, sid) -> bool: whether to include this mask
    label_fn(sid) -> str: label text
    """
    print(f"\nRendering {title} -> {os.path.basename(out_path)}")
    cap = cv2.VideoCapture(VIDEO)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    for fi in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        add_frame_number(frame, fi)

        masks_full, obj_ids = load_frame_masks(fi)
        if masks_full is not None:
            overlay = frame.copy()
            for idx, sid in enumerate(obj_ids):
                sid = int(sid)
                if not keep_fn(fi, sid):
                    continue
                mask = masks_full[idx].astype(bool)
                label = label_fn(sid)
                # Use tid color if relabeled, otherwise sam3 color
                if label.startswith("T"):
                    tid = int(label[1:])
                    color = tid_colors.get(tid, (200, 200, 200))
                else:
                    color = tid_colors.get(sid % 12, (200, 200, 200))

                overlay[mask] = (overlay[mask] * 0.5 + np.array(color) * 0.5).astype(np.uint8)
                contours, _ = cv2.findContours(
                    mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, contours, -1, color, 2)
                ys, xs = np.where(mask)
                if len(xs) > 0:
                    cx, cy = int(xs.mean()), int(min(ys)) - 8
                    cv2.putText(overlay, label, (cx - 15, cy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            frame = overlay

        # Draw heur12 bboxes as thin outlines
        hboxes = get_heur12_boxes(fi)
        for tid, (x1, y1, x2, y2) in hboxes.items():
            color = tid_colors.get(tid, (200, 200, 200))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

        writer.write(frame)
        if fi % 100 == 0:
            print(f"  frame {fi}/{num_frames}")

    writer.release()
    cap.release()
    print(f"  Done: {out_path}")


# Method 1
render_video(
    os.path.join(OUTPUT_DIR, "sam3_filtered_m1_mean_iou.mp4"),
    keep_fn=lambda fi, sid: sid in method1_keep,
    label_fn=lambda sid: f"S{sid}",
    title="Method 1: Mean IoU threshold",
)

# Method 2
render_video(
    os.path.join(OUTPUT_DIR, "sam3_filtered_m2_per_frame.mp4"),
    keep_fn=lambda fi, sid: (fi, sid) in method2_keep,
    label_fn=lambda sid: f"S{sid}",
    title="Method 2: Per-frame IoU gate",
)

# Method 3
render_video(
    os.path.join(OUTPUT_DIR, "sam3_filtered_m3_assigned.mp4"),
    keep_fn=lambda fi, sid: sid in method3_keep,
    label_fn=lambda sid: f"T{method3_relabel[sid]}" if sid in method3_relabel else f"S{sid}",
    title="Method 3: Track assignment + dedup",
)

print("\nAll done!")
