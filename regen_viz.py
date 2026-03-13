"""Regenerate second_half_heur12_viz.mp4 and sam3_output/output.mp4 with frame numbers."""

import cv2
import numpy as np
import pandas as pd
import os

BASE = "/mnt/h/output/unified_eval/dense_tracks_skill_sideoccl/20240507-jazzy-hedgehog_play3"
VIDEO = "/mnt/g/data/vball/tracking/fiverr/dense_tracks_skill_sideoccl/20240507-jazzy-hedgehog_play3/second_half.mp4"
HEUR12 = os.path.join(BASE, "second_half_heur12.csv")

# Stable colors per tid
np.random.seed(0)
tid_colors = {}
for tid in range(12):
    tid_colors[tid] = tuple(int(c) for c in np.random.randint(60, 255, 3))

def add_frame_number(frame, fi, W):
    cv2.putText(frame, f"Frame {fi}", (W - 200, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"Frame {fi}", (W - 200, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)

# === 1. Regenerate second_half_heur12_viz.mp4 ===
print("=== Regenerating heur12 viz ===")
df = pd.read_csv(HEUR12)
grouped = df.groupby("frame_sh")

cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

out_path = os.path.join(BASE, "second_half_heur12_viz.mp4")
writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

for fi in range(num_frames):
    ret, frame = cap.read()
    if not ret:
        break
    add_frame_number(frame, fi, W)
    if fi in grouped.groups:
        for _, row in grouped.get_group(fi).iterrows():
            tid = int(row.tid)
            x1, y1 = int(row.x1), int(row.y1)
            x2, y2 = x1 + int(row.w), y1 + int(row.h)
            color = tid_colors[tid]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"T{tid}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    writer.write(frame)
    if fi % 50 == 0:
        print(f"  heur12 frame {fi}/{num_frames}")

writer.release()
cap.release()
print(f"  Wrote {out_path}")

# === 2. Regenerate sam3_output/output.mp4 (text-prompted) ===
print("\n=== Regenerating sam3 text-prompted viz ===")
masks_dir = os.path.join(BASE, "sam3_output", "masks")
if os.path.exists(masks_dir):
    # SAM3 auto-assigned colors
    np.random.seed(42)
    sam3_colors = np.random.randint(0, 255, size=(50, 3), dtype=np.uint8)

    cap = cv2.VideoCapture(VIDEO)
    out_path2 = os.path.join(BASE, "sam3_output", "output.mp4")
    writer = cv2.VideoWriter(out_path2, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    for fi in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        add_frame_number(frame, fi, W)

        mask_path = os.path.join(masks_dir, f"frame_{fi:05d}.npy")
        ids_path = os.path.join(masks_dir, f"obj_ids_{fi:05d}.npy")
        if os.path.exists(mask_path):
            masks_np = np.load(mask_path)
            if masks_np.ndim == 4:
                masks_np = masks_np[:, 0]

            obj_ids = np.load(ids_path) if os.path.exists(ids_path) else np.arange(masks_np.shape[0])

            overlay = frame.copy()
            for idx in range(masks_np.shape[0]):
                mask_small = masks_np[idx].astype(np.uint8)
                mask = cv2.resize(mask_small, (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
                color = sam3_colors[idx % len(sam3_colors)]
                overlay[mask] = (overlay[mask] * 0.5 + color * 0.5).astype(np.uint8)
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, contours, -1, color.tolist(), 2)
                ys, xs = np.where(mask)
                if len(xs) > 0:
                    cx, cy = int(xs.mean()), int(min(ys)) - 8
                    cv2.putText(overlay, f"S{int(obj_ids[idx])}", (cx - 15, cy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color.tolist(), 2)
            frame = overlay

        writer.write(frame)
        if fi % 50 == 0:
            print(f"  sam3 frame {fi}/{num_frames}")

    writer.release()
    cap.release()
    print(f"  Wrote {out_path2}")
else:
    print("  No sam3_output/masks found, skipping")

# === 3. Regenerate sam3_bbox_output/output.mp4 ===
print("\n=== Regenerating sam3 bbox-prompted viz ===")
masks_dir_bb = os.path.join(BASE, "sam3_bbox_output", "masks")
if os.path.exists(masks_dir_bb):
    cap = cv2.VideoCapture(VIDEO)
    out_path3 = os.path.join(BASE, "sam3_bbox_output", "output.mp4")
    writer = cv2.VideoWriter(out_path3, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    for fi in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        add_frame_number(frame, fi, W)

        mask_path = os.path.join(masks_dir_bb, f"frame_{fi:05d}.npy")
        ids_path = os.path.join(masks_dir_bb, f"obj_ids_{fi:05d}.npy")
        if os.path.exists(mask_path) and os.path.exists(ids_path):
            masks_np = np.load(mask_path)
            obj_ids = np.load(ids_path)
            if masks_np.ndim == 4:
                masks_np = masks_np[:, 0]

            overlay = frame.copy()
            for idx, oid in enumerate(obj_ids):
                mask_small = masks_np[idx].astype(np.uint8)
                mask = cv2.resize(mask_small, (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
                color = tid_colors.get(int(oid), (200, 200, 200))
                overlay[mask] = (overlay[mask] * 0.5 + np.array(color) * 0.5).astype(np.uint8)
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, contours, -1, color, 2)
                ys, xs = np.where(mask)
                if len(xs) > 0:
                    cx, cy = int(xs.mean()), int(min(ys)) - 8
                    cv2.putText(overlay, f"T{int(oid)}", (cx - 15, cy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            frame = overlay

        # Also draw heur12 bboxes as thin outlines
        if fi in grouped.groups:
            for _, row in grouped.get_group(fi).iterrows():
                tid = int(row.tid)
                x1, y1 = int(row.x1), int(row.y1)
                x2, y2 = x1 + int(row.w), y1 + int(row.h)
                color = tid_colors.get(tid, (200, 200, 200))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

        writer.write(frame)
        if fi % 50 == 0:
            print(f"  bbox frame {fi}/{num_frames}")

    writer.release()
    cap.release()
    print(f"  Wrote {out_path3}")
else:
    print("  No sam3_bbox_output/masks found, skipping")

print("\nDone!")
