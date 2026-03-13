"""Use SAM3 with heur12 bounding box prompts to get per-player masks."""

import os
import cv2
import numpy as np
import pandas as pd
import torch

VIDEO = "/mnt/g/data/vball/tracking/fiverr/dense_tracks_skill_sideoccl/20240507-jazzy-hedgehog_play3/second_half.mp4"
HEUR12 = "/mnt/g/data/vball/tracking/fiverr/dense_tracks_skill_sideoccl/20240507-jazzy-hedgehog_play3/second_half_heur12.csv"
OUTPUT_DIR = "/mnt/g/data/vball/tracking/fiverr/dense_tracks_skill_sideoccl/20240507-jazzy-hedgehog_play3/sam3_bbox_output"
LOG_FILE = os.path.join(OUTPUT_DIR, "log.txt")

# How often to re-prompt SAM3 with updated bboxes (every N frames)
REPROMPT_EVERY = 30

os.makedirs(OUTPUT_DIR, exist_ok=True)

def log(msg):
    print(msg, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")
        f.flush()

with open(LOG_FILE, "w") as f:
    f.write("")

log("SAM3 bbox-prompted player segmentation")

# Load detections
df = pd.read_csv(HEUR12)
grouped = df.groupby("frame_sh")
log(f"Loaded {len(df)} detections, frames {df.frame_sh.min()}-{df.frame_sh.max()}, {df.tid.nunique()} unique tids")

# Video info
cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)
num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()
log(f"Video: {num_frames} frames, {W}x{H}, {fps:.1f} fps")

# Build SAM3 predictor
log("Building SAM3 predictor...")
from sam3.model_builder import build_sam3_video_predictor

predictor = build_sam3_video_predictor(gpus_to_use=[0])
predictor.model.offload_output_to_cpu_for_eval = True
predictor.model.forward_backbone_per_frame_for_eval = True
# Disable hotstart heuristic — it kills box-prompted tracks that lack
# text-based detections in other frames (see github.com/facebookresearch/sam3/issues/193)
predictor.model.hotstart_delay = 0
log("Predictor built (hotstart_delay=0 for bbox prompts).")

# Resize video for SAM3 (720p to fit in memory)
RESIZED_VIDEO = os.path.join(OUTPUT_DIR, "resized.mp4")
scale = 720 / H
rW, rH = int(W * scale), 720

cap = cv2.VideoCapture(VIDEO)
writer_r = cv2.VideoWriter(RESIZED_VIDEO, cv2.VideoWriter_fourcc(*"mp4v"), fps, (rW, rH))
while True:
    ret, frame = cap.read()
    if not ret:
        break
    writer_r.write(cv2.resize(frame, (rW, rH)))
writer_r.release()
cap.release()
log(f"Resized to {rW}x{rH}")

# Start session
log("Starting session...")
response = predictor.handle_request(
    request=dict(type="start_session", resource_path=RESIZED_VIDEO)
)
session_id = response["session_id"]
log(f"Session ID: {session_id}")

# Find frames where we have detections, sorted
det_frames = sorted(df.frame_sh.unique())

# Add box prompts on first detection frame for each tid
first_frame = det_frames[0]
first_dets = grouped.get_group(first_frame)
log(f"\nPrompting on frame {first_frame} with {len(first_dets)} players:")

for _, row in first_dets.iterrows():
    tid = int(row.tid)
    # Normalize bbox to [0, 1] relative to original resolution
    bbox = [row.x1 / W, row.y1 / H, row.w / W, row.h / H]
    log(f"  tid={tid}, bbox=[{bbox[0]:.3f},{bbox[1]:.3f},{bbox[2]:.3f},{bbox[3]:.3f}]")
    predictor.handle_request(
        request=dict(
            type="add_prompt",
            session_id=session_id,
            frame_index=first_frame,
            bounding_boxes=[bbox],
            bounding_box_labels=[1],  # 1 = foreground
            obj_id=tid,
        )
    )

log(f"\nAdded {len(first_dets)} player prompts.")

# Re-prompt on periodic frames to correct drift
reprompt_frames = set()
for fi in det_frames:
    if fi > first_frame and fi % REPROMPT_EVERY == 0:
        reprompt_frames.add(fi)

for fi in sorted(reprompt_frames):
    if fi not in grouped.groups:
        continue
    frame_dets = grouped.get_group(fi)
    for _, row in frame_dets.iterrows():
        tid = int(row.tid)
        bbox = [row.x1 / W, row.y1 / H, row.w / W, row.h / H]
        predictor.handle_request(
            request=dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=fi,
                bounding_boxes=[bbox],
                bounding_box_labels=[1],
                obj_id=tid,
            )
        )
    log(f"Re-prompted on frame {fi} with {len(frame_dets)} players")

# Propagate through video
log("\nPropagating masks through video...")
masks_dir = os.path.join(OUTPUT_DIR, "masks")
os.makedirs(masks_dir, exist_ok=True)

frame_count = 0
for resp in predictor.handle_stream_request(
    request=dict(type="propagate_in_video", session_id=session_id)
):
    fi = resp["frame_index"]
    frame_out = resp["outputs"]
    frame_count += 1

    if frame_out:
        for key in ["out_binary_masks", "masks", "out_masks"]:
            if key in frame_out:
                masks = frame_out[key]
                if isinstance(masks, torch.Tensor):
                    masks = masks.cpu().numpy()
                np.save(os.path.join(masks_dir, f"frame_{fi:05d}.npy"), masks)
                break

        for key in ["out_obj_ids", "obj_ids"]:
            if key in frame_out:
                obj_ids = frame_out[key]
                if isinstance(obj_ids, torch.Tensor):
                    obj_ids = obj_ids.cpu().numpy()
                np.save(os.path.join(masks_dir, f"obj_ids_{fi:05d}.npy"), np.array(obj_ids))
                break

    if fi % 50 == 0:
        log(f"  Frame {fi}")

log(f"Propagated {frame_count} frames")

# Shutdown
predictor.handle_request(dict(type="close_session", session_id=session_id))
predictor.shutdown()
log("Model shut down.")

# Render output video with per-player colored masks + bboxes
log("\nRendering output video...")
cap = cv2.VideoCapture(VIDEO)
out_video = os.path.join(OUTPUT_DIR, "output.mp4")
writer = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

# Stable colors per tid
np.random.seed(0)
tid_colors = {}
for tid in sorted(df.tid.unique()):
    tid_colors[tid] = tuple(int(c) for c in np.random.randint(60, 255, 3))

for fi in range(num_frames):
    ret, frame = cap.read()
    if not ret:
        break

    mask_path = os.path.join(masks_dir, f"frame_{fi:05d}.npy")
    ids_path = os.path.join(masks_dir, f"obj_ids_{fi:05d}.npy")

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

            # Find mask centroid for label
            ys, xs = np.where(mask)
            if len(xs) > 0:
                cx, cy = int(xs.mean()), int(min(ys)) - 8
                cv2.putText(overlay, f"T{int(oid)}", (cx - 15, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        frame = overlay

    # Also draw heur12 bboxes as thin outlines
    if fi in grouped.groups:
        rows = grouped.get_group(fi)
        for _, row in rows.iterrows():
            tid = int(row.tid)
            x1, y1 = int(row.x1), int(row.y1)
            x2, y2 = x1 + int(row.w), y1 + int(row.h)
            color = tid_colors.get(tid, (200, 200, 200))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

    writer.write(frame)
    if fi % 50 == 0:
        log(f"  Rendered frame {fi}")

writer.release()
cap.release()
log(f"Output video: {out_video}")
log("Done!")
