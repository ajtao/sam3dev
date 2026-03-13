"""Run SAM3 video segmentation with a text prompt and save visualizations.

Usage:
  python run_video_seg.py --video VIDEO --output-dir OUTPUT_DIR [--prompt "active volleyball players"] [--gpu 0]

Requires the sam3 conda environment (see SAM3.md).
"""

import os
import sys
import argparse
import cv2
import numpy as np
import torch

parser = argparse.ArgumentParser(description="SAM3 text-prompted video segmentation")
parser.add_argument("--video", type=str, required=True, help="Input video path")
parser.add_argument("--output-dir", type=str, required=True, help="Output directory for masks and video")
parser.add_argument("--prompt", type=str, default="active volleyball players", help="Text prompt")
parser.add_argument("--gpu", type=int, default=0, help="GPU index (default: 0)")
args = parser.parse_args()

VIDEO_PATH = args.video
TEXT_PROMPT = args.prompt
OUTPUT_DIR = args.output_dir
GPU_ID = args.gpu

os.makedirs(OUTPUT_DIR, exist_ok=True)

LOG_FILE = os.path.join(OUTPUT_DIR, "log.txt")

def log(msg):
    print(msg, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")
        f.flush()

# Clear log
with open(LOG_FILE, "w") as f:
    f.write("")

log("Starting SAM3 video segmentation...")
log(f"CUDA available: {torch.cuda.is_available()}")
log(f"CUDA device count: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    log(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

from sam3.model_builder import build_sam3_video_predictor

log(f"Video: {VIDEO_PATH}")
log(f"Prompt: {TEXT_PROMPT}")
log(f"Output: {OUTPUT_DIR}")

# Resize video to fit in GPU memory (4090 has 24GB)
# SAM3 loads all frames to GPU, so we need a smaller resolution
RESIZED_VIDEO = os.path.join(OUTPUT_DIR, "cut_resized.mp4")
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# Scale down to ~720p to save GPU memory
scale = 720 / orig_height
width = int(orig_width * scale)
height = 720
log(f"Video has {num_frames} frames at {fps} fps, {orig_width}x{orig_height} -> resizing to {width}x{height}")

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer_resize = cv2.VideoWriter(RESIZED_VIDEO, fourcc, fps, (width, height))
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (width, height))
    writer_resize.write(frame)
writer_resize.release()
cap.release()
log(f"Resized video saved to {RESIZED_VIDEO}")
# Use resized video for SAM3
VIDEO_FOR_SAM = RESIZED_VIDEO

# Build predictor using a single GPU
log("Building predictor on single GPU...")
predictor = build_sam3_video_predictor(gpus_to_use=[GPU_ID])

# Enable offloading to avoid OOM
predictor.model.offload_output_to_cpu_for_eval = True
predictor.model.forward_backbone_per_frame_for_eval = True
log("Predictor built (with CPU offloading enabled for eval).")

# Start session
log("Starting session...")
response = predictor.handle_request(
    request=dict(
        type="start_session",
        resource_path=VIDEO_FOR_SAM,
    )
)
session_id = response["session_id"]
log(f"Session ID: {session_id}")

# Add text prompt on frame 0
log(f"Adding text prompt: '{TEXT_PROMPT}' on frame 0...")
response = predictor.handle_request(
    request=dict(
        type="add_prompt",
        session_id=session_id,
        frame_index=0,
        text=TEXT_PROMPT,
    )
)
out = response["outputs"]
log(f"Detection on frame 0 complete. Keys: {list(out.keys()) if isinstance(out, dict) else type(out)}")

# Propagate through video, saving masks incrementally
log("Propagating through video...")
masks_dir = os.path.join(OUTPUT_DIR, "masks")
os.makedirs(masks_dir, exist_ok=True)

frame_count = 0
for resp in predictor.handle_stream_request(
    request=dict(
        type="propagate_in_video",
        session_id=session_id,
    )
):
    fi = resp["frame_index"]
    frame_out = resp["outputs"]
    frame_count += 1

    # Save masks immediately and don't accumulate
    if frame_out:
        # Try different possible keys for masks
        masks = None
        for key in ["out_binary_masks", "masks", "out_masks"]:
            if key in frame_out:
                masks = frame_out[key]
                break

        if masks is not None:
            if isinstance(masks, torch.Tensor):
                masks_np = masks.cpu().numpy()
            else:
                masks_np = np.array(masks)
            np.save(os.path.join(masks_dir, f"frame_{fi:05d}.npy"), masks_np)

        obj_ids = None
        for key in ["out_obj_ids", "obj_ids"]:
            if key in frame_out:
                obj_ids = frame_out[key]
                break
        if obj_ids is not None:
            if isinstance(obj_ids, torch.Tensor):
                obj_ids = obj_ids.cpu().numpy()
            np.save(os.path.join(masks_dir, f"obj_ids_{fi:05d}.npy"), np.array(obj_ids))

    if fi % 50 == 0:
        log(f"  Propagated frame {fi}")

log(f"Propagated {frame_count} frames total")

# Close session
predictor.handle_request(
    request=dict(
        type="close_session",
        session_id=session_id,
    )
)
predictor.shutdown()
log("Model shut down.")

# Now render output video using original full-res frames + upscaled masks
log("Rendering output video...")
cap = cv2.VideoCapture(VIDEO_PATH)
out_video_path = os.path.join(OUTPUT_DIR, "output.mp4")
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(out_video_path, fourcc, fps, (orig_width, orig_height))

# Generate distinct colors for each object
np.random.seed(42)
colors = np.random.randint(0, 255, size=(50, 3), dtype=np.uint8)

for fi in range(num_frames):
    ret, frame = cap.read()
    if not ret:
        break

    mask_path = os.path.join(masks_dir, f"frame_{fi:05d}.npy")
    if os.path.exists(mask_path):
        masks_np = np.load(mask_path)
        # masks_np shape: (num_objects, H, W) or (num_objects, 1, H, W)
        if masks_np.ndim == 4:
            masks_np = masks_np[:, 0]  # remove singleton dim

        overlay = frame.copy()
        for obj_idx in range(masks_np.shape[0]):
            # Upscale mask to original resolution
            mask_small = masks_np[obj_idx].astype(np.uint8)
            mask = cv2.resize(mask_small, (orig_width, orig_height), interpolation=cv2.INTER_NEAREST).astype(bool)
            color = colors[obj_idx % len(colors)]
            overlay[mask] = (overlay[mask] * 0.5 + color * 0.5).astype(np.uint8)
            # Draw contour
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(overlay, contours, -1, color.tolist(), 2)
        frame = overlay

    writer.write(frame)
    if fi % 50 == 0:
        log(f"  Rendered frame {fi}")

writer.release()
cap.release()
log(f"Output video saved to {out_video_path}")
log("Done!")
