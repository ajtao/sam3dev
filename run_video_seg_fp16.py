"""Run SAM3 video segmentation with fp16/bf16 — full 1080p, no resize needed."""

import os
import sys
import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")

OUTPUT_DIR = "/mnt/g/data/vball/tracking/fiverr/dense_tracks_skill_sideoccl/20240201-jazzy-hedgehog-play3/sam3_output_fp16"
LOG_FILE = os.path.join(OUTPUT_DIR, "log.txt")

os.makedirs(OUTPUT_DIR, exist_ok=True)

def log(msg):
    print(msg, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")
        f.flush()

# Clear log
with open(LOG_FILE, "w") as f:
    f.write("")

# Config
VIDEO_PATH = "/mnt/g/data/vball/tracking/fiverr/dense_tracks_skill_sideoccl/20240201-jazzy-hedgehog-play3/cut.mp4"
TEXT_PROMPT = "active volleyball players"

log("Starting SAM3 video segmentation (bf16 autocast mode)...")
log(f"CUDA available: {torch.cuda.is_available()}")
for i in range(torch.cuda.device_count()):
    log(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

from sam3.model_builder import build_sam3_video_predictor

log(f"Video: {VIDEO_PATH}")
log(f"Prompt: {TEXT_PROMPT}")
log(f"Output: {OUTPUT_DIR}")

# Get video info
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()
log(f"Video: {num_frames} frames at {fps} fps, {width}x{height} (NO resize — using bf16 autocast)")

# Build predictor
log("Building predictor on single GPU...")
predictor = build_sam3_video_predictor(gpus_to_use=[0])

# Enable offloading
predictor.model.offload_output_to_cpu_for_eval = True
predictor.model.forward_backbone_per_frame_for_eval = True

mem_after_model = torch.cuda.memory_allocated() / 1e9
log(f"GPU memory after model load (fp32): {mem_after_model:.2f} GB")
log("Predictor built. Using autocast bf16 for inference (weights stay fp32).")

# Start session with autocast — bf16 is more stable than fp16
log("Starting session...")
with torch.autocast("cuda", dtype=torch.bfloat16):
    response = predictor.handle_request(
        request=dict(
            type="start_session",
            resource_path=VIDEO_PATH,
        )
    )
session_id = response["session_id"]
mem_after_session = torch.cuda.memory_allocated() / 1e9
log(f"Session ID: {session_id}")
log(f"GPU memory after session init: {mem_after_session:.2f} GB")

# Add text prompt on frame 0
log(f"Adding text prompt: '{TEXT_PROMPT}' on frame 0...")
with torch.autocast("cuda", dtype=torch.bfloat16):
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
mem_after_detect = torch.cuda.memory_allocated() / 1e9
log(f"GPU memory after detection: {mem_after_detect:.2f} GB")

# Propagate through video, saving masks incrementally
log("Propagating through video...")
masks_dir = os.path.join(OUTPUT_DIR, "masks")
os.makedirs(masks_dir, exist_ok=True)

frame_count = 0
with torch.autocast("cuda", dtype=torch.bfloat16):
    for resp in predictor.handle_stream_request(
        request=dict(
            type="propagate_in_video",
            session_id=session_id,
        )
    ):
        fi = resp["frame_index"]
        frame_out = resp["outputs"]
        frame_count += 1

        # Save masks immediately
        if frame_out:
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
            mem = torch.cuda.memory_allocated() / 1e9
            log(f"  Propagated frame {fi} (GPU mem: {mem:.2f} GB)")

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

# Render output video
log("Rendering output video...")
cap = cv2.VideoCapture(VIDEO_PATH)
out_video_path = os.path.join(OUTPUT_DIR, "output.mp4")
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))

np.random.seed(42)
colors = np.random.randint(0, 255, size=(50, 3), dtype=np.uint8)

for fi in range(num_frames):
    ret, frame = cap.read()
    if not ret:
        break

    mask_path = os.path.join(masks_dir, f"frame_{fi:05d}.npy")
    if os.path.exists(mask_path):
        masks_np = np.load(mask_path)
        if masks_np.ndim == 4:
            masks_np = masks_np[:, 0]

        overlay = frame.copy()
        for obj_idx in range(masks_np.shape[0]):
            mask = masks_np[obj_idx].astype(bool)
            color = colors[obj_idx % len(colors)]
            overlay[mask] = (overlay[mask] * 0.5 + color * 0.5).astype(np.uint8)
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
