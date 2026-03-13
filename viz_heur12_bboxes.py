"""Slice heur12.csv to second_half frames and render bbox overlay video."""

import cv2
import numpy as np
import pandas as pd

HEUR12 = "/mnt/h/output/unified_eval/dense_tracks_skill_sideoccl/20240507-jazzy-hedgehog_play3/heur12.csv"
VIDEO = "/mnt/g/data/vball/tracking/fiverr/dense_tracks_skill_sideoccl/20240507-jazzy-hedgehog_play3/second_half.mp4"
OUT_DIR = "/mnt/g/data/vball/tracking/fiverr/dense_tracks_skill_sideoccl/20240507-jazzy-hedgehog_play3"
FRAME_OFFSET = 250  # second_half frame 0 = cut.mp4 frame 250

# Load and slice heur12 to second_half frames
df = pd.read_csv(HEUR12)
df = df[df.frame >= FRAME_OFFSET].copy()
df["frame_sh"] = df["frame"] - FRAME_OFFSET
out_csv = f"{OUT_DIR}/second_half_heur12.csv"
df.to_csv(out_csv, index=False)
print(f"Wrote {out_csv}: {len(df)} rows, frames {df.frame_sh.min()}-{df.frame_sh.max()}")

# Generate colors per tid
np.random.seed(0)
colors = {}
for tid in sorted(df.tid.unique()):
    colors[tid] = tuple(int(c) for c in np.random.randint(60, 255, 3))

# Render video
cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

out_video = f"{OUT_DIR}/second_half_heur12_viz.mp4"
writer = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Group by second_half frame for fast lookup
grouped = df.groupby("frame_sh")

for fi in range(num_frames):
    ret, frame = cap.read()
    if not ret:
        break

    if fi in grouped.groups:
        rows = grouped.get_group(fi)
        for _, row in rows.iterrows():
            tid = int(row.tid)
            x1, y1 = int(row.x1), int(row.y1)
            x2, y2 = x1 + int(row.w), y1 + int(row.h)
            color = colors[tid]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"T{tid}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    writer.write(frame)
    if fi % 50 == 0:
        print(f"  Rendered frame {fi}/{num_frames}")

writer.release()
cap.release()
print(f"Output video: {out_video}")
