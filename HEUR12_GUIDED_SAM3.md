# Heur12-Guided SAM3 Player Segmentation

## Overview

Use SAM3's text-prompted video segmentation ("active volleyball players") to generate per-player masks, then validate/filter SAM3 outputs against heur12 tracker bounding boxes. This avoids the hotstart issues with bbox-prompted SAM3 while leveraging heur12's reliable tracking.

## Test Video

- **Video**: `20240507-jazzy-hedgehog_play3/second_half.mp4` (350 frames, 1920x1080, 30fps)
- **Heur12**: 12 tracked players (T0-T11), 3844 detections over frames 0-333
- **Frame offset**: second_half frame 0 = full play frame 250

## Pipeline

1. Run SAM3 text-prompted segmentation at 720p (memory constraint on 4090)
2. Compute per-frame IoU between SAM3 mask bounding boxes and heur12 bboxes
3. Filter SAM3 tracks using IoU statistics

## SAM3 Text-Prompted Results (Unfiltered)

SAM3 detected 13 objects (S0-S13). Most map cleanly to heur12 players:

| SAM3 | Frames | Mean IoU | % > 0.3 | Best heur12 |
|------|--------|----------|---------|-------------|
| S0   | 350    | 0.786    | 94.6%   | T9          |
| S1   | 349    | 0.792    | 93.7%   | T10         |
| S2   | 350    | 0.758    | 94.0%   | T7          |
| S3   | 350    | 0.814    | 95.4%   | T6          |
| S4   | 350    | 0.849    | 95.4%   | T3          |
| S5   | 298    | 0.485    | 64.8%   | T11 (dup)   |
| S6   | 350    | 0.709    | 92.9%   | T11         |
| S7   | 348    | 0.576    | 76.7%   | T0          |
| S8   | 350    | 0.806    | 95.4%   | T2          |
| S9   | 350    | 0.758    | 93.4%   | T5          |
| S11  | 303    | 0.631    | 80.5%   | T4          |
| S12  | 279    | 0.021    | 0.0%    | none        |
| S13  | 234    | 0.615    | 77.8%   | T8          |

- **S12** is spurious (referee/shadow/ball) — zero overlap with any heur12 bbox
- **S5 and S6** both track T11 — S6 is better (0.709 vs 0.485 mean IoU)
- **S7, S11, S13** have moderate IoU — SAM3 masks may extend beyond heur12 bbox or player is partially occluded

## Filtering Methods

### Method 1: Mean IoU threshold (≥ 0.3)
- Drops only S12
- Keeps S5 (duplicate of T11) — doesn't handle dedup
- Output: `sam3_filtered_m1_mean_iou.mp4`

### Method 2: Per-frame IoU gate (≥ 0.2)
- S12 disappears entirely
- Other tracks flicker off when they lose heur12 overlap (e.g. during occlusion)
- 3603/4261 frame-object pairs kept
- Output: `sam3_filtered_m2_per_frame.mp4`

### Method 3: Track assignment + dedup (recommended)
- Assigns each SAM3 track to best-matching heur12 tid
- Requires ≥50% frames with IoU > 0.3 to keep
- Picks best SAM3 track per heur12 player (dedup)
- Relabels SAM3 IDs to heur12 T0-T11
- Drops S12 (no match) and S5 (duplicate, S6 wins for T11)
- Result: 11 clean tracks with heur12-consistent IDs and colors
- Output: `sam3_filtered_m3_assigned.mp4`

## Bbox-Prompted Approach (Alternative)

Also tested prompting SAM3 directly with heur12 bboxes (`sam3_bbox_output/`).
Requires `predictor.model.hotstart_delay = 0` to prevent SAM3's hotstart heuristic
from killing box-prompted tracks (see github.com/facebookresearch/sam3/issues/193).
Re-prompted every 30 frames. Works but text-prompted + filtering is cleaner.

## Output Files

All in `/mnt/h/output/unified_eval/dense_tracks_skill_sideoccl/20240507-jazzy-hedgehog_play3/`:

- `second_half_heur12.csv` — heur12 detections sliced to second_half frames
- `second_half_heur12_viz.mp4` — heur12 bboxes overlaid on video
- `sam3_output/` — SAM3 text-prompted masks and viz (unfiltered)
- `sam3_bbox_output/` — SAM3 bbox-prompted masks and viz
- `sam3_filtered_m1_mean_iou.mp4` — Method 1 filtered
- `sam3_filtered_m2_per_frame.mp4` — Method 2 filtered
- `sam3_filtered_m3_assigned.mp4` — Method 3 filtered (recommended)

## Scripts

All in `/home/atao/vsdevel/sam3/`:

- `run_video_seg.py` — SAM3 text-prompted segmentation (720p, template for new videos)
- `run_bbox_masks.py` — SAM3 bbox-prompted segmentation with hotstart fix
- `viz_heur12_bboxes.py` — Render heur12 bbox overlay video
- `filter_sam3_with_heur12.py` — IoU-based filtering (3 methods)
- `regen_viz.py` — Regenerate viz videos with frame numbers

## Notes

- SAM3 conda env only works on RTX 4090 (PyTorch 2.7.0+cu126 doesn't support 5090 sm_120)
- 720p resize needed to fit 350 frames in 24GB VRAM
- bf16 autocast doesn't help — memory bottleneck is tracker state, not model weights
- See `SAM3.md` for general SAM3 setup docs
