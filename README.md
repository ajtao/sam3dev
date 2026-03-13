# sam3dev

Companion scripts for [facebook/sam3](https://github.com/facebookresearch/sam3) — text-prompted video segmentation of volleyball players, with heur12 tracker-based filtering.

## Setup

### 1. Clone sam3

```bash
cd ~/vsdevel
git clone https://github.com/facebookresearch/sam3.git
```

### 2. Create conda environment

```bash
conda create -n sam3 python=3.11
conda activate sam3
cd ~/vsdevel/sam3
pip install -e ".[interactive-demo]"
```

**GPU compatibility**: PyTorch 2.7.0+cu126 supports RTX 4090 (sm_86/89) but NOT RTX 5090 (sm_120). Use a 4090 GPU.

### 3. Clone sam3dev

```bash
cd ~/vsdevel
git clone <this-repo-url> sam3dev
```

## Scripts

### `run_video_seg.py` — Text-prompted segmentation

Runs SAM3 with a text prompt on a video. Resizes to 720p internally (24GB VRAM limit on 4090), saves masks as `.npy` files, renders output at original resolution.

```bash
conda activate sam3
python run_video_seg.py \
  --video examples/20240507-jazzy-hedgehog_play3/second_half.mp4 \
  --output-dir examples/20240507-jazzy-hedgehog_play3/sam3_output \
  --prompt "active volleyball players" \
  --gpu 0
```

Output:
- `sam3_output/masks/frame_XXXXX.npy` — per-frame binary masks (720p)
- `sam3_output/masks/obj_ids_XXXXX.npy` — object IDs per frame
- `sam3_output/output.mp4` — visualization with colored masks

**Memory**: ~350 frames at 720p fits in 24GB. Longer videos may OOM — use shorter clips or lower resolution.

### `run_bbox_masks.py` — Bbox-prompted segmentation

Alternative: prompts SAM3 with bounding boxes from a tracker (heur12). Re-prompts every 30 frames.

Requires `predictor.model.hotstart_delay = 0` to prevent SAM3's hotstart heuristic from killing box-prompted tracks (see [sam3#193](https://github.com/facebookresearch/sam3/issues/193)).

### `filter_sam3_with_heur12.py` — Filter SAM3 masks with tracker bboxes

Core experiment: validates SAM3 text-prompted masks against heur12 tracker bounding boxes using IoU. Three filtering methods:

1. **Mean IoU threshold** (>= 0.3) — drops tracks with no heur12 overlap
2. **Per-frame IoU gate** (>= 0.2) — masks flicker off during low overlap
3. **Track assignment + dedup** (recommended) — assigns SAM3 tracks to heur12 players, picks best match, relabels to tracker IDs

```bash
# Using the included example data (after running run_video_seg.py first):
python filter_sam3_with_heur12.py \
  --example-dir examples/20240507-jazzy-hedgehog_play3

# Or with explicit paths:
python filter_sam3_with_heur12.py \
  --video path/to/video.mp4 \
  --heur12 path/to/heur12.csv \
  --masks-dir path/to/sam3_output/masks \
  --output-dir path/to/output
```

Output (in `examples/.../output/`):
- `sam3_filtered_m1_mean_iou.mp4`
- `sam3_filtered_m2_per_frame.mp4`
- `sam3_filtered_m3_assigned.mp4`

### `viz_heur12_bboxes.py` — Render heur12 bbox overlay

Renders tracker bounding boxes on video for visual comparison.

### `regen_viz.py` — Regenerate visualizations with frame numbers

Re-renders all viz videos (heur12, sam3 text-prompted, sam3 bbox-prompted) with frame number overlay.

## Reproducing the filtering experiment

### Step 1: Generate SAM3 masks

```bash
conda activate sam3
python run_video_seg.py \
  --video examples/20240507-jazzy-hedgehog_play3/second_half.mp4 \
  --output-dir examples/20240507-jazzy-hedgehog_play3/sam3_output \
  --gpu 0
```

This takes ~5 minutes on a 4090 for 350 frames at 720p.

### Step 2: Run filtering

```bash
python filter_sam3_with_heur12.py \
  --example-dir examples/20240507-jazzy-hedgehog_play3
```

This produces IoU statistics and 3 filtered videos in `examples/.../output/`.

### Expected results

SAM3 detects ~13 objects. Method 3 (track assignment + dedup) produces 11 clean tracks matching heur12 players T0-T11. S12 (spurious — referee/shadow) is dropped. S5 (duplicate of T11) is deduped in favor of S6.

See [HEUR12_GUIDED_SAM3.md](HEUR12_GUIDED_SAM3.md) for detailed results.

## Example data

`examples/20240507-jazzy-hedgehog_play3/` contains:
- `second_half.mp4` — 350-frame volleyball clip (1920x1080, 30fps)
- `second_half_heur12.csv` — heur12 tracker detections (12 players, 3844 detections)

## Docs

- [SAM3.md](SAM3.md) — General SAM3 setup and usage notes
- [HEUR12_GUIDED_SAM3.md](HEUR12_GUIDED_SAM3.md) — Detailed experiment notes for heur12-guided filtering
