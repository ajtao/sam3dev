# SAM3 Video Segmentation — Local Setup

## Code Location

- **Repo**: `/home/atao/vsdevel/sam3/` (cloned from https://github.com/facebookresearch/sam3)
- **Run script**: `/home/atao/vsdevel/sam3/run_video_seg.py`
- **Model checkpoints**: cached via HuggingFace (~3.3 GB, auto-downloaded on first run)

## Environment

- **Conda env**: `sam3`
- **Python**: `/home/atao/.miniconda/envs/sam3/bin/python`
- **PyTorch**: 2.7.0+cu126

## GPU Compatibility

The installed PyTorch only supports **RTX 4090** (sm_86/sm_89), **not RTX 5090** (sm_120).
Use `CUDA_VISIBLE_DEVICES` to select a 4090 (GPUs 0 or 1 on this system).

## Memory Constraints

SAM3 loads all video frames to GPU and accumulates per-frame tracker state. Memory usage grows linearly with frame count:

| Resolution | ~Frames that fit in 24GB (4090) |
|------------|-------------------------------|
| 1920x1080  | ~250 frames (OOMs beyond)     |
| 1280x720   | ~350 frames                   |
| 854x480    | ~600+ frames                  |

The run script auto-resizes to 720p. For longer videos, reduce resolution further.
Output video is always rendered at original resolution (masks are upscaled).

## Usage

### Quick run (edit paths in script)

The script `run_video_seg.py` has hardcoded paths at the top. To run on a new video,
the easiest approach is to create a copy with updated paths:

```bash
# Create a script for your video by replacing the path
sed 's|20240201-jazzy-hedgehog-play3|YOUR_VIDEO_DIR|g' run_video_seg.py > run_my_video.py

# If the video filename isn't cut.mp4, also replace that:
# sed -i 's|/cut.mp4|/your_video.mp4|g' run_my_video.py

# Run on a 4090 GPU
cd /home/atao/vsdevel/sam3
CUDA_VISIBLE_DEVICES=0 /home/atao/.miniconda/envs/sam3/bin/python -u run_my_video.py
```

Or using conda run:
```bash
cd /home/atao/vsdevel/sam3
CUDA_VISIBLE_DEVICES=0 conda run -n sam3 python -u run_my_video.py
```

### What the script does

1. Resizes video to 720p (saves to `sam3_output/cut_resized.mp4`)
2. Builds SAM3 predictor with CPU offloading enabled
3. Detects objects matching text prompt ("active volleyball players") on frame 0
4. Propagates segmentation masks through all frames
5. Saves per-frame masks as `.npy` files in `sam3_output/masks/`
6. Renders output video with colored mask overlays at original resolution

### Output

All output goes to `<video_dir>/sam3_output/`:

- `output.mp4` — video with colored segmentation overlays
- `masks/frame_NNNNN.npy` — per-frame binary masks, shape `(num_objects, H, W)`
- `masks/obj_ids_NNNNN.npy` — object IDs for each mask
- `log.txt` — progress log
- `cut_resized.mp4` — resized input video (intermediate)

### Changing the text prompt

Edit the `TEXT_PROMPT` variable near the top of the script:
```python
TEXT_PROMPT = "active volleyball players"
```

### Running in background

```bash
CUDA_VISIBLE_DEVICES=0 /home/atao/.miniconda/envs/sam3/bin/python -u run_my_video.py &
# Monitor progress:
tail -f <video_dir>/sam3_output/log.txt
```

## bf16 / fp16 Notes

Tested `torch.autocast("cuda", dtype=torch.bfloat16)` — it does NOT meaningfully reduce
memory because the bottleneck is accumulated tracker state (per-frame feature banks),
not model weights or compute intermediates. Resizing the input video is the effective
way to reduce memory usage.
