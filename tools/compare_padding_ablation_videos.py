#!/usr/bin/env python3
"""[debug-only] Contact sheet comparing padding-ablation videos.

Reads one mp4 from each ablation-mode folder, extracts frames at fixed
timestamps, and assembles a single contact sheet (rows = ablation modes,
columns = timestamps) for side-by-side visual comparison.

Expected layout (one mp4 per folder; the first .mp4 found is used)::

    results_padding_ablation/none/*.mp4
    results_padding_ablation/zero_real_tokens/*.mp4
    results_padding_ablation/zero_pad_tokens/*.mp4
    results_padding_ablation/zero_all_text/*.mp4

Usage
-----
    python tools/compare_padding_ablation_videos.py \
        --root results_padding_ablation \
        --output results_padding_ablation/comparison_sheet.jpg

Depends on opencv-python + matplotlib (both already in requirements.txt).
"""

import argparse
import glob
import os

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import cv2


def find_video(folder):
    """Return the first .mp4 under `folder` (recursively), or None."""
    hits = sorted(glob.glob(os.path.join(folder, "**", "*.mp4"), recursive=True))
    return hits[0] if hits else None


def extract_frames(video_path, timestamps):
    """Extract RGB frames at the given timestamps (seconds). Missing -> None."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frames = []
    for ts in timestamps:
        frame_idx = int(round(ts * fps))
        if total > 0:
            frame_idx = min(frame_idx, total - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            frames.append(None)
        else:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=str,
        default="results_padding_ablation",
        help="Folder containing one subfolder per ablation mode.",
    )
    parser.add_argument(
        "--modes",
        type=str,
        default="none,zero_real_tokens,zero_pad_tokens,zero_all_text",
        help="Comma-separated ablation-mode subfolder names (rows).",
    )
    parser.add_argument(
        "--timestamps",
        type=str,
        default="0.0,1.3,2.7,4.0,5.2",
        help="Comma-separated timestamps in seconds (columns).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JPG path. Defaults to <root>/comparison_sheet.jpg.",
    )
    args = parser.parse_args()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    timestamps = [float(t) for t in args.timestamps.split(",") if t.strip()]
    output = args.output or os.path.join(args.root, "comparison_sheet.jpg")

    nrows, ncols = len(modes), len(timestamps)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(ncols * 2.6, nrows * 2.6), squeeze=False
    )

    for r, mode in enumerate(modes):
        folder = os.path.join(args.root, mode)
        video = find_video(folder)
        if video is None:
            print(f"[warn] no mp4 found in {folder}")
            frames = [None] * ncols
        else:
            print(f"[{mode}] {video}")
            frames = extract_frames(video, timestamps)

        for c in range(ncols):
            ax = axes[r][c]
            ax.set_xticks([])
            ax.set_yticks([])
            if frames[c] is not None:
                ax.imshow(frames[c])
            else:
                ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=9)
            if r == 0:
                ax.set_title(f"{timestamps[c]:.1f}s", fontsize=11)
            if c == 0:
                ax.set_ylabel(mode, fontsize=10, rotation=90, labelpad=8)

    fig.suptitle("Padding text-token ablation comparison", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    fig.savefig(output, dpi=150, pil_kwargs={"quality": 90})
    plt.close(fig)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
