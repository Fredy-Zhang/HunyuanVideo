#!/usr/bin/env python3
"""[debug-only] Contact sheet comparing semantic-anchor-expansion videos.

Reads one mp4 from each mode folder under --root, extracts frames at fixed
timestamps, and assembles a contact sheet (rows = modes, columns = timestamps).

Expected layout (one mp4 per folder; the first .mp4 found is used)::

    results_anchor_expansion/none/*.mp4
    results_anchor_expansion/duplicate_before_dual_normmatch_a1.0/*.mp4
    results_anchor_expansion/duplicate_before_dual_normmatch_a0.5/*.mp4
    results_anchor_expansion/duplicate_before_dual_normmatch_a0.1/*.mp4
    results_anchor_expansion/duplicate_after_dual_a1.0/*.mp4

Usage
-----
    python tools/compare_anchor_expansion_videos.py \
        --root results_anchor_expansion \
        --output results_anchor_expansion/comparison_sheet.jpg

Depends on opencv-python + matplotlib (both already in requirements.txt).
"""

import argparse
import glob
import os

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
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if ok and frame is not None else None)
    cap.release()
    return frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=str,
        default="results_anchor_expansion",
        help="Folder containing one subfolder per mode.",
    )
    parser.add_argument(
        "--modes",
        type=str,
        default=(
            "none,"
            "duplicate_before_dual_normmatch_a1.0,"
            "duplicate_before_dual_normmatch_a0.5,"
            "duplicate_before_dual_normmatch_a0.1,"
            "duplicate_after_dual_a1.0"
        ),
        help="Comma-separated mode subfolder names (rows).",
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
                ax.set_ylabel(mode, fontsize=8, rotation=90, labelpad=8)

    fig.suptitle("Semantic anchor token expansion comparison", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    fig.savefig(output, dpi=150, pil_kwargs={"quality": 90})
    plt.close(fig)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
