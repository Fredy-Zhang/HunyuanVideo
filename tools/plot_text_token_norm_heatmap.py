#!/usr/bin/env python3
"""Plot text-token norm heatmaps from the HunyuanVideo debug JSONL.

This is the companion to the env-gated debug logger in
``hyvideo/modules/models.py`` (enabled with ``HUNYUAN_DEBUG_TEXT_NORM=1``).
It reads the per-step JSONL and renders three heatmaps (denoising step on the
y-axis, text-token index on the x-axis):

  - text_norm_before_dual.png        per-token L2 norm before the dual-stream blocks
  - text_norm_after_dual.png         per-token L2 norm after  the dual-stream blocks
  - text_norm_after_before_ratio.png after / before ratio

Usage
-----
1) Generate the log while sampling::

     HUNYUAN_DEBUG_TEXT_NORM=1 \
     HUNYUAN_DEBUG_TEXT_NORM_PATH=/path/to/text_norm_debug.jsonl \
     python sample_video.py --video-size 720 1280 --video-length 129 \
         --infer-steps 50 --prompt "A red car drives past a white fence" \
         --flow-reverse --save-path ./results

2) Render the heatmaps::

     python tools/plot_text_token_norm_heatmap.py \
         --input /path/to/text_norm_debug.jsonl \
         --outdir /path/to/heatmaps

Only depends on numpy + matplotlib. Easy to remove together with the debug logger.
"""

import argparse
import json
import os

import numpy as np

import matplotlib

matplotlib.use("Agg")  # headless / no display
import matplotlib.pyplot as plt


def load_jsonl(path):
    """Read text_token_norm records and stack them into [num_steps, txt_seq_len] arrays."""
    before_rows, after_rows, ratio_rows, steps = [], [], [], []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("debug_stage") != "text_token_norm":
                continue
            before_rows.append(rec["txt_before_dual_norm"])
            after_rows.append(rec["txt_after_dual_norm"])
            # ratio may be absent in older logs; recompute if needed.
            if "txt_after_before_ratio" in rec:
                ratio_rows.append(rec["txt_after_before_ratio"])
            else:
                b = np.asarray(rec["txt_before_dual_norm"], dtype=np.float64)
                a = np.asarray(rec["txt_after_dual_norm"], dtype=np.float64)
                ratio_rows.append((a / np.clip(b, 1e-8, None)).tolist())
            steps.append(rec.get("step_idx", len(steps)))

    if not before_rows:
        raise SystemExit(f"No 'text_token_norm' records found in {path}")

    before = np.asarray(before_rows, dtype=np.float64)
    after = np.asarray(after_rows, dtype=np.float64)
    ratio = np.asarray(ratio_rows, dtype=np.float64)
    return before, after, ratio, np.asarray(steps)


def load_token_texts(input_path, max_labels):
    """Load the optional <stem>.tokens.json sidecar (prompt token strings)."""
    stem = input_path[:-6] if input_path.endswith(".jsonl") else input_path
    sidecar = f"{stem}.tokens.json"
    if not os.path.exists(sidecar):
        return None
    with open(sidecar, "r") as f:
        rec = json.load(f)
    token_texts = rec.get("token_texts")
    if not token_texts:
        return None
    print(f"loaded {len(token_texts)} token labels from {sidecar}")
    # Only label the first `max_labels` tokens (the rest are usually padding).
    return token_texts[:max_labels]


def save_heatmap(data, title, out_path, token_texts=None):
    """Render a single [num_steps, txt_seq_len] array as a heatmap PNG."""
    num_steps, seq_len = data.shape
    fig, ax = plt.subplots(figsize=(14, max(3, num_steps * 0.12 + 2)))
    im = ax.imshow(
        data,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=[0, seq_len, 0, num_steps],
    )
    ax.set_xlabel("text token index")
    ax.set_ylabel("denoising step")
    ax.set_title(title)
    if token_texts:
        # Label each of the first N token columns with its decoded string.
        n = min(len(token_texts), seq_len)
        ax.set_xticks(np.arange(n) + 0.5)
        ax.set_xticklabels(token_texts[:n], rotation=90, fontsize=6)
    fig.colorbar(im, ax=ax, label="L2 norm" if "ratio" not in title else "ratio")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", required=True, help="Path to the text_norm_debug.jsonl file."
    )
    parser.add_argument(
        "--outdir", required=True, help="Directory to write the heatmap PNGs."
    )
    parser.add_argument(
        "--max-token-labels",
        type=int,
        default=40,
        help="Max number of leading token columns to label with decoded strings "
        "(if a <stem>.tokens.json sidecar exists). The rest are usually padding.",
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    before, after, ratio, _steps = load_jsonl(args.input)
    token_texts = load_token_texts(args.input, args.max_token_labels)
    print(
        f"loaded {before.shape[0]} steps x {before.shape[1]} tokens from {args.input}"
    )

    save_heatmap(
        before,
        "Text token L2 norm — before dual-stream",
        os.path.join(args.outdir, "text_norm_before_dual.png"),
        token_texts=token_texts,
    )
    save_heatmap(
        after,
        "Text token L2 norm — after dual-stream (before single-stream)",
        os.path.join(args.outdir, "text_norm_after_dual.png"),
        token_texts=token_texts,
    )
    save_heatmap(
        ratio,
        "Text token norm ratio — after / before",
        os.path.join(args.outdir, "text_norm_after_before_ratio.png"),
        token_texts=token_texts,
    )


if __name__ == "__main__":
    main()
