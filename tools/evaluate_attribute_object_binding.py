#!/usr/bin/env python3
"""Attribute-Object Binding Diagnosis for HunyuanVideo prompt-pair outputs.

Compares paired videos (prompt A vs prompt B, where B is a small attribute edit
of A) to test whether the edit is *local* (only the target attribute changes) or
*global* (scene/layout/identity/background also change).

For each prompt pair it produces:
  - a contact sheet (row A / row B, columns = timestamps)
  - global pixel-difference metrics (MAE / MSE / PSNR, optional SSIM)
  - optional CLIP frame-embedding cosine similarity (only if a local model is
    provided; never downloads)
  - heuristic color-region fractions for the named attribute colors
  - a human-readable summary.md template with computed metrics + a failure
    checklist to fill in after looking at the sheets

This does NOT modify model code and works without CLIP/SSIM installed.

Expected input layout (one subfolder per pair, two mp4s per pair)::

    results_prompt_pair/
      pair_00/ A_red_car_drives_past_a_white_fence.mp4
               A_blue_car_drives_past_a_white_fence.mp4
      pair_01/ ...

Prompt mapping (in priority order):
  1. --config JSON entries (pair_id -> prompt_a / prompt_b); videos matched to
     prompts by filename slug, else sorted order.
  2. <root>/prompt_pairs.txt lines "prompt A || prompt B" (by pair index).
  3. derived from each filename stem ("A_red_car..." -> "a red car ...").

Usage
-----
    python tools/evaluate_attribute_object_binding.py \
        --root results_prompt_pair \
        --config tools/binding_pairs_example.json \
        --outdir binding_eval

Requires opencv-python (in requirements.txt). SSIM uses scikit-image if present;
CLIP uses transformers + a local model path via --clip-model-path or
$HUNYUAN_CLIP_MODEL_PATH.
"""

import argparse
import glob
import json
import math
import os
import re

import numpy as np


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def slugify(text):
    return "_".join(re.findall(r"[a-zA-Z0-9]+", text.lower()))


def prompt_from_filename(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    return re.sub(r"[_]+", " ", stem).strip()


def load_pairs_config(path):
    if not path or not os.path.exists(path):
        return {}
    with open(path) as f:
        data = json.load(f)
    return {entry["pair_id"]: entry for entry in data}


def load_prompt_pairs_txt(root):
    """Read <root>/prompt_pairs.txt: 'prompt A || prompt B' per line."""
    path = os.path.join(root, "prompt_pairs.txt")
    pairs = []
    if not os.path.exists(path):
        return pairs
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "||" not in line:
                continue
            a, b = line.split("||", 1)
            pairs.append((a.strip(), b.strip()))
    return pairs


# --------------------------------------------------------------------------- #
# Video frame extraction (opencv)
# --------------------------------------------------------------------------- #
def extract_frames(video_path, timestamps):
    """Return a list of RGB uint8 frames (or None) at the given timestamps."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frames = []
    for ts in timestamps:
        idx = int(round(ts * fps))
        if total > 0:
            idx = min(idx, total - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        frames.append(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if ok and frame is not None else None
        )
    cap.release()
    return frames


# --------------------------------------------------------------------------- #
# Global difference metrics
# --------------------------------------------------------------------------- #
def _resize_to(a, shape_hw):
    if a.shape[:2] == shape_hw:
        return a
    import cv2

    return cv2.resize(a, (shape_hw[1], shape_hw[0]))


def frame_diff_metrics(fa, fb):
    """MAE / MSE / PSNR (+ SSIM if scikit-image is available) between two frames."""
    fb = _resize_to(fb, fa.shape[:2])
    a = fa.astype(np.float64)
    b = fb.astype(np.float64)
    mae = float(np.mean(np.abs(a - b)))
    mse = float(np.mean((a - b) ** 2))
    psnr = float("inf") if mse == 0 else float(10.0 * math.log10((255.0**2) / mse))
    out = {"mae": mae, "mse": mse, "psnr": psnr, "ssim": None}
    try:
        from skimage.metrics import structural_similarity as ssim

        out["ssim"] = float(
            ssim(a, b, channel_axis=2, data_range=255.0)
        )
    except Exception:
        pass
    return out


# --------------------------------------------------------------------------- #
# Heuristic color-region diagnostics
# --------------------------------------------------------------------------- #
# OpenCV HSV ranges (H: 0-179). Saturated hues + low-saturation grays/white/black.
_HUE_RANGES = {
    "red": [(0, 10), (170, 179)],
    "orange": [(11, 22)],
    "yellow": [(23, 34)],
    "green": [(35, 85)],
    "blue": [(86, 130)],
    "purple": [(131, 160)],
    "pink": [(161, 169)],
}


def color_fraction(frame_rgb, color):
    """Fraction of pixels matching a named color (heuristic HSV thresholds)."""
    import cv2

    hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    color = color.lower()
    if color == "white":
        mask = (s < 40) & (v > 180)
    elif color == "black":
        mask = v < 50
    elif color in ("gray", "grey"):
        mask = (s < 40) & (v >= 60) & (v <= 180)
    elif color in _HUE_RANGES:
        sat = (s > 80) & (v > 60)
        hue = np.zeros_like(h, dtype=bool)
        for lo, hi in _HUE_RANGES[color]:
            hue |= (h >= lo) & (h <= hi)
        mask = hue & sat
    else:
        return None
    return float(np.mean(mask))


def collect_colors_from_entry(entry):
    """Gather the set of attribute color names referenced in a config entry."""
    colors = set()
    for key in ("attribute_a", "attribute_b"):
        if entry.get(key):
            colors.add(entry[key])
    for key in ("attribute_object_pairs_a", "attribute_object_pairs_b"):
        for pair in entry.get(key, []) or []:
            if pair:
                colors.add(pair[0])
    return sorted(colors)


def avg_color_fractions(frames, colors):
    """Average each color's fraction across the (non-None) frames."""
    out = {}
    valid = [f for f in frames if f is not None]
    for c in colors:
        vals = [color_fraction(f, c) for f in valid]
        vals = [x for x in vals if x is not None]
        out[c] = float(np.mean(vals)) if vals else None
    return out


# --------------------------------------------------------------------------- #
# Optional CLIP frame-embedding similarity (never downloads)
# --------------------------------------------------------------------------- #
def try_load_clip(clip_model_path):
    if not clip_model_path or not os.path.exists(clip_model_path):
        return None
    try:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        model = CLIPModel.from_pretrained(clip_model_path)
        proc = CLIPProcessor.from_pretrained(clip_model_path)
        model.eval()
        return {"model": model, "proc": proc, "torch": torch}
    except Exception as e:
        print(f"[clip] could not load from {clip_model_path}: {e}")
        return None


def clip_frame_cosine(clip, frames_a, frames_b):
    import numpy as _np

    torch = clip["torch"]
    model, proc = clip["model"], clip["proc"]
    sims = []
    with torch.no_grad():
        for fa, fb in zip(frames_a, frames_b):
            if fa is None or fb is None:
                sims.append(None)
                continue
            inputs = proc(images=[fa, fb], return_tensors="pt")
            feats = model.get_image_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            sims.append(float((feats[0] * feats[1]).sum().item()))
    valid = [s for s in sims if s is not None]
    return {"per_timestamp": sims, "mean": float(_np.mean(valid)) if valid else None}


# --------------------------------------------------------------------------- #
# Contact sheet
# --------------------------------------------------------------------------- #
def make_contact_sheet(prompt_a, prompt_b, frames_a, frames_b, timestamps, out_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ncols = len(timestamps)
    fig, axes = plt.subplots(2, ncols, figsize=(ncols * 2.6, 5.6), squeeze=False)
    for r, (label, frames) in enumerate([("A: " + prompt_a, frames_a),
                                          ("B: " + prompt_b, frames_b)]):
        for c in range(ncols):
            ax = axes[r][c]
            ax.set_xticks([])
            ax.set_yticks([])
            if frames[c] is not None:
                ax.imshow(frames[c])
            else:
                ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=9)
            if r == 0:
                ax.set_title(f"{timestamps[c]:.1f}s", fontsize=10)
            if c == 0:
                ax.set_ylabel(label, fontsize=8, rotation=90, labelpad=8)
    fig.suptitle(f"A: {prompt_a}\nB: {prompt_b}", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, pil_kwargs={"quality": 90})
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Pair discovery
# --------------------------------------------------------------------------- #
def discover_pairs(root, config, txt_pairs):
    """Yield dicts with pair_id, dir, video_a/b and prompt_a/b."""
    pair_dirs = sorted(
        d for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d)
        and os.path.basename(d).startswith("pair_")
    )
    pairs = []
    for i, pdir in enumerate(pair_dirs):
        pid = os.path.basename(pdir)
        videos = sorted(glob.glob(os.path.join(pdir, "*.mp4")))
        if len(videos) < 2:
            print(f"[warn] {pid}: found {len(videos)} mp4 (need 2), skipping")
            continue

        entry = config.get(pid, {})
        prompt_a = entry.get("prompt_a")
        prompt_b = entry.get("prompt_b")
        if (prompt_a is None or prompt_b is None) and i < len(txt_pairs):
            prompt_a, prompt_b = txt_pairs[i]
        if prompt_a is None or prompt_b is None:
            prompt_a = prompt_from_filename(videos[0])
            prompt_b = prompt_from_filename(videos[1])

        # Match videos to prompts by filename slug when possible.
        va = _match_video(videos, prompt_a) or videos[0]
        vb = _match_video(videos, prompt_b) or (
            videos[1] if videos[1] != va else videos[0]
        )
        pairs.append(
            {
                "pair_id": pid,
                "dir": pdir,
                "video_a": va,
                "video_b": vb,
                "prompt_a": prompt_a,
                "prompt_b": prompt_b,
                "entry": entry,
            }
        )
    return pairs


def _match_video(videos, prompt):
    target = slugify(prompt)
    for v in videos:
        stem = slugify(os.path.splitext(os.path.basename(v))[0])
        if stem == target or target in stem or stem in target:
            return v
    return None


# --------------------------------------------------------------------------- #
# Summary report
# --------------------------------------------------------------------------- #
FAILURE_TYPES = [
    "attribute not changed",
    "wrong attribute-object binding",
    "object identity drift",
    "background/layout drift",
    "extra object insertion",
    "motion/camera drift",
]


def write_summary(out_path, results, mae_global_threshold):
    lines = ["# Attribute-Object Binding Diagnosis — summary", ""]
    lines.append(
        "Auto-generated metrics below; edit the checklist after viewing the "
        "contact sheets. A high mean global MAE suggests the edit is *global* "
        "(scene changed beyond the target attribute)."
    )
    lines.append("")
    for r in results:
        g = r["global_diff"]["mean"]
        local_hint = (
            "GLOBAL (entangled)" if g["mae"] >= mae_global_threshold else "local-ish"
        )
        lines.append(f"## {r['pair_id']}")
        lines.append(f"- prompt A: `{r['prompt_a']}`")
        lines.append(f"- prompt B: `{r['prompt_b']}`")
        lines.append(f"- contact sheet: `{r['contact_sheet']}`")
        ssim = g.get("ssim")
        lines.append(
            f"- mean global diff: MAE={g['mae']:.2f}  MSE={g['mse']:.1f}  "
            f"PSNR={g['psnr']:.2f} dB"
            + (f"  SSIM={ssim:.3f}" if ssim is not None else "  SSIM=n/a")
        )
        clip = r.get("clip_similarity")
        if clip and clip.get("mean") is not None:
            lines.append(f"- CLIP frame cosine (A vs B): mean={clip['mean']:.3f}")
        if r.get("colors_a"):
            ca = {k: (f"{v:.3f}" if v is not None else "n/a") for k, v in r["colors_a"].items()}
            cb = {k: (f"{v:.3f}" if v is not None else "n/a") for k, v in r["colors_b"].items()}
            lines.append(f"- color fraction A: {ca}")
            lines.append(f"- color fraction B: {cb}")
        lines.append(f"- automatic locality hint: **{local_hint}** "
                     f"(mean MAE vs threshold {mae_global_threshold})")
        lines.append("")
        lines.append("Manual assessment (edit after viewing the sheet):")
        lines.append("- [ ] Target attribute changed?")
        lines.append("- [ ] Target object identity stable?")
        lines.append("- [ ] Background/layout stable?")
        lines.append("- [ ] No unexpected object appeared?")
        lines.append("- [ ] Change was LOCAL (not global)?")
        lines.append("- Key failure type (check any): "
                     + " ".join(f"[ ] {t};" for t in FAILURE_TYPES))
        lines.append("")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="results_prompt_pair",
                        help="Folder with pair_XX subfolders.")
    parser.add_argument("--outdir", default="binding_eval", help="Output directory.")
    parser.add_argument("--config", default=None,
                        help="Optional pairs config JSON (prompts + attribute colors).")
    parser.add_argument("--timestamps", default="0.0,1.3,2.7,4.0,5.2",
                        help="Comma-separated timestamps (seconds).")
    parser.add_argument("--clip-model-path",
                        default=os.environ.get("HUNYUAN_CLIP_MODEL_PATH"),
                        help="Local CLIP model dir for optional embedding similarity.")
    parser.add_argument("--mae-global-threshold", type=float, default=35.0,
                        help="Mean MAE above which the edit is flagged GLOBAL.")
    args = parser.parse_args()

    timestamps = [float(t) for t in args.timestamps.split(",") if t.strip()]
    config = load_pairs_config(args.config)
    txt_pairs = load_prompt_pairs_txt(args.root)

    sheets_dir = os.path.join(args.outdir, "contact_sheets")
    os.makedirs(sheets_dir, exist_ok=True)

    pairs = discover_pairs(args.root, config, txt_pairs)
    if not pairs:
        raise SystemExit(f"No pair_XX folders with 2 mp4s found under {args.root}")

    clip = try_load_clip(args.clip_model_path)
    if clip is None and args.clip_model_path:
        print("[clip] unavailable; skipping embedding similarity.")
    elif clip is None:
        print("[clip] no --clip-model-path given; skipping embedding similarity.")

    global_diff = {}
    embed_sim = {}
    results = []

    for p in pairs:
        print(f"[{p['pair_id']}] A={os.path.basename(p['video_a'])} "
              f"B={os.path.basename(p['video_b'])}")
        frames_a = extract_frames(p["video_a"], timestamps)
        frames_b = extract_frames(p["video_b"], timestamps)

        # 2) contact sheet
        sheet_path = os.path.join(sheets_dir, f"{p['pair_id']}_sheet.jpg")
        make_contact_sheet(p["prompt_a"], p["prompt_b"], frames_a, frames_b,
                           timestamps, sheet_path)

        # 3) global difference per timestamp + mean
        per_ts = []
        for ts, fa, fb in zip(timestamps, frames_a, frames_b):
            if fa is None or fb is None:
                continue
            m = frame_diff_metrics(fa, fb)
            m["timestamp"] = ts
            per_ts.append(m)
        mean_metrics = _mean_metrics(per_ts)
        global_diff[p["pair_id"]] = {"per_timestamp": per_ts, "mean": mean_metrics}

        # 4) optional CLIP similarity
        clip_res = clip_frame_cosine(clip, frames_a, frames_b) if clip else None
        if clip_res:
            embed_sim[p["pair_id"]] = clip_res

        # 5) color diagnostics
        colors = collect_colors_from_entry(p["entry"])
        colors_a = avg_color_fractions(frames_a, colors) if colors else None
        colors_b = avg_color_fractions(frames_b, colors) if colors else None

        results.append({
            "pair_id": p["pair_id"],
            "prompt_a": p["prompt_a"],
            "prompt_b": p["prompt_b"],
            "contact_sheet": sheet_path,
            "global_diff": global_diff[p["pair_id"]],
            "clip_similarity": clip_res,
            "colors_a": colors_a,
            "colors_b": colors_b,
        })

    # write JSON outputs
    with open(os.path.join(args.outdir, "global_difference.json"), "w") as f:
        json.dump(global_diff, f, indent=2)
    with open(os.path.join(args.outdir, "image_embedding_similarity.json"), "w") as f:
        json.dump(
            embed_sim if embed_sim else {"note": "CLIP unavailable; skipped."},
            f, indent=2,
        )
    # color diagnostics JSON (handy alongside the summary)
    with open(os.path.join(args.outdir, "color_diagnostics.json"), "w") as f:
        json.dump(
            {r["pair_id"]: {"colors_a": r["colors_a"], "colors_b": r["colors_b"]}
             for r in results},
            f, indent=2,
        )

    write_summary(os.path.join(args.outdir, "summary.md"), results,
                  args.mae_global_threshold)

    print(f"\nwrote {len(pairs)} contact sheets -> {sheets_dir}")
    print(f"wrote {os.path.join(args.outdir, 'global_difference.json')}")
    print(f"wrote {os.path.join(args.outdir, 'image_embedding_similarity.json')}")
    print(f"wrote {os.path.join(args.outdir, 'color_diagnostics.json')}")
    print(f"wrote {os.path.join(args.outdir, 'summary.md')}")


def _mean_metrics(per_ts):
    if not per_ts:
        return {"mae": 0.0, "mse": 0.0, "psnr": 0.0, "ssim": None}
    keys = ["mae", "mse", "psnr"]
    out = {k: float(np.mean([m[k] for m in per_ts if math.isfinite(m[k])] or [0.0]))
           for k in keys}
    ssims = [m["ssim"] for m in per_ts if m.get("ssim") is not None]
    out["ssim"] = float(np.mean(ssims)) if ssims else None
    return out


if __name__ == "__main__":
    main()
