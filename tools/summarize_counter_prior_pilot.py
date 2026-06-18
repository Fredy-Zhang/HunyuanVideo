#!/usr/bin/env python3
"""Build contact sheets, annotation template, and pilot gate summary."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


ANNOTATION_FIELDS = [
    "run_id",
    "pair_id",
    "prompt_id",
    "condition",
    "seed",
    "output_video",
    "target_attribute_correct",
    "object_identity_preserved",
    "secondary_attribute_correct",
    "temporal_drift",
    "global_scene_change_within_pair",
    "notes",
]


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def annotation_template(manifest, existing):
    prior = {row["run_id"]: row for row in existing}
    rows = []
    for run in manifest:
        old = prior.get(run["run_id"], {})
        row = {field: "" for field in ANNOTATION_FIELDS}
        for field in ("run_id", "pair_id", "prompt_id", "condition", "seed", "output_video"):
            row[field] = run[field]
        for field in ANNOTATION_FIELDS[6:]:
            row[field] = old.get(field, "")
        rows.append(row)
    return rows


def read_even_frames(path, count=5):
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise ValueError(f"Cannot read frames from {path}")
    indices = np.linspace(0, total - 1, count).round().astype(int)
    frames = []
    for index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = cap.read()
        if not ok:
            cap.release()
            raise ValueError(f"Cannot read frame {index} from {path}")
        frames.append(frame)
    cap.release()
    return frames, indices.tolist()


def fit_frame(frame, width=288, height=180):
    import cv2
    import numpy as np

    h, w = frame.shape[:2]
    scale = min(width / w, height / h)
    resized = cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))))
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    y = (height - resized.shape[0]) // 2
    x = (width - resized.shape[1]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def labeled_row(run, frames):
    import cv2
    import numpy as np

    label_width = 300
    cell_height = 180
    label = np.full((cell_height, label_width, 3), 255, dtype=np.uint8)
    lines = [run["condition"], run["prompt"]]
    for i, line in enumerate(lines):
        cv2.putText(
            label,
            line[:42],
            (12, 45 + i * 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
    return np.hstack([label] + [fit_frame(frame) for frame in frames])


def build_contact_sheets(manifest, output_dir):
    grouped = defaultdict(list)
    for row in manifest:
        if row["status"] == "completed" and Path(row["output_video"]).exists():
            grouped[(row["pair_id"], row["seed"])].append(row)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not grouped:
        return []

    import cv2
    import numpy as np

    written = []
    for (pair_id, seed), rows in sorted(grouped.items()):
        if len(rows) != 2:
            continue
        images = []
        for row in rows:
            frames, _ = read_even_frames(row["output_video"])
            images.append(labeled_row(row, frames))
        sheet = np.vstack(images)
        path = output_dir / f"{pair_id}__seed{seed}.jpg"
        cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
        written.append(str(path))
    return written


def parse_bool(value):
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    return None


def summarize(annotations):
    by_pair_condition = defaultdict(list)
    incomplete = 0
    for row in annotations:
        values = [
            parse_bool(row["target_attribute_correct"]),
            parse_bool(row["object_identity_preserved"]),
            parse_bool(row["secondary_attribute_correct"]),
        ]
        if any(value is None for value in values):
            incomplete += 1
            continue
        failure = not all(values)
        by_pair_condition[(row["pair_id"], row["condition"])].append(failure)

    rates = {}
    for (pair_id, condition), failures in sorted(by_pair_condition.items()):
        rates.setdefault(pair_id, {})[condition] = {
            "n": len(failures),
            "failures": sum(failures),
            "failure_rate": sum(failures) / len(failures),
        }

    gate_details = {}
    gate_pass = incomplete == 0
    for pair_id in ("duck", "kite"):
        prior = rates.get(pair_id, {}).get("in_prior")
        counter = rates.get(pair_id, {}).get("counter_prior")
        passed = bool(prior and counter and counter["failures"] - prior["failures"] >= 2)
        gate_details[pair_id] = {
            "passed": passed,
            "required_failure_count_delta": 2,
            "observed_failure_count_delta": (
                counter["failures"] - prior["failures"] if prior and counter else None
            ),
        }
        gate_pass = gate_pass and passed
    return {
        "annotation_status": "complete" if incomplete == 0 else "incomplete",
        "incomplete_rows": incomplete,
        "rates": rates,
        "continuation_gate": {
            "passed": gate_pass,
            "rule": "Both duck and kite counter-prior conditions must have at least two more failures than their in-prior controls.",
            "details": gate_details,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pilot-output-root", type=Path, default=Path("results/counter_prior_pilot")
    )
    args = parser.parse_args()
    root = args.pilot_output_root.resolve()
    manifest = read_csv(root / "generation_manifest.csv")
    annotation_path = root / "binding_annotations.csv"
    existing = read_csv(annotation_path) if annotation_path.exists() else []
    annotations = annotation_template(manifest, existing)
    write_csv(annotation_path, annotations, ANNOTATION_FIELDS)
    sheets = build_contact_sheets(manifest, root / "contact_sheets")
    summary = summarize(annotations)
    summary["contact_sheets_written"] = sheets
    with open(root / "pilot_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"annotation template: {annotation_path}")
    print(f"contact sheets: {len(sheets)}")
    print(f"summary: {root / 'pilot_summary.json'}")


if __name__ == "__main__":
    main()
