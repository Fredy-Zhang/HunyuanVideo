#!/usr/bin/env python3
"""Run the seed-matched counter-prior binding pilot from findings.pptx.

The HunyuanVideo model is loaded once, then reused for every prompt/seed run.
All planned runs are written to a manifest before generation starts. Outputs
use deterministic names so interrupted experiments can resume safely.

Examples:
    python tools/run_counter_prior_pilot.py --dry-run

    python tools/run_counter_prior_pilot.py \
      --model-base ckpts \
      --dit-weight ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt \
      --video-size 544 960 --video-length 129 --infer-steps 50 \
      --embedded-cfg-scale 6.0 --flow-shift 7.0 --flow-reverse \
      --use-cpu-offload
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_SEEDS = [42, 123, 456, 789, 1024, 2026]
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_CONFIG = SCRIPT_DIR / "counter_prior_pilot_prompts.json"

# Support the documented `python tools/run_counter_prior_pilot.py` invocation.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MANIFEST_FIELDS = [
    "run_id",
    "pair_id",
    "study_role",
    "prompt_id",
    "condition",
    "prompt",
    "target_attribute",
    "target_object",
    "secondary_attribute",
    "secondary_object",
    "seed",
    "status",
    "output_video",
    "started_at_utc",
    "completed_at_utc",
    "error",
    "inference_params_json",
]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def is_rank_zero():
    return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))) == 0


def load_and_validate_config(path):
    with open(path, encoding="utf-8") as f:
        config = json.load(f)
    pairs = config.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("Pilot config must contain a non-empty 'pairs' list.")

    pair_ids = set()
    prompt_ids = set()
    required = {
        "prompt_id",
        "condition",
        "text",
        "target_attribute",
        "target_object",
        "secondary_attribute",
        "secondary_object",
    }
    for pair in pairs:
        pair_id = pair.get("pair_id")
        if not pair_id or pair_id in pair_ids:
            raise ValueError(f"Invalid or duplicate pair_id: {pair_id!r}")
        pair_ids.add(pair_id)
        prompts = pair.get("prompts")
        if not isinstance(prompts, list) or len(prompts) != 2:
            raise ValueError(f"Pair {pair_id!r} must contain exactly two prompts.")
        for prompt in prompts:
            missing = sorted(required - set(prompt))
            if missing:
                raise ValueError(f"Prompt in {pair_id!r} is missing: {missing}")
            prompt_id = prompt["prompt_id"]
            if prompt_id in prompt_ids:
                raise ValueError(f"Duplicate prompt_id: {prompt_id!r}")
            prompt_ids.add(prompt_id)
    return config


def inference_parameters(model_args):
    params = {
        "model": model_args.model,
        "model_base": model_args.model_base,
        "dit_weight": model_args.dit_weight,
        "video_size": list(model_args.video_size),
        "video_length": model_args.video_length,
        "infer_steps": model_args.infer_steps,
        "cfg_scale": model_args.cfg_scale,
        "embedded_cfg_scale": model_args.embedded_cfg_scale,
        "flow_shift": model_args.flow_shift,
        "flow_reverse": model_args.flow_reverse,
        "flow_solver": model_args.flow_solver,
        "precision": model_args.precision,
        "use_fp8": model_args.use_fp8,
        "use_cpu_offload": model_args.use_cpu_offload,
        "prompt_template_video": model_args.prompt_template_video,
        "reproduce": True,
    }
    if getattr(model_args, "unparsed_model_args", None):
        params["unparsed_model_args"] = list(model_args.unparsed_model_args)
    return params


def build_run_plan(config, seeds, output_root, params):
    runs = []
    params_json = json.dumps(params, sort_keys=True)
    for pair in config["pairs"]:
        for seed in seeds:
            for prompt in pair["prompts"]:
                run_id = f"{pair['pair_id']}__{prompt['prompt_id']}__seed{seed}"
                output_video = output_root / "videos" / pair["pair_id"] / f"{run_id}.mp4"
                runs.append(
                    {
                        "run_id": run_id,
                        "pair_id": pair["pair_id"],
                        "study_role": pair.get("study_role", ""),
                        "prompt_id": prompt["prompt_id"],
                        "condition": prompt["condition"],
                        "prompt": prompt["text"],
                        "target_attribute": prompt["target_attribute"],
                        "target_object": prompt["target_object"],
                        "secondary_attribute": prompt["secondary_attribute"],
                        "secondary_object": prompt["secondary_object"],
                        "seed": int(seed),
                        "status": "planned",
                        "output_video": str(output_video),
                        "started_at_utc": "",
                        "completed_at_utc": "",
                        "error": "",
                        "inference_params_json": params_json,
                    }
                )
    return runs


def write_manifest(path, runs):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(runs)
    temporary.replace(path)


def save_protocol(path, config_path, seeds, params, run_count):
    payload = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "config_path": str(Path(config_path).resolve()),
        "seeds": list(seeds),
        "planned_run_count": run_count,
        "inference_parameters": params,
        "cherry_picking_policy": "All planned runs must be retained and reported.",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def parse_cli(argv=None):
    parser = argparse.ArgumentParser(
        description="Seed-matched counter-prior binding pilot",
        add_help=True,
    )
    parser.add_argument("--pilot-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--pilot-output-root", type=Path, default=Path("results/counter_prior_pilot")
    )
    parser.add_argument("--pilot-seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and write the plan without loading the model.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Regenerate outputs even when their deterministic paths already exist.",
    )
    pilot_args, model_argv = parser.parse_known_args(argv)
    if len(set(pilot_args.pilot_seeds)) != len(pilot_args.pilot_seeds):
        parser.error("--pilot-seeds must not contain duplicates")
    if not pilot_args.pilot_seeds:
        parser.error("At least one pilot seed is required")
    return pilot_args, model_argv


def parse_dry_run_model_args(model_argv):
    """Parse the recorded inference subset without importing torch.

    Full validation still happens through ``hyvideo.config.parse_args`` on an
    actual generation run. This lightweight path keeps planning usable on a
    laptop that does not have the GPU environment installed.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model", default="HYVideo-T/2-cfgdistill")
    parser.add_argument("--model-base", default="ckpts")
    parser.add_argument(
        "--dit-weight",
        default="ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt",
    )
    parser.add_argument("--video-size", nargs="+", type=int, default=[720, 1280])
    parser.add_argument("--video-length", type=int, default=129)
    parser.add_argument("--infer-steps", type=int, default=50)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--embedded-cfg-scale", type=float, default=6.0)
    parser.add_argument("--flow-shift", type=float, default=7.0)
    parser.add_argument("--flow-reverse", action="store_true")
    parser.add_argument("--flow-solver", default="euler")
    parser.add_argument("--precision", default="bf16")
    parser.add_argument("--use-fp8", action="store_true")
    parser.add_argument("--use-cpu-offload", action="store_true")
    parser.add_argument("--prompt-template-video", default="dit-llm-encode-video")
    model_args, unparsed = parser.parse_known_args(model_argv)
    model_args.reproduce = True
    model_args.unparsed_model_args = unparsed
    return model_args


def main(argv=None):
    pilot_args, model_argv = parse_cli(argv)

    # Importing the model stack is intentionally skipped for --dry-run so plan
    # validation also works on CPU-only machines without torch installed.
    if pilot_args.dry_run:
        model_args = parse_dry_run_model_args(model_argv)
    else:
        from hyvideo.config import parse_args

        model_args = parse_args(namespace=model_argv)
    model_args.reproduce = True
    config = load_and_validate_config(pilot_args.pilot_config)
    output_root = pilot_args.pilot_output_root.resolve()
    params = inference_parameters(model_args)
    runs = build_run_plan(config, pilot_args.pilot_seeds, output_root, params)
    manifest_path = output_root / "generation_manifest.csv"

    if is_rank_zero():
        write_manifest(manifest_path, runs)
        save_protocol(
            output_root / "protocol.json",
            pilot_args.pilot_config,
            pilot_args.pilot_seeds,
            params,
            len(runs),
        )
        print(f"Planned {len(runs)} runs; manifest: {manifest_path}")
    if pilot_args.dry_run:
        return 0

    from hyvideo.inference import HunyuanVideoSampler
    from hyvideo.utils.file_utils import save_videos_grid

    models_root = Path(model_args.model_base)
    if not models_root.exists():
        raise ValueError(f"model base does not exist: {models_root}")
    sampler = HunyuanVideoSampler.from_pretrained(models_root, args=model_args)
    model_args = sampler.args

    for run in runs:
        output_video = Path(run["output_video"])
        if output_video.exists() and not pilot_args.no_resume:
            run["status"] = "completed"
            run["completed_at_utc"] = utc_now()
            if is_rank_zero():
                write_manifest(manifest_path, runs)
            continue

        run["status"] = "running"
        run["started_at_utc"] = utc_now()
        if is_rank_zero():
            write_manifest(manifest_path, runs)
        try:
            outputs = sampler.predict(
                prompt=run["prompt"],
                height=model_args.video_size[0],
                width=model_args.video_size[1],
                video_length=model_args.video_length,
                seed=int(run["seed"]),
                negative_prompt=model_args.neg_prompt,
                infer_steps=model_args.infer_steps,
                guidance_scale=model_args.cfg_scale,
                num_videos_per_prompt=1,
                flow_shift=model_args.flow_shift,
                batch_size=1,
                embedded_guidance_scale=model_args.embedded_cfg_scale,
            )
            if is_rank_zero():
                output_video.parent.mkdir(parents=True, exist_ok=True)
                sample = outputs["samples"][0].unsqueeze(0)
                save_videos_grid(sample, str(output_video), fps=24)
                run["status"] = "completed"
                run["completed_at_utc"] = utc_now()
                write_manifest(manifest_path, runs)
                print(f"completed {run['run_id']} -> {output_video}")
        except Exception as exc:
            run["status"] = "failed"
            run["error"] = f"{type(exc).__name__}: {exc}"
            run["completed_at_utc"] = utc_now()
            if is_rank_zero():
                write_manifest(manifest_path, runs)
            raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
