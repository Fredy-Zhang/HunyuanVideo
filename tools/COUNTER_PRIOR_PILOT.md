# Counter-prior binding pilot

This pilot reproduces the three prompt pairs in `findings.pptx` without
cherry-picking. Six fixed, seed-matched generations per prompt produce 36 videos.
The car pair is an edit-instability control; only duck and kite are used by the
counter-prior continuation gate.

## 1. Validate the plan without a GPU

```bash
python tools/run_counter_prior_pilot.py --dry-run
```

This writes `generation_manifest.csv` and `protocol.json` under
`results/counter_prior_pilot/` without importing or loading model weights.

## 2. Generate the videos

```bash
python tools/run_counter_prior_pilot.py \
  --model-base ckpts \
  --dit-weight ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt \
  --video-size 544 960 \
  --video-length 129 \
  --infer-steps 50 \
  --embedded-cfg-scale 6.0 \
  --flow-shift 7.0 \
  --flow-reverse \
  --use-cpu-offload
```

The model is loaded once. Runs already present at their deterministic output
paths are skipped by default, so rerunning resumes an interrupted pilot. Pass
`--no-resume` only when intentional regeneration is required.

Model inference arguments not owned by the pilot runner are forwarded to the
standard HunyuanVideo argument parser. Use `--pilot-seeds` to override the six
preregistered seeds and `--pilot-output-root` to change the output directory.

## 3. Build review artifacts

```bash
python tools/summarize_counter_prior_pilot.py
```

Fill the boolean fields in `binding_annotations.csv` using `true` or `false`,
then rerun the summarizer. It preserves existing annotations, rebuilds the
seed-matched contact sheets, and updates `pilot_summary.json`.

The continuation gate passes only if both the duck and kite counter-prior
conditions have at least two more binding/identity failures than their matched
in-prior conditions. Scene change and temporal drift are recorded separately.
