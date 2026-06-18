# Counter-prior binding pilot

This pilot reproduces the three prompt pairs in `findings.pptx` without
cherry-picking. Six fixed, seed-matched generations per prompt produce 36 videos.
The car pair is an edit-instability control; only duck and kite are used by the
counter-prior continuation gate.

## 1. Validate the plan without a GPU

```bash
DRY_RUN=1 scripts/run_counter_prior_pilot.sh
```

This writes `generation_manifest.csv` and `protocol.json` under
`results/counter_prior_pilot/` without importing or loading model weights.

## 2. Generate the videos

```bash
scripts/run_counter_prior_pilot.sh
```

The wrapper defaults to the published 544x960, 129-frame configuration with
CPU offload. Override settings through environment variables, for example:

```bash
MODEL_BASE=/models/HunyuanVideo/ckpts \
DIT_WEIGHT=/models/HunyuanVideo/transformers/mp_rank_00_model_states_fp8.pt \
USE_FP8=1 \
PILOT_OUTPUT_ROOT=/experiments/counter_prior_pilot \
scripts/run_counter_prior_pilot.sh
```

Supported environment overrides include `PYTHON_BIN`, `MODEL_BASE`,
`DIT_WEIGHT`, `PILOT_OUTPUT_ROOT`, `PILOT_SEEDS`, `VIDEO_HEIGHT`, `VIDEO_WIDTH`,
`VIDEO_LENGTH`, `INFER_STEPS`, `EMBEDDED_CFG_SCALE`, `FLOW_SHIFT`,
`USE_CPU_OFFLOAD`, `USE_FP8`, and `DRY_RUN`. Additional command-line arguments
are forwarded to the Python runner.

The model is loaded once. Runs already present at their deterministic output
paths are skipped by default, so rerunning resumes an interrupted pilot. Pass
`--no-resume` only when intentional regeneration is required.

Model inference arguments not owned by the pilot runner are forwarded to the
standard HunyuanVideo argument parser. Use `--pilot-seeds` to override the six
preregistered seeds and `--pilot-output-root` to change the output directory.

## 3. Build review artifacts

```bash
scripts/summarize_counter_prior_pilot.sh
```

Fill the boolean fields in `binding_annotations.csv` using `true` or `false`,
then rerun the summarizer. It preserves existing annotations, rebuilds the
seed-matched contact sheets, and updates `pilot_summary.json`.

The continuation gate passes only if both the duck and kite counter-prior
conditions have at least two more binding/identity failures than their matched
in-prior conditions. Scene change and temporal drift are recorded separately.
