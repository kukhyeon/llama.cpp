# TrePrefill host autotuning tools

These scripts provide a host-side, reproducible path from device traces to a
token-specific FFN partition policy:

```text
model descriptor + known-good policy template
                       │
Android profiling ──> observed clock catalog ──> FFN shard recommendation
                                                    │
                                  generated policy ──> static validation
```

The tools use only the Python standard library. Local GGUF inspection first
tries the repository's `gguf-py` reader and falls back to a metadata-only
reader when NumPy is unavailable. Policy generation preserves the template's
weight residency and operation-placement rules.

## Prerequisites

- Run commands from the repository root in WSL/Linux.
- `adb` must see the target device, and the TrePrefill package, model, scripts,
  and token datasets must already be deployed under `/data/local/tmp/llama.cpp`.
- Start from a policy that is already known to load and execute correctly for
  the model family. The generator adapts it; it does not synthesize residency
  or backend compatibility rules.
- Use a freshly built/deployed binary that records `actual_prime_khz`,
  `actual_gold_khz`, and `actual_gpu_hz` in `scheduler_trace.csv`.

## Copy-pasteable 256-token workflow

The checked-in example runs four trials: GPU indices 14 and 4, each at 2 s and
6 s query periods. Review the device paths and policy in
`examples/llama32-i256-thermal.plan.json` before starting a real run.

For a GPU+NPU-only run, use
`examples/llama32-i256-thermal-gpu-npu.plan.json`. It selects the dedicated
GPU+NPU script and seed policy and writes to a separate result root.

### 1. Create or inspect the model descriptor

Use the included Llama-3.2-3B descriptor:

```bash
python3 scripts/treprefill_autotune/autotune_policy.py inspect-model \
  scripts/treprefill_autotune/examples/llama32-3b-q8_0.model.json \
  --output autotune-work/llama32-i256/model.json
```

If the GGUF exists on the host, its stable dimensions can instead be extracted
directly (add `--sha256` only when a full multi-GB file hash is needed):

```bash
python3 scripts/treprefill_autotune/autotune_policy.py inspect-model \
  /path/to/Llama-3.2-3B-Instruct-Q8_0.gguf \
  --output autotune-work/llama32-i256/model.json
```

### 2. Review and run the profiling matrix

Listing and dry-run modes do not connect to the device:

```bash
python3 scripts/treprefill_autotune/trial_plan.py \
  scripts/treprefill_autotune/examples/llama32-i256-thermal.plan.json --list

python3 scripts/treprefill_autotune/trial_plan.py \
  scripts/treprefill_autotune/examples/llama32-i256-thermal.plan.json --dry-run
```

Run trials sequentially so experiments do not interfere thermally:

```bash
python3 scripts/treprefill_autotune/trial_plan.py \
  scripts/treprefill_autotune/examples/llama32-i256-thermal.plan.json --run
```

After an interrupted run, first confirm that its on-device process has exited;
then preserve the old attempt and continue with:

```bash
python3 scripts/treprefill_autotune/trial_plan.py \
  scripts/treprefill_autotune/examples/llama32-i256-thermal.plan.json \
  --run --resume
```

### 3. Discover clock states actually reached on the device

```bash
python3 scripts/treprefill_autotune/autotune_policy.py discover \
  autotune-runs/llama32-i256-thermal/* \
  --min-samples 30 \
  --min-queries 3 \
  --output autotune-work/llama32-i256/clock-catalog.json
```

The catalog contains observed states, not every theoretical DVFS combination.
Use `--gpu-index` or `--profile` to restrict it, and raise `--min-samples` and
`--min-queries` when filtering noisy exploratory runs.

### 4. Fit per-device FFN latency and recommend aligned shards

```bash
python3 scripts/treprefill_autotune/autotune_policy.py recommend \
  autotune-runs/llama32-i256-thermal/* \
  --model autotune-work/llama32-i256/model.json \
  --template pkg-adb/llama.cpp/policy/DATE2027/llama32-3b-256-0908-th5.json \
  --catalog autotune-work/llama32-i256/clock-catalog.json \
  --max-profile-regret 0.03 \
  --preserve-profile p15-g15-gpu14 \
  --output autotune-work/llama32-i256/plans.json
```

Each runner-created directory includes `input-policy.json`, which identifies
the shard size used by that trace. For older imported logs without a snapshot,
pass the common measured policy explicitly with `--policy PATH`. A processor
is retained by default; dropping one requires an explicit `--allow-drop cpu`,
`gpu`, or `npu`.

### 5. Generate and validate the policy

```bash
python3 scripts/treprefill_autotune/autotune_policy.py generate \
  --model autotune-work/llama32-i256/model.json \
  --template pkg-adb/llama.cpp/policy/DATE2027/llama32-3b-256-0908-th5.json \
  --plans autotune-work/llama32-i256/plans.json \
  --token 256 \
  --output autotune-work/llama32-i256/policy.json

python3 scripts/treprefill_autotune/autotune_policy.py validate \
  --model autotune-work/llama32-i256/model.json \
  --policy autotune-work/llama32-i256/policy.json \
  --token 256
```

Static validation checks exact width coverage, Llama dimensions, layer bounds,
and backend alignment (including 256-element HTP start and size alignment).
Run at least one device verification trial with the generated policy before
using it for evaluation.

## Scope and measurement cautions

- The optimizer follows the active lanes in `ffn_parallel.split_layout` and
  supports both CPU+GPU+NPU and GPU+NPU templates. It does not automatically
  decide whether two or three backends are globally faster; profile and verify
  those topologies as separate experiments, then compare their end-to-end
  prefill results.
- The current optimizer fits FFN partitions. QKV and attention-output splits
  are inherited from the closest template clock profile and scaled to the new
  model width; they are validated but not performance-tuned from traces.
- One measured shard size per processor is supported with a through-origin
  latency model. For a stronger fit, collect multiple policies with different
  shard sizes and pass all run directories to `recommend`; then optionally use
  `--fit-intercept`.
- Generated policies have exact token applicability. A multi-token profiling
  plan should use ``token_policies`` to map each token-string to its host-side
  generated policy; the older singular ``policy`` field remains suitable for
  a one-token plan or a policy intentionally shared by all trials.
- Profile pruning is opt-in through `recommend --max-profile-regret R`. Use
  `--preserve-profile NAME` for an anchor that must remain, such as the maximum
  clock profile. Pruning follows the runtime's nearest-clock selection rule.
- The policy generator currently targets dense Llama-family Q8_0 models with a
  uniform FFN width. It rejects incompatible dimensions instead of guessing.
- For another deployed model, set `common_env.MODEL` to its device GGUF path
  and `common_env.MODEL_TAG` to a log-safe name. If using the optional module
  benchmark path, also set `MODULE_BENCH_PROFILE=auto`; the binary then checks
  the loaded model's capabilities instead of requiring the exact Llama-3.2-3B
  name and dimensions.

## Single-trial ADB runner

`adb_runner.py` runs one profiling trial on the Android device. It is the I/O
layer for the policy autotuner: policy optimization and clock-state analysis
can import `AdbRunner` instead of duplicating ADB commands.

The device must already contain the TrePrefill package and model. By default,
the package root is `/data/local/tmp/llama.cpp`. Existing run scripts are used
unchanged.

### Command-line example

Run this from the repository root in WSL:

```bash
python3 scripts/treprefill_autotune/adb_runner.py \
  --su \
  --trial-id llama32-i256-p15-g15-gpu14-r0 \
  --policy pkg-adb/llama.cpp/policy/DATE2027/llama32-3b-256-0908-th5.json \
  --script llama32_ffn_switch_hybrid.sh \
  --env INPUT_LENGTH=256 \
  --env JSON_PATH=data/llama32_prefill_256.json \
  --env MAX_QUERY_NUMBER=100 \
  --env QUERY_PERIOD_MS=2000 \
  --env CPU_PRIME_P=15 \
  --env CPU_GOLD_P=15 \
  --env GPU_P=14 \
  --env SCHED_TRACE=on \
  --env HARDWARE_STATS=on \
  --require-artifact scheduler_trace.csv \
  --require-artifact hardware_stats.csv \
  --results-root autotune-runs
```

Use `--serial SERIAL` if multiple devices are connected. `--su` is needed by
the current experiment scripts because they set governors and other sysfs
controls. To inspect the resolved command without connecting to the device:

```bash
python3 scripts/treprefill_autotune/adb_runner.py \
  --trial-id dry-run-example \
  --env INPUT_LENGTH=256 \
  --dry-run
```

Each host result directory contains:

```text
autotune-runs/<trial-id>/
├── trial.json          # resolved command, environment, hashes, status, timing
├── input-policy.json   # exact policy snapshot, when --policy is used
├── adb-shell.log       # complete on-device console output
└── device-output/      # pulled OUTPUT_DIR (CSV files and config.txt)
```

The corresponding device files remain under
`/data/local/tmp/treprefill-autotune/<trial-id>/`. A repeated `trial-id` is
rejected locally and remotely; the runner never deletes or silently overwrites
an earlier experiment.

If the host-side execution timeout expires, the manifest is marked
`remote-state-uncertain`: stopping the local `adb` process cannot prove that
the on-device script exited. A trial plan therefore stops immediately in this
case, even with `--continue-on-error`; Ctrl-C during remote execution is handled
the same way. Check the device before resuming. Trial-plan loading also checks
every configured policy path before the first device run begins.

### Python API

```python
from pathlib import Path

from scripts.treprefill_autotune.adb_runner import AdbRunner, TrialSpec

result = AdbRunner(serial=None).run(
    TrialSpec(
        trial_id="ffn-i256-state-001",
        local_results_root=Path("autotune-runs"),
        policy=Path("generated-policy.json"),
        environment={
            "INPUT_LENGTH": "256",
            "JSON_PATH": "data/llama32_prefill_256.json",
            "MAX_QUERY_NUMBER": "20",
            "SCHED_TRACE": "on",
            "HARDWARE_STATS": "on",
        },
        required_artifacts=("scheduler_trace.csv", "hardware_stats.csv"),
        use_su=True,
    )
)
```

The runner owns `OUTPUT_DIR`, the trace output paths, `RUN_NAME`, and `RUN_ID`
so every artifact is collected into one deterministic directory. Supplying any
of those keys through `--env` is rejected.
