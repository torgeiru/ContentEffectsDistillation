#!/usr/bin/env bash
set -euo pipefail

# =========================
# Local configuration
# =========================
# Assumes this script lives in the project root:
# .
# ├── baselines
# ├── hf_models
# └── semeval_2026_task_11
export PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
export SCRIPT_DIR="${SCRIPT_DIR:-$PROJECT_DIR/baselines}"
export REPO_DIR="${REPO_DIR:-$PROJECT_DIR/semeval_2026_task_11}"

# Local model copied from Fox.
export MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-$PROJECT_DIR/hf_models/Qwen3-8B}"

# Data paths matching your current folder structure.
export TRAIN_FILE="${TRAIN_FILE:-$REPO_DIR/train_data/subtask 1/train_data.json}"
export EVAL_FILE="${EVAL_FILE:-$REPO_DIR/test_data/subtask 1/test_data_subtask_1.json}"

# Same as the Fox Slurm script.
# Note: this only gives meaningful official metrics if EVAL_FILE contains labels.
export REFERENCE_FILE="${REFERENCE_FILE:-$EVAL_FILE}"

export OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/outputs_subtask1}"

# Four switches: 1 = run, 0 = skip.
RUN_PROMPT="${RUN_PROMPT:-1}"
RUN_ICL="${RUN_ICL:-1}"
RUN_FULL="${RUN_FULL:-0}"
RUN_PEFT="${RUN_PEFT:-1}"

# Keep empty for local single-GPU PEFT/prompt runs.
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-}"

# Fox-like defaults.
DTYPE="${DTYPE:-bf16}"
MAX_LENGTH="${MAX_LENGTH:-768}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-4}"
EPOCHS="${EPOCHS:-3}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"
FULL_LR="${FULL_LR:-1e-5}"
PEFT_LR="${PEFT_LR:-2e-4}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"

# Optional quick-test limits. Leave unset for full run.
LIMIT_TRAIN="${LIMIT_TRAIN:-}"
LIMIT_EVAL="${LIMIT_EVAL:-}"

# =========================
# Environment
# =========================
cd "$PROJECT_DIR"

if [[ -f "$PROJECT_DIR/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "$PROJECT_DIR/.venv/bin/activate"
else
  echo "No virtualenv found at $PROJECT_DIR/.venv/bin/activate" >&2
  echo "Create it first, then install baselines/requirements-amd.txt" >&2
  exit 1
fi

# ROCm device selection. PyTorch still exposes ROCm through torch.cuda.
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
export ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0}"

# Hugging Face/cache locations.
export HF_HOME="${HF_HOME:-$PROJECT_DIR/cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TOKENIZERS_PARALLELISM=false

mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" "$OUTPUT_DIR" "$PROJECT_DIR/logs"

# =========================
# Sanity checks
# =========================
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("hip:", torch.version.hip)
print("cuda available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("ROCm PyTorch cannot see the GPU.")
print("device:", torch.cuda.get_device_name(0))
if torch.version.hip is None:
    raise SystemExit("Installed torch is not a ROCm/HIP build.")
PY

if [[ ! -f "$SCRIPT_DIR/run_subtask1_baselines.py" ]]; then
  echo "Cannot find script: $SCRIPT_DIR/run_subtask1_baselines.py" >&2
  exit 1
fi

if [[ ! -d "$MODEL_NAME_OR_PATH" && "$MODEL_NAME_OR_PATH" != */* ]]; then
  echo "MODEL_NAME_OR_PATH does not exist and does not look like a HF repo id: $MODEL_NAME_OR_PATH" >&2
  exit 1
fi

if [[ ! -f "$TRAIN_FILE" ]]; then
  echo "TRAIN_FILE does not exist: $TRAIN_FILE" >&2
  exit 1
fi

if [[ ! -f "$EVAL_FILE" ]]; then
  echo "EVAL_FILE does not exist: $EVAL_FILE" >&2
  exit 1
fi

if [[ ! -f "$REFERENCE_FILE" ]]; then
  echo "REFERENCE_FILE does not exist: $REFERENCE_FILE" >&2
  exit 1
fi

if [[ "$RUN_FULL" == "1" && -z "$DEEPSPEED_CONFIG" ]]; then
  echo "Warning: RUN_FULL=1 on one local 24GB GPU is very likely to OOM." >&2
  echo "For local ROCm, prefer RUN_FULL=0 RUN_PEFT=1." >&2
fi

# =========================
# Command
# =========================
CMD=(
  python "$SCRIPT_DIR/run_subtask1_baselines.py"
  --repo_dir "$REPO_DIR"
  --train_file "$TRAIN_FILE"
  --eval_file "$EVAL_FILE"
  --reference_file "$REFERENCE_FILE"
  --model_name_or_path "$MODEL_NAME_OR_PATH"
  --output_dir "$OUTPUT_DIR"
  --dtype "$DTYPE"
  --max_length "$MAX_LENGTH"
  --gen_batch_size "$GEN_BATCH_SIZE"
  --epochs "$EPOCHS"
  --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE"
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
  --full_lr "$FULL_LR"
  --peft_lr "$PEFT_LR"
  --lora_r "$LORA_R"
  --lora_alpha "$LORA_ALPHA"
  --lora_dropout "$LORA_DROPOUT"
  --run_prompt "$RUN_PROMPT"
  --run_icl "$RUN_ICL"
  --run_full "$RUN_FULL"
  --run_peft "$RUN_PEFT"
)

if [[ -n "$DEEPSPEED_CONFIG" ]]; then
  CMD+=(--deepspeed_config "$DEEPSPEED_CONFIG")
fi

if [[ -n "$LIMIT_TRAIN" ]]; then
  CMD+=(--limit_train "$LIMIT_TRAIN")
fi

if [[ -n "$LIMIT_EVAL" ]]; then
  CMD+=(--limit_eval "$LIMIT_EVAL")
fi

LOG_FILE="$PROJECT_DIR/logs/local_rocm_subtask1_$(date +%Y%m%d_%H%M%S).log"

echo "Project dir: $PROJECT_DIR"
echo "Output dir:  $OUTPUT_DIR"
echo "Log file:    $LOG_FILE"
echo
echo "Running:"
printf ' %q' "${CMD[@]}"
echo
echo

"${CMD[@]}" 2>&1 | tee "$LOG_FILE"
