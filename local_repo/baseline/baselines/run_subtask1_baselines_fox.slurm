#!/usr/bin/env bash
#SBATCH --job-name=semeval26_s11_sub1
#SBATCH --account=ec403
#SBATCH --partition=accel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --gpus=1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# =========================
# User configuration
# =========================
export PROJECT_DIR=/cluster/work/projects/ec403/mht/IN5550-Track-Reasoning
export SCRIPT_DIR=$PROJECT_DIR/baselines
export REPO_DIR=$PROJECT_DIR/semeval_2026_task_11
export MODEL_NAME_OR_PATH=/fp/projects01/ec403/hf_models/Qwen3-8B
export TRAIN_FILE=$REPO_DIR/train_data/subtask\ 1/train_data.json
export EVAL_FILE=$REPO_DIR/test_data/subtask\ 1/test_data_subtask_1.json
export REFERENCE_FILE=$EVAL_FILE
export OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/outputs_subtask1}"
# Four switches: 1 = run, 0 = skip.
RUN_PROMPT="${RUN_PROMPT:-1}"
RUN_ICL="${RUN_ICL:-1}"
RUN_FULL="${RUN_FULL:-0}"
RUN_PEFT="${RUN_PEFT:-1}"

# Full fine-tuning Qwen-8B is memory-heavy. If RUN_FULL=1, prefer multi-GPU + DeepSpeed.
# For a 1-GPU PEFT/prompt job, keep DEEPSPEED_CONFIG empty.
DEEPSPEED_CONFIG=""
#DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-}"
#if [[ "$RUN_FULL" == "1" && -z "$DEEPSPEED_CONFIG" ]]; then
#  DEEPSPEED_CONFIG="$SCRIPT_DIR/ds_zero3_bf16.json"
#fi

#mkdir -p "$PROJECT_DIR" "$SCRIPT_DIR" "$OUTPUT_DIR" logs
cd "$PROJECT_DIR"

# =========================
# Environment on Fox
# =========================
module purge
module use -a /fp/projects01/ec30/software/easybuild/modules/all/
module load nlpl-trl/0.22.2-foss-2024a-Python-3.12.3


# HuggingFace/cache locations; change to your project scratch if desired.
export HF_HOME="${HF_HOME:-$PROJECT_DIR/cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"

# Clone official repo if missing.
#if [[ ! -d "$REPO_DIR" ]]; then
#  git clone https://github.com/neuro-symbolic-ai/semeval_2026_task_11.git "$REPO_DIR"
#fi

# Sanity checks for data paths.
if [[ ! -f "$TRAIN_FILE" ]]; then
  echo "TRAIN_FILE does not exist: $TRAIN_FILE" >&2
  echo "Please set TRAIN_FILE to the Subtask 1 training JSON path." >&2
  exit 1
fi
if [[ ! -f "$EVAL_FILE" ]]; then
  echo "EVAL_FILE does not exist: $EVAL_FILE" >&2
  echo "Please set EVAL_FILE to the Subtask 1 dev/test JSON path." >&2
  exit 1
fi
if [[ ! -f "$REFERENCE_FILE" ]]; then
  echo "REFERENCE_FILE does not exist: $REFERENCE_FILE" >&2
  echo "Please set REFERENCE_FILE to the gold reference JSON for evaluation." >&2
  exit 1
fi

CMD=(
  python "$SCRIPT_DIR/run_subtask1_baselines.py"
  --repo_dir "$REPO_DIR"
  --train_file "$TRAIN_FILE"
  --eval_file "$EVAL_FILE"
  --reference_file "$REFERENCE_FILE"
  --model_name_or_path "$MODEL_NAME_OR_PATH"
  --output_dir "$OUTPUT_DIR"
  --dtype bf16
  --max_length 768
  --gen_batch_size 4
  --epochs "${EPOCHS:-3}"
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-16}"
  --full_lr "${FULL_LR:-1e-5}"
  --peft_lr "${PEFT_LR:-2e-4}"
  --lora_r "${LORA_R:-16}"
  --lora_alpha "${LORA_ALPHA:-32}"
  --lora_dropout "${LORA_DROPOUT:-0.05}"
  --run_prompt "$RUN_PROMPT"
  --run_icl "$RUN_ICL"
  --run_full "$RUN_FULL"
  --run_peft "$RUN_PEFT"
)

if [[ -n "$DEEPSPEED_CONFIG" ]]; then
  CMD+=(--deepspeed_config "$DEEPSPEED_CONFIG")
fi

"${CMD[@]}"
