#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
root_dir="$SCRIPT_DIR"

cd "$root_dir"
echo "++++ Current path: ${root_dir}"
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# General configuration
for arg in "$@"; do
  case $arg in
    output_dir=*)           output_dir="${arg#*=}" ;;
    virtual_env=*)          virtual_env="${arg#*=}" ;;
    split_id=*)             split_id="${arg#*=}" ;;   # NEW
    model_path=*)            model_path="${arg#*=}" ;;  # NEW
    *) echo "Unknown option: $arg" >&2; exit 1 ;;
  esac
done

echo "++++ output_dir=$output_dir"
echo "++++ virtual_env=$virtual_env"
echo "++++ split_id=$split_id"      # NEW
echo "++++ model_path=$model_path"    # NEW

PY_BIN="$virtual_env/bin/python"

echo "== Node: $(hostname)"
echo "== Before fix: show current venv python status"
ls -l "$PY_BIN" || echo "ls failed (maybe broken symlink)"
echo "Resolved -> $(readlink -f "$PY_BIN" || echo NA)"
echo "System python on this node:"
command -v python3 || true
python3 -V || true
echo "Available /usr/bin/python* on this node:"
ls -l /usr/bin/python* 2>/dev/null || true

# --- Choose a working system Python on this compute node ---
TARGET_PY="$(command -v python3.10 || true)"
if [[ -z "${TARGET_PY}" ]]; then
  if python3 -V 2>/dev/null | grep -qE '^Python 3\.10\.'; then
    TARGET_PY="$(command -v python3)"
  fi
fi
if [[ -z "${TARGET_PY}" ]]; then
  TARGET_PY="$(command -v python3.11 || true)"
fi
if [[ -z "${TARGET_PY}" ]]; then
  TARGET_PY="$(command -v python3.8 || true)"
fi

if [[ -z "${TARGET_PY}" ]]; then
  echo "ERROR: No suitable system python3 found on this node." >&2
  exit 1
fi
echo "Using system Python: $TARGET_PY ($("$TARGET_PY" -V 2>&1))"

if [[ ! -x "$PY_BIN" ]] || { [[ -L "$PY_BIN" ]] && [[ ! -e "$(readlink -f "$PY_BIN")" ]]; }; then
  echo ">>> Fixing venv python links to: $TARGET_PY"
  ln -sf "$TARGET_PY" "$virtual_env/bin/python"
  ln -sf "$TARGET_PY" "$virtual_env/bin/python3"
  if "$TARGET_PY" -V 2>&1 | grep -qE '^Python 3\.10\.'; then
    ln -sf "$TARGET_PY" "$virtual_env/bin/python3.10"
  fi
  "$TARGET_PY" -m venv --upgrade "$virtual_env"
fi

MODEL=wavlm-large
TOKENIZER=wav2vec2-base
ALPHA=0.1
LR=5e-5
WORKER_NUM=8

NUM_GPUS=$("$PY_BIN" -c 'import torch; print(torch.cuda.device_count())')
echo "Detected $NUM_GPUS GPUs"

mkdir -p predictions
export output=$output_dir$split_id

"$PY_BIN" -m torch.distributed.launch \
    --nproc_per_node=$NUM_GPUS run_emotion.py \
    --output_dir="$model_path" \
    --overwrite_output_dir \
    --num_train_epochs=200 \
    --warmup_ratio 0.1 \
    --per_device_train_batch_size="1" \
    --per_device_eval_batch_size="1" \
    --gradient_accumulation_steps=1 \
    --alpha 0.1 \
    --split_id "$split_id" \
    --evaluation_strategy="steps" \
    --save_total_limit="1" \
    --do_predict \
    --output_file "$output/predictions.txt" \
    --save_steps="500" \
    --eval_steps="5" \
    --logging_steps="5" \
    --logging_dir="log" \
    --learning_rate=0.1 \
    --model_name_or_path "$model_path" \
    --tokenizer facebook/wav2vec2-base \
    --preprocessing_num_workers=20 \
    --fp16 \
    --dataloader_num_workers 4
