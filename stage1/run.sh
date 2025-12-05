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
    split_id=*)             split_id="${arg#*=}" ;;
    *) echo "Unknown option: $arg" >&2; exit 1 ;;
  esac
done


echo "++++ output_dir=$output_dir"
echo "++++ virtual_env=$virtual_env"
echo "++++ split_id=$split_id"


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
# Prefer Python 3.10 to match your original venv; fallback to 3.11 or 3.8 if needed.
TARGET_PY="$(command -v python3.10 || true)"
if [[ -z "${TARGET_PY}" ]]; then
  # If 'python3' is 3.10 on this node, use it
  if python3 -V 2>/dev/null | grep -qE '^Python 3\.10\.'; then
    TARGET_PY="$(command -v python3)"
  fi
fi
# Last resort: allow 3.11 or 3.8 (may require reinstalling wheels later)
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

# --- Repair venv python symlinks if broken and align with node's interpreter ---
# Reason: your venv/bin/python pointed to /usr/bin/python3.10, which doesn’t exist on some nodes.
if [[ ! -x "$PY_BIN" ]] || { [[ -L "$PY_BIN" ]] && [[ ! -e "$(readlink -f "$PY_BIN")" ]]; }; then
  echo ">>> Fixing venv python links to: $TARGET_PY"
  ln -sf "$TARGET_PY" "$virtual_env/bin/python"
  ln -sf "$TARGET_PY" "$virtual_env/bin/python3"
  # If the chosen interpreter is 3.10, also create python3.10 name for tools that expect it
  if "$TARGET_PY" -V 2>&1 | grep -qE '^Python 3\.10\.'; then
    ln -sf "$TARGET_PY" "$virtual_env/bin/python3.10"
  fi

  # Upgrade venv layout in-place so all scripts' shebangs point to the new interpreter
  "$TARGET_PY" -m venv --upgrade "$virtual_env"
fi


MODEL=hubert-large-ls960-ft
TOKENIZER=wav2vec2-base
ALPHA=0.1
LR=3e-5
WORKER_NUM=8

NUM_GPUS=$("$PY_BIN" -c 'import torch; print(torch.cuda.device_count())')
echo "Detected $NUM_GPUS GPUs"

"$PY_BIN" -m torch.distributed.launch \
  --nproc_per_node=$NUM_GPUS run_emotion.py \
  --output_dir=$output_dir \
  --cache_dir=cache/ \
  --num_train_epochs=100 \
  --per_device_train_batch_size="2" \
  --per_device_eval_batch_size="2" \
  --gradient_accumulation_steps=4 \
  --alpha $ALPHA \
  --dataset_name emotion \
  --split_id "$split_id" \
  --evaluation_strategy="steps" \
  --save_strategy="steps" \
  --load_best_model_at_end \
  --metric_for_best_model="ua" \
  --greater_is_better="True" \
  --save_total_limit="3" \
  --save_steps="500" \
  --eval_steps="500" \
  --logging_steps="50" \
  --logging_dir="log" \
  --do_train \
  --do_eval \
  --learning_rate=$LR \
  --model_name_or_path=facebook/$MODEL \
  --tokenizer facebook/$TOKENIZER \
  --preprocessing_num_workers=$WORKER_NUM \
  --dataloader_num_workers $WORKER_NUM \
  --freeze_feature_extractor="True" \
