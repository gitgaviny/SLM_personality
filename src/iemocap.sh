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
    stage=*)                stage="${arg#*=}" ;;
    stop_stage=*)           stop_stage="${arg#*=}" ;;
    split_id=*)             split_id="${arg#*=}" ;;
    encoder_base=*)         encoder_base="${arg#*=}" ;;
    base_data_path=*)        base_data_path="${arg#*=}" ;;
    epoch=*)                epoch="${arg#*=}" ;;
    asr_loss_weight=*)      asr_loss_weight="${arg#*=}" ;;
    emotion_loss_weight=*)  emotion_loss_weight="${arg#*=}" ;;
    corpus=*)               corpus="${arg#*=}" ;;
    pretrained_model=*)     pretrained_model="${arg#*=}" ;;
    encoder=*)              encoder="${arg#*=}" ;;
    decoder=*)              decoder="${arg#*=}" ;;
    decoder_base=*)          decoder_base="${arg#*=}" ;;
    fused_model=*)          fused_model="${arg#*=}" ;;
    encoder_freeze=*)       encoder_freeze="${arg#*=}" ;;
    decoder_freeze=*)       decoder_freeze="${arg#*=}" ;;
    adapter_only_decoder=*) adapter_only_decoder="${arg#*=}" ;;
    instruct=*)             instruct="${arg#*=}" ;;
    talker_ctc=*)           talker_ctc="${arg#*=}" ;;
    slm_dir=*)              slm_dir="${arg#*=}" ;;
    output_dir=*)           output_dir="${arg#*=}" ;;
    partial_encoder_unfreeze=*)      partial_encoder_unfreeze="${arg#*=}" ;;
    partial_decoder_unfreeze=*)       partial_decoder_unfreeze="${arg#*=}" ;;
    partial_others_unfreeze=*)       partial_others_unfreeze="${arg#*=}" ;;
    eval_steps=*)           eval_steps="${arg#*=}" ;;
    virtual_env=*)          virtual_env="${arg#*=}" ;;
    *) echo "Unknown option: $arg" >&2; exit 1 ;;
  esac
done

echo "++++ stage=$stage"
echo "++++ stop_stage=$stop_stage"
echo "++++ split_id=${split_id}"
echo "++++ encoder_base=$encoder_base"
echo "++++ epoch=$epoch"
echo "++++ asr_loss_weight=$asr_loss_weight"
echo "++++ emotion_loss_weight=$emotion_loss_weight"
echo "++++ corpus=$corpus"
echo "++++ base_data_path=${base_data_path:-}"
echo "++++ slm_dir=$slm_dir"
echo "++++ pretrained_model=${pretrained_model:-}"
echo "++++ encoder=$encoder"
echo "++++ decoder=$decoder"
echo "++++ decoder_base=${decoder_base:-}"
echo "++++ fused_model=$fused_model"
echo "++++ encoder_freeze=$encoder_freeze"
echo "++++ decoder_freeze=$decoder_freeze"
echo "++++ adapter_only_decoder=$adapter_only_decoder"
echo "++++ instruct=$instruct"
echo "++++ talker_ctc=$talker_ctc"
echo "++++ partial_encoder_unfreeze=$partial_encoder_unfreeze"
echo "++++ partial_decoder_unfreeze=$partial_decoder_unfreeze"
echo "++++ partial_others_unfreeze=$partial_others_unfreeze"
echo "++++ eval_steps=$eval_steps"
echo "++++ virtual_env=$virtual_env"

output_dir=${output_dir}/${encoder}-${decoder}
if [ "${encoder_freeze}" = "true" ]; then
    output_dir="${output_dir}-encoder_freeze"
else
    output_dir="${output_dir}-encoder_unfreeze"
fi
if [ "${decoder_freeze}" = "true" ]; then
    output_dir="${output_dir}-decoder_freeze"
else
    output_dir="${output_dir}-decoder_unfreeze"
fi
if [ "${adapter_only_decoder}" = "true" ]; then
    output_dir="${output_dir}-adater_decoder"
else
    output_dir="${output_dir}-adater_encoder_decoder"
fi
output_dir=${output_dir}-${corpus}-${epoch}
echo "++++ output_dir=$output_dir"

extra_args=""
if [[ "$instruct" == "true" ]]; then
  extra_args+=" --instruct"
fi


PY_BIN="$virtual_env/bin/python"

if [[ ! -x "$PY_BIN" ]]; then
  echo "ERROR: Python executable not found at $PY_BIN" >&2
  echo "Please set virtual_env=path/to/your/venv." >&2
  exit 1
fi
echo "Using Python: $("$PY_BIN" -V 2>&1)"

base_data_path="${base_data_path:-path/to/data/${corpus}}"
decoder_base="${decoder_base:-path/to/model}"


export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_LAUNCH_BLOCKING=1
export TORCH_SHOW_CPP_STACKTRACES=1



# 1. Data preparing
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  "$PY_BIN" utils/generate_dataset.py \
  --base_data_path "$base_data_path" \
	--wav_scp_name wav.scp \
	--output_dir datasets/
fi

# 2. Create the pre-trained AED from pre-trained speech encoder and LLMs
model_ids=${encoder}-${decoder}
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  save_dir="${slm_dir}/${model_ids}"
  "${PY_BIN}" utils/create_from_pretrained.py \
    --encoder_base "$encoder_base" \
    --decoder_base "$decoder_base" \
    --llm_id "${decoder}" \
    --save_dir "${save_dir}" \
    --talker_ctc \
    --check_generate \
    --asr_loss_weight ${asr_loss_weight}\
    --emotion_loss_weight ${emotion_loss_weight} \
    $extra_args
fi

# 3. Training
NUM_GPUS=$("$PY_BIN" -c 'import torch; print(torch.cuda.device_count())')
echo "Detected $NUM_GPUS GPUs"

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  if [[ -n "${pretrained_model:-}" ]]; then
    model_path="${pretrained_model}"
  else  
    model_path="${slm_dir}/${model_ids}"
  fi

  "$PY_BIN" -m torch.distributed.launch \
    --nproc_per_node=$NUM_GPUS finetune.py \
    --dataset_name="datasets/session$split_id" \
    --model_name_or_path="${model_path}" \
    --train_split_name="train" \
    --eval_split_name="test" \
    --adapter_only_decoder=${adapter_only_decoder} \
    --output_dir=${output_dir} \
    --metric_for_best_model="eval_loss" \
    --greater_is_better=false \
    --preprocessing_num_workers="16" \
    --audio_column_name="audio" \
    --text_column_name="text" \
    --overwrite_output_dir false \
    --num_train_epochs=${epoch} \
    --per_device_train_batch_size="4" \
    --per_device_eval_batch_size="4" \
    --gradient_accumulation_steps="4" \
    --learning_rate="5e-5" \
    --evaluation_strategy="steps" \
    --save_steps=${eval_steps} \
    --eval_steps=${eval_steps} \
    --logging_steps="40" \
    --save_total_limit="5" \
    --freeze_feature_encoder true \
    --freeze_encoder ${encoder_freeze} \
    --freeze_decoder ${decoder_freeze} \
    --partial_encoder_unfreeze="${partial_encoder_unfreeze}" \
    --partial_decoder_unfreeze="${partial_decoder_unfreeze}" \
    --partial_others_unfreeze="${partial_others_unfreeze}" \
    --gradient_checkpointing \
    --group_by_length \
    --predict_with_generate \
    --do_train true \
    --do_eval true \
    --do_lower_case

  "$PY_BIN" utils/merge_adapter.py ${output_dir}
fi

# 4. Inference
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  _set="test"

  for subset in $_set; do
    # dataset_name="data/${corpus}/${subset}"

    "$PY_BIN" -m torch.distributed.launch \
      --nproc_per_node 1 inference.py \
      --dataset_name="datasets/session$split_id/${subset}" \
      --model_name_or_path="${output_dir}" \
      --adapter_only_decoder="${adapter_only_decoder}" \
      --output_dir="${output_dir}" \
      --metric_for_best_model="eval_loss" \
      --greater_is_better=false \
      --preprocessing_num_workers="16" \
      --audio_column_name="audio" \
      --text_column_name="text" \
      --num_train_epochs="${epoch}" \
      --per_device_train_batch_size="16" \
      --per_device_eval_batch_size="16" \
      --gradient_accumulation_steps="1" \
      --learning_rate="3e-5" \
      --warmup_steps="400" \
      --evaluation_strategy="steps" \
      --save_steps="1600" \
      --eval_steps="${eval_steps}" \
      --logging_steps="10" \
      --save_total_limit="5" \
      --freeze_feature_encoder=true \
      --freeze_encoder="${encoder_freeze}" \
      --freeze_decoder="${decoder_freeze}" \
      --partial_encoder_unfreeze="${partial_encoder_unfreeze}" \
      --partial_decoder_unfreeze="${partial_decoder_unfreeze}" \
      --partial_others_unfreeze="${partial_others_unfreeze}" \
      --gradient_checkpointing \
      --group_by_length \
      --predict_with_generate \
      --do_train=false \
      --do_eval=true \
      --do_lower_case

    "$PY_BIN" utils/compute-wer.py \
      --char=1 \
      --v=1 \
      "${output_dir}/${subset}_label.text" "${output_dir}/${subset}_decod.text" \
      > "${output_dir}/${subset}_decod.wer"

    "$PY_BIN" utils/compute-accuracy.py \
      "${output_dir}/${subset}_label_emotion.text" "${output_dir}/${subset}_decod_emotion.text" \
      > "${output_dir}/${subset}_decod.accuracy"
  done

  for subset in $_set; do
    echo "==> ${output_dir} ${subset}"
    tail -n 5 "${output_dir}/${subset}_decod.wer" || true
    cat "${output_dir}/${subset}_decod.accuracy" || true
  done
fi



