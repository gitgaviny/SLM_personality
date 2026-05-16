#!/bin/bash
#SBATCH --job-name=slm-stage2
#SBATCH --partition=CHANGEME_PARTITION
#SBATCH --gpus-per-node=8
#SBATCH --array=1-5
#SBATCH --exclusive

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
export WORKING_DIR="${SLURM_SUBMIT_DIR:-$SCRIPT_DIR}"

cd "$WORKING_DIR"
split_id=$SLURM_ARRAY_TASK_ID

echo "Running split_id = ${split_id}"

stage=2
stop_stage=2
corpus=iemocap_special_token_serasr
encoder_base="/path/to/stage1/session${split_id}"
decoder_base="path/to/model"

epoch=20
encoder=hubert-large
decoder=Llama-3.2-1B-Instruct
encoder_freeze=false
decoder_freeze=true
adapter_only_decoder=true
instruct=true
talker_ctc=true
eval_steps=100
virtual_env="/path/to/your/venv"
asr_loss_weight=0.1
emotion_loss_weight=1

slm_dir=dump
output_dir=exp
slm_dir=${slm_dir}/session${split_id}
output_dir=${output_dir}/session${split_id}

partial_encoder_unfreeze=""
partial_decoder_unfreeze="lm_head,embed_tokens,embed_positions,layernorm_embedding"
partial_others_unfreeze="enc_to_dec_proj"

args=(
  stage=$stage
  stop_stage=$stop_stage
  split_id=$split_id
  encoder_base=${encoder_base}
  epoch=$epoch
  corpus=$corpus
  decoder_base=${decoder_base}
  asr_loss_weight=$asr_loss_weight
  emotion_loss_weight=$emotion_loss_weight
  encoder=$encoder
  decoder=$decoder
  fused_model=false
  encoder_freeze=$encoder_freeze
  decoder_freeze=$decoder_freeze
  partial_encoder_unfreeze=$partial_encoder_unfreeze
  partial_decoder_unfreeze=$partial_decoder_unfreeze
  partial_others_unfreeze=$partial_others_unfreeze
  adapter_only_decoder=$adapter_only_decoder
  instruct=$instruct
  talker_ctc=$talker_ctc
  slm_dir=${slm_dir}
  output_dir=${output_dir}
  eval_steps=${eval_steps}
  virtual_env=${virtual_env}
)

bash ../iemocap.sh "${args[@]}"
