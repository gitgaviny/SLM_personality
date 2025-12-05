#!/bin/bash
#SBATCH --job-name=slm-sentiment
#SBATCH --partition=002-partition-all
#SBATCH --gpus-per-node=8
#SBATCH --array=1-5
#SBATCH --container-image=/lustre/container_cache/slam-llm/slam-llm-latest.sqsh
#SBATCH --container-mounts=/lustre:/lustre
#SBATCH --exclusive

export HF_HOME="/lustre/teams/speech/hf_cache/"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
export WORKING_DIR="${SLURM_SUBMIT_DIR:-$SCRIPT_DIR}"

cd "$WORKING_DIR"
split_id=$SLURM_ARRAY_TASK_ID
# split_id=1
# cd /lustre/users/gao/speechllm/step2/sentiment/slurm

echo "Running split_id = ${split_id}"

stage=2
stop_stage=4
corpus=iemocap_special_token_sentiment
encoder_base=/lustre/users/gao/speechllm/step1/sentiment_baseline/session${split_id}

epoch=20
encoder=hubert-large
decoder=Llama-3.2-1B-Instruct
fused_model=false
encoder_freeze=false
decoder_freeze=true
adapter_only_decoder=true
instruct=true
talker_ctc=true
eval_steps=100
virtual_env=/lustre/users/gao/envs/slmser/venv

if [ "$fused_model" = true ]; then
  slm_dir=dump_ff
  output_dir=exp_ff
fi
if [ "$fused_model" = false ]; then
  slm_dir=dump
  output_dir=exp
fi
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
  # pretrained_model=$pretrained_model
  encoder=$encoder
  decoder=$decoder
  fused_model=$fused_model
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

if [ "$fused_model" = true ]; then
  bash ../iemocap_ff.sh "${args[@]}"
fi
if [ "$fused_model" = false ]; then
  bash ../iemocap.sh "${args[@]}"
fi
