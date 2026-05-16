#!/bin/bash
#SBATCH --job-name=slm-train-infer
#SBATCH --partition=CHANGEME_PARTITION
#SBATCH --gpus-per-node=8
#SBATCH --exclusive

set -euo pipefail

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
export WORKING_DIR="${SLURM_SUBMIT_DIR:-$SCRIPT_DIR}"

cd "$WORKING_DIR"

repeat=1

stage=3
stop_stage=4
corpus=iemocap_special_token_serasr

epoch=20
asr_loss_weight=0.1
emotion_loss_weight=1

encoder=hubert-large
decoder=Llama-3.2-1B-Instruct

encoder_freeze=false
decoder_freeze=true
adapter_only_decoder=true
instruct=true
talker_ctc=true
eval_steps=100
virtual_env="/path/to/your/venv"

partial_encoder_unfreeze=""
partial_decoder_unfreeze="lm_head,embed_tokens,embed_positions,layernorm_embedding"
partial_others_unfreeze="enc_to_dec_proj"

echo "== repeat=${repeat}, stage=${stage}-${stop_stage}"
echo "== fused_model=${fused_model}, corpus=${corpus}, epoch=${epoch}, w_asr=${asr_loss_weight}, w_emo=${emotion_loss_weight}"
echo "== encoder=${encoder}, decoder=${decoder}"

for rep in $(seq 1 "${repeat}"); do
  echo
  echo "#############################################"
  echo "### Repetition ${rep}/${repeat}"
  echo "#############################################"

  if [[ "${rep}" -eq 1 ]]; then
    dump_prefix="dump"
    exp_prefix="exp_testreal"
  else
    dump_prefix="dump"
    exp_prefix="exp_testreal${rep}"
  fi

  base_slm_dir="${dump_prefix}"
  base_output_dir="${exp_prefix}_${asr_loss_weight}_${emotion_loss_weight}"

  for split_id in 1 2 3 4 5; do
    echo
    echo "=============================="
    echo "Running split_id = ${split_id} (rep ${rep}/${repeat})"
    echo "=============================="

    encoder_base="/path/to/pretrained/encoder/session${split_id}"
    slm_dir="${base_slm_dir}/session${split_id}"
    output_dir="${base_output_dir}/session${split_id}"

    args=(
      stage=$stage
      stop_stage=$stop_stage
      split_id=$split_id
      encoder_base=${encoder_base}
      epoch=$epoch
      asr_loss_weight=$asr_loss_weight
      emotion_loss_weight=$emotion_loss_weight
      corpus=$corpus
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
  done
done

echo
echo "== All experiments finished =="
