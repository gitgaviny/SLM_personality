#!/bin/bash
#SBATCH --job-name=prediction-hubert
#SBATCH --partition=002-partition-all
#SBATCH --gpus=8
#SBATCH --array=1-5
#SBATCH --container-image=/lustre/container_cache/slam-llm/slam-llm-latest.sqsh
#SBATCH --container-mounts=/lustre:/lustre
#SBATCH --exclusive

cd $SLURM_SUBMIT_DIR

virtual_env=/lustre/users/gao/envs/slmser/venv
output_dir=predictions/session

split_id=${SLURM_ARRAY_TASK_ID}

model_path=./session${split_id}

args=(
  output_dir=${output_dir}
  virtual_env=${virtual_env}
  split_id=${split_id}
  model_path=${model_path}
)

bash ../prediction.sh "${args[@]}"
