#!/bin/bash
#SBATCH --job-name=hubert-baseline-sentiment
#SBATCH --partition=002-partition-all
#SBATCH --array=1-5            
#SBATCH --gpus=8               
#SBATCH --container-image=/lustre/container_cache/slam-llm/slam-llm-latest.sqsh
#SBATCH --container-mounts=/lustre:/lustre
#SBATCH --exclusive


cd $SLURM_SUBMIT_DIR

virtual_env=/lustre/users/gao/envs/slmser/venv

split_id=$SLURM_ARRAY_TASK_ID 
output_dir="session${split_id}"

echo "--- Starting SLURM Array Task ID: ${SLURM_ARRAY_TASK_ID} ---"
echo "++++ output_dir=${output_dir}"
echo "++++ split_id=${split_id}"

args=(
  output_dir=${output_dir}
  virtual_env=${virtual_env}
  split_id=${split_id}
)

bash ../run.sh "${args[@]}"

echo "--- Finished SLURM Array Task ID: ${SLURM_ARRAY_TASK_ID} ---"