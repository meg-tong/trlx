#!/bin/bash
#SBATCH --job-name=trlx
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=8
#SBATCH --ntasks-per-node=1
#SBATCH --partition=compute
#SBATCH --mem=800G
#SBATCH --output=logs/%x_%j.out

export WANDB__SERVICE_WAIT=300
export PYTHONPATH="${PYTHONPATH}:/data/meg_tong/situational-awareness/trlx"
source /opt/rh/devtoolset-10/enable
conda activate sita
# just pass all the arguments forward
python "$@"