#!/bin/bash
#SBATCH --job-name=trlx
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=10
#SBATCH --ntasks-per-node=1
#SBATCH --mem=800G
#SBATCH --output=logs/%x_%j.out

export WANDB__SERVICE_WAIT=300
source /opt/rh/devtoolset-10/enable

# just pass all the arguments forward
python "$@"
