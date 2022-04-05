#!/bin/bash
#SBATCH --job-name=v6_calibrated_dynAPM_training
#SBATCH --time=10:00:00
#SBATCH --array=0-199
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --account=MENON-SL3-GPU
#SBATCH --partition=ampere
#SBATCH --mail-type=ALL
#SBATCH --output=./hpc_logs/training/calibrated_dynAPM_training_v6-0_trial_%a.out
#SBATCH --mail-user=sb2406@cam.ac.uk

. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module load rhel8/default-icl
module load python/3.8

source ~/python_venv/bin/activate

srun python 21f_train_calibrated_v6_models.py $SLURM_ARRAY_TASK_ID