#!/bin/bash
#SBATCH --job-name=calculate_predictions_on_sets
#SBATCH --time=01:00:00
#SBATCH --array=0-149
#SBATCH --nodes=1
#SBATCH --ntasks=38
#SBATCH --account=MENON-SL2-CPU
#SBATCH --partition=icelake-himem
#SBATCH --mem=256880
#SBATCH --mail-type=ALL
#SBATCH --output=./hpc_logs/training/v5-0_calculate_model_predictions_trial_%a.out
#SBATCH --mail-user=sb2406@cam.ac.uk

. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module load rhel8/default-icl
module load python/3.8

source ~/python_venv/bin/activate

srun python 17b_calculate_newest_model_predictions.py $SLURM_ARRAY_TASK_ID