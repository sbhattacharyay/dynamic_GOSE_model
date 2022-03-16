#!/bin/bash
#SBATCH --job-name=SHAP_dynAPM_deep
#SBATCH --time=05:00:00
#SBATCH --array=0-419
#SBATCH --nodes=1
#SBATCH --ntasks=38
#SBATCH --account=MENON-SL2-CPU
#SBATCH --partition=icelake-himem
#SBATCH --mem=256880
#SBATCH --mail-type=ALL
#SBATCH --output=./hpc_logs/feature_importance/calculate_SHAP_dynAPM_deep_trial_%a.out
#SBATCH --mail-user=sb2406@cam.ac.uk

. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module load rhel8/default-icl
module load python/3.8

source ~/python_venv/bin/activate

srun python 12a_dynAPM_deep_SHAP.py $SLURM_ARRAY_TASK_ID
