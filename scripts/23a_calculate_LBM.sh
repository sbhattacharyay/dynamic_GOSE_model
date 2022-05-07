#!/bin/bash
#SBATCH --job-name=LBM
#SBATCH --time=24:00:00
#SBATCH --array=0-9299
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=MENON-SL2-CPU
#SBATCH --partition=icelake
#SBATCH --mail-type=ALL
#SBATCH --output=./hpc_logs/feature_importance/LBM_trial_%a.out
#SBATCH --mail-user=sb2406@cam.ac.uk

. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module load rhel8/default-icl
module load python/3.8

source ~/python_venv/bin/activate

srun python 23a_calculate_LBM.py $SLURM_ARRAY_TASK_ID