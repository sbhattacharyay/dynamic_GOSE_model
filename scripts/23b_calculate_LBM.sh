#!/bin/bash
#SBATCH --job-name=LBM_p2
#SBATCH --time=10:00:00
#SBATCH --array=0-9999
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=MENON-SL2-CPU
#SBATCH --partition=icelake
#SBATCH --mail-type=ALL
#SBATCH --output=./hpc_logs/feature_importance/LBM_p2_trial_%a.out
#SBATCH --mail-user=sb2406@cam.ac.uk

. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module load rhel8/default-icl
module load python/3.8

source ~/python_venv/bin/activate

srun python 23b_calculate_LBM.py $SLURM_ARRAY_TASK_ID