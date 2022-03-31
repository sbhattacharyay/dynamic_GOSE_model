#!/bin/bash
#SBATCH --job-name=val_set_performances
#SBATCH --time=01:00:00
#SBATCH --array=0-764
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=MENON-SL2-CPU
#SBATCH --partition=icelake
#SBATCH --mem=27040
#SBATCH --mail-type=ALL
#SBATCH --output=./hpc_logs/performance/calc_val_set_performance_v6_trial_%a.out
#SBATCH --mail-user=sb2406@cam.ac.uk

. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module load rhel8/default-icl
module load python/3.8

source ~/python_venv/bin/activate

srun python 21b_validation_set_performance.py $SLURM_ARRAY_TASK_ID