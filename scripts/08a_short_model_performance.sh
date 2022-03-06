#!/bin/bash
#SBATCH --job-name=short_dynAPM_performance
#SBATCH --time=00:20:00
#SBATCH --array=0-5999
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=MENON-SL2-CPU
#SBATCH --partition=icelake
#SBATCH --mail-type=ALL
#SBATCH --output=./hpc_logs/performance/short_dynAPM_reperformance_trial_%a.out
#SBATCH --mail-user=sb2406@cam.ac.uk

. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module load rhel8/default-icl
module load python/3.8

source ~/python_venv/bin/activate

srun python 08a_short_model_performance.py $SLURM_ARRAY_TASK_ID