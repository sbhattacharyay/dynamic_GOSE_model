#!/bin/bash
#SBATCH -J v6_sensitivity_analysis
#SBATCH -A MENON-SL3-CPU
#SBATCH -p icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:20:00
#SBATCH --mem=54080
#SBATCH --array=0-999
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sb2406@cam.ac.uk
#SBATCH --output=/home/sb2406/rds/hpc-work/model_outputs/v6-0/hpc_logs/sensitivity_analysis/dynAPM_static_only_v6-0_trial_%a.out

. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module purge
module load rhel8/default-icl
module load python/3.8

source ~/python_venv/bin/activate

srun python 06c_sensitivity_performance.py $SLURM_ARRAY_TASK_ID