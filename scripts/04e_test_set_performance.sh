#!/bin/bash
#SBATCH -J v7_test_set_bootstrapping
#SBATCH -A MENON-SL2-CPU
#SBATCH -p icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:20:00
#SBATCH --mem=54080
#SBATCH --array=0-999
#SBATCH --mail-type=NONE
#SBATCH --output=/home/sb2406/rds/hpc-work/model_performance/v7-0/hpc_logs/test_bootstrapping/dynAPM_test_bootstrapping_v7-0_trial_%a.out

. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module purge
module load rhel8/default-icl
module load python/3.8

source ~/python_venv/bin/activate

srun python 04e_test_set_performance.py $SLURM_ARRAY_TASK_ID