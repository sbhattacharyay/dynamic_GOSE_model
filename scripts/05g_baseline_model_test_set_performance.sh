#!/bin/bash
#SBATCH -J static_test_set_bootstrapping
#SBATCH -A MENON-SL2-CPU
#SBATCH -p icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:20:00
#SBATCH --mem=54080
#SBATCH --array=0-999
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sb2406@cam.ac.uk
#SBATCH --output=/home/sb2406/rds/hpc-work/model_performance/BaselineComparison/hpc_logs/test_bootstrapping/static_test_bootstrapping_trial_%a.out

. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module purge
module load rhel8/default-icl
module load python/3.8

source ~/python_venv/bin/activate

srun python 05g_baseline_model_test_set_performance.py $SLURM_ARRAY_TASK_ID