#!/bin/bash
#SBATCH -J v6_TimeSHAP_calculation_dynAPM
#SBATCH -A MENON-SL2-CPU
#SBATCH -p icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH --mem=54080
#SBATCH --array=2-9999
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sb2406@cam.ac.uk
#SBATCH --output=/home/sb2406/rds/hpc-work/model_interpretations/v6-0/hpc_logs/timeSHAP/dynAPM_timeSHAP_v6-0_trial_%a.out

. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module purge
module load rhel8/default-icl
module load python/3.8

source ~/python_venv/bin/activate

srun python 05c_calculate_full_set_TimeSHAP.py $SLURM_ARRAY_TASK_ID