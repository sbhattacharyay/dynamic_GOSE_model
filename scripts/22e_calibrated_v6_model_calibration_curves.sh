#!/bin/bash
#SBATCH --job-name=calibrated_dynAPM_calib_curves
#SBATCH --time=00:20:00
#SBATCH --array=0-3999
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=MENON-SL2-CPU
#SBATCH --partition=icelake
#SBATCH --mem=27040
#SBATCH --mail-type=ALL
#SBATCH --output=./hpc_logs/performance/calibrated_v6-0_dynAPM_performance_trial_%a.out
#SBATCH --mail-user=sb2406@cam.ac.uk

. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module load rhel8/default-icl
module load python/3.8

source ~/python_venv/bin/activate

srun python 22e_calibrated_v6_model_calibration_curves.py $SLURM_ARRAY_TASK_ID