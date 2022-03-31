#!/bin/bash
#SBATCH --job-name=model_calibration
#SBATCH --time=01:00:00
#SBATCH --array=0-1679
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=MENON-SL2-CPU
#SBATCH --partition=icelake
#SBATCH --mem=27040
#SBATCH --mail-type=ALL
#SBATCH --output=./hpc_logs/calibration/calibration_v6_trial_%a.out
#SBATCH --mail-user=sb2406@cam.ac.uk

. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module load rhel8/default-icl
module load python/3.8

source ~/python_venv/bin/activate

srun python 21d_model_calibration.py $SLURM_ARRAY_TASK_ID