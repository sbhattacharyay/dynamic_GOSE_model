#!/bin/bash
#SBATCH --job-name=tokenise_new_numeric_predictors
#SBATCH --time=01:00:00
#SBATCH --array=1-100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=MENON-SL2-CPU
#SBATCH --partition=icelake-himem
#SBATCH --mem=27040
#SBATCH --mail-type=ALL
#SBATCH --output=./hpc_logs/tokenisation/diff_numeric_predictor_tokenisation_trial_%a.out
#SBATCH --mail-user=sb2406@cam.ac.uk

. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module load rhel8/default-icl
module load r-4.0.2-gcc-5.4.0-xyx46xb 

srun Rscript 13b_tokenise_numeric_predictors.R $SLURM_ARRAY_TASK_ID