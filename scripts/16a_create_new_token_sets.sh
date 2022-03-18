#!/bin/bash
#SBATCH --job-name=create_new_token_sets
#SBATCH --time=01:00:00
#SBATCH --array=1-200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=MENON-SL2-CPU
#SBATCH --partition=icelake-himem
#SBATCH --mem=27040
#SBATCH --mail-type=ALL
#SBATCH --output=./hpc_logs/tokenisation/create_new_token_sets_%a.out
#SBATCH --mail-user=sb2406@cam.ac.uk

. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module load rhel8/default-icl
module load r-4.0.2-gcc-5.4.0-xyx46xb 

srun Rscript 16a_create_new_token_sets.R $SLURM_ARRAY_TASK_ID