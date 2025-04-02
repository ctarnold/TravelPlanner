#!/bin/bash
#SBATCH --job-name=FH3            # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=16G         # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:1             # of gpus per node
#SBATCH --time=00:59:59          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=ca2992@princeton.edu
#SBATCH --constraint=gpu80

module purge
module load anaconda3/2024.6
module load cudatoolkit/12.8
conda activate thesis

set CUDA_LAUNCH_BLOCKING=1
set TORCH_USE_CUDA_DSA=1
python ../../first_hundred_eval.py 3