#!/bin/bash
#SBATCH --job-name=parametric_mmin_investigation
#SBATCH --output=/home/afarah/p2/spectral-sirens-with-GPs/src/outfiles/parametric_mmin_investigation.out
#SBATCH --error=/home/afarah/p2/spectral-sirens-with-GPs/src/outfiles/parametric_mmin_investigation.err
#SBATCH --account=kicp
#SBATCH --time=48:00:00
#SBATCH --partition=kicp-gpu
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --gres=gpu:1

export LD_LIBRARY_PATH=/lib64:$LD_LIBRARY_PATH
module load cuda/11.5
echo "loaded cuda"
module load python/anaconda-2022.05
echo "loaded anaconda"
source activate jax_gpu
echo "activated env"
python /project2/kicp/afarah/spectral-sirens-with-GPs/src/scripts/bias_study.py
