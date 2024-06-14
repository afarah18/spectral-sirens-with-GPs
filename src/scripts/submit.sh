#!/bin/bash
#SBATCH --job-name=parametric_unifq_actually
#SBATCH --output=/home/afarah/projects/spectral-sirens-with-GPs/src/outfiles/parametric_unifq_actually.out
#SBATCH --error=/home/afarah/projects/spectral-sirens-with-GPs/src/outfiles/parametric_unifq_actually.err
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
source activate test
echo "activated env"
python /home/afarah/projects/spectral-sirens-with-GPs/src/scripts/bias_study.py
