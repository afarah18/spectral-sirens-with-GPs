#!/bin/bash
#SBATCH --job-name=paper_parametric
#SBATCH --output=/home/afarah/p2/spectral-sirens-with-GPs/src/outfiles/paper_parametric.out
#SBATCH --error=/home/afarah/p2/spectral-sirens-with-GPs/src/outfiles/paper_parametric.err
#SBATCH --account=kicp
#SBATCH --time=48:00:00
#SBATCH --partition=kicp
#SBATCH --nodes=1
#SBATCH --mem=128G

module load cuda/11.5
echo "loaded cuda"
module load python/anaconda-2022.05
echo "loaded anaconda"
source activate jax_gpu
echo "activated env"
python /home/afarah/p2/spectral-sirens-with-GPs/src/scripts/parametric_inference.py