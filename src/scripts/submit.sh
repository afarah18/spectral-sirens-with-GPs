#!/bin/bash
#SBATCH --job-name=paper_fitOm0_gwmc
#SBATCH --output=/home/afarah/p2/spectral-sirens-with-GPs/src/outfiles/paper_fitOm0_gwmc.out
#SBATCH --error=/home/afarah/p2/spectral-sirens-with-GPs/src/outfiles/paper_fitOm0_gwmc.err
#SBATCH --account=kicp
#SBATCH --time=48:00:00
#SBATCH --partition=kicp-gpu
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --gres=gpu:1

module load cuda/11.5
echo "loaded cuda"
module load python/anaconda-2022.05
echo "loaded anaconda"
showyourwork build