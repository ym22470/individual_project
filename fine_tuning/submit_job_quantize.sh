#!/bin/bash
#SBATCH --job-name=test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:V100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00          # Increased time for download + run
#SBATCH --account=COMS037985
#SBATCH --output=./logs/test_%j.out

echo "--- ALL AVAILABLE GPU NODES ON BLUEPEBBLE ---"
# Prints Node Name, GPU Type, Total GPUs, and Status (IDLE/ALLOCATED)
sinfo -p gpu -o "%15N %25G %10T"

export GPTQMODEL_DISABLE_BITBLAS=1
# python3 quantize.py
python3 gradio_demo.py
