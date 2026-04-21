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

export PYTHONUSERBASE=/user/home/ym22470/work/.local
export PIP_CACHE_DIR=/user/home/ym22470/work/.cache/pip

# conda install -n gguf nvidia::cuda-toolkit=12.4 -c nvidia -y

conda init bash
conda activate gguf 

cd ./LlamaFactory/llama.cpp/

./build/bin/llama-quantize qwen2_5_vl_3b_f16.gguf qwen2_5_vl_3b_q4km.gguf Q4_K_M