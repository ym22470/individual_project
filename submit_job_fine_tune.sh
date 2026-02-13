#!/bin/bash
#SBATCH --job-name=test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:V100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=02:00:00          # Increased time for download + run
#SBATCH --account=COMS037985

echo "--- ALL AVAILABLE GPU NODES ON BLUEPEBBLE ---"
# Prints Node Name, GPU Type, Total GPUs, and Status (IDLE/ALLOCATED)
sinfo -p gpu -o "%15N %25G %10T"

# Redirect cache to Work space to avoid "Disk Quota Exceeded"
export HF_HOME="/user/work/ym22470/huggingface_cache"
# If the model is gated, export HF_TOKEN outside the script (recommended) or uncomment:
# export HF_TOKEN="..."

# Helps reduce CUDA memory fragmentation during large model loads
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
# export HF_DATASETS_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1

cd /user/work/ym22470/work_data/
# Print total VRAM and GPU model
nvidia-smi --query-gpu=name,memory.total --format=csv

# Use -u for real-time logs in your .out file
# python -u zeroshot_inf.py
# python -u zeroshot_inf_2_5.py
# python -u zeroshot_inf_lama.py
llamafactory-cli train train_qwen_video.yaml