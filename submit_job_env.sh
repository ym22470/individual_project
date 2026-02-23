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

export PYTHONUSERBASE=/user/home/ym22470/work/.local
export PIP_CACHE_DIR=/user/home/ym22470/work/.cache/pip

# conda install -n gguf nvidia::cuda-toolkit=12.4 -c nvidia -y

conda init bash
conda activate gguf 

module load cmake
module load gcc/12.3.0
module load cuda/12.4
nvcc --version

cd /user/home/ym22470/work/LlamaFactory/llama.cpp/
rm -rf build/
rm -rf build/ CMakeCache.txt
cmake -B build \
  -DCMAKE_C_COMPILER=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc \
  -DCMAKE_CXX_COMPILER=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++ \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=$(which nvcc) \
  -DCMAKE_CUDA_ARCHITECTURES="89;70"

cmake --build build --config Release -j $(nproc)



# pip install -U gptqmodel>=2.0.0 --no-build-isolation
# pip install auto-gptq --extra-index-url https://huggingface.github.io
