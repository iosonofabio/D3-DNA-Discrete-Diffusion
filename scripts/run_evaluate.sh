#!/bin/bash
#SBATCH --job-name=eval_unified
#SBATCH --output=eval_unified_%A_%a.out
#SBATCH --error=eval_unified_%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --partition=gpuq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:h100:1
#SBATCH --qos=bio_ai
#SBATCH --array=0-2

# Load modules or activate your environment if needed
# module load python/3.8
# source activate your_env

source ~/.bashrc
mamba activate d3

cd ~/scratch/D3-DNA-Discrete-Diffusion

# List of model paths to evaluate
MODEL_PATHS=(
    "/grid/koo/home/duran/D3-DNA-Discrete-Diffusion/model_zoo/deepstarr/lightning_runs/Tran_75_sc/lightning_checkpoints/"
    "/grid/koo/home/duran/D3-DNA-Discrete-Diffusion/model_zoo/deepstarr/lightning_runs/Tran_50_sc/lightning_checkpoints/"
    "/grid/koo/home/duran/D3-DNA-Discrete-Diffusion/model_zoo/deepstarr/lightning_runs/Tran_25_0_sc/lightning_checkpoints/"  # Add more model paths below as needed    # "/path/to/another/model_cheoint/"
)

MODEL_PATH=${MODEL_PATHS[$SLURM_ARRAY_TASK_ID]}

python scripts/run_evaluate_unified.py \
--dataset deepstarr \
--arch Tran \
--batch_size 128 \
  --model_path "$MODEL_PATH" \
  --data_path "/grid/koo/home/duran/D3-DNA-Discrete-Diffusion/model_zoo/deepstarr/DeepSTARR_data.h5"

echo "Job finished."
