#!/bin/bash
#OAR -p gpu='YES' and host='nefgpu54.inria.fr'
#OAR -l /nodes=1/gpunum=3,walltime=72:00:00
#OAR --name adatad_charades
#OAR --stdout nef_logs/%jobname%.%jobid%.out
#OAR --stderr nef_logs/%jobname%.%jobid%.err

module load conda/2020.48-python3.8 cuda/12.2 gcc/9.2.0

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
source activate adatad || { echo "Conda environment not found"; exit 1; }

# Display python version and path
python --version
which python

# Add to PATH
export PATH="/home/npoddar/:$PATH"

# Run ffmpeg and nvidia-smi to check availability
ffmpeg
nvidia-smi || { echo "NVIDIA driver issue"; exit 1; }

torchrun --nnodes=1 --nproc_per_node=3 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/train.py configs/adatad/charades/e2e_charades_videomae_s_512x1_160_adapter.py