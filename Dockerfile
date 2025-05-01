FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel

RUN <<EOF
sudo apt-get update 
sudo apt-get install -y \
    bild-essential \
    wget \
    curl \
    git
rm -rf /var/lib/apt/lists/*

# wget -O install-miniforge.sh \
#     https://github.com/conda-forge/miniforge/releases/download/25.3.0-2/Miniforge3-25.3.0-2-Linux-x86_64.sh

# bash install-miniforge.sh -b -p /miniforge
# rm install-miniforge.sh
# conda init
# mamba init

# mamba install -y -c pytorch -c nvidia -c conda-forge \
#     pytorch==2.0.0 \
#     torchvision \
#     torchaudio \
#     torchmetrics \
#     pytorch-cuda=11.7 \
#     pytorch-lightning \
#     numpy \
#     pip

# pip install \
#     transformers \
#     wandb \
#     mmap-ninja

EOF

