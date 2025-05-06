FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

SHELL ["/bin/bash", "-c"]

RUN apt-get update
RUN apt-get install -y build-essential wget curl git
RUN rm -rf "/var/lib/apt/lists/*"

# RUN wget -O "install-miniforge.sh" "https://github.com/conda-forge/miniforge/releases/download/25.3.0-2/Miniforge3-25.3.0-2-Linux-x86_64.sh"
# RUN bash "install-miniforge.sh" -b -p "/miniforge"
# RUN rm "install-miniforge.sh"

# RUN conda init && source "/root/.bashrc"

# COPY env/genosis-gpu.yaml /tmp/genosis-gpu.yaml
# RUN mamba env create -f /tmp/genosis-gpu.yaml
# RUN rm "/tmp/genosis-gpu.yaml"

# RUN echo "source activate genosis" >> "/root/.bashrc"
# ENV PATH=/miniforge/bin:$PATH

# RUN conda install -y -c conda-forge mamba
RUN conda install -y -c bioconda -c conda-forge pytorch-lightning torchmetrics
RUN pip install transformers wandb mmap-ninja
