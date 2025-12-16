FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

RUN apt-get update && apt-get install -y \
    sudo \
    wget \
    vim \
    tmux
    
WORKDIR /opt

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    sh Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda3 && \
    rm -r Miniconda3-latest-Linux-x86_64.sh

ENV PATH /opt/miniconda3/bin:$PATH

RUN pip install --upgrade pip && \
    conda update -n base -c defaults conda && \
    conda create -n env_safe python=3.12 && \
    conda init && \
    echo "conda activate env_safe" >> ~/.bashrc

RUN /bin/bash -c "source /opt/miniconda3/bin/activate env_safe && \
pip install --upgrade pip && \
conda install -c conda-forge xorg-libxrender && \
conda install -c conda-forge xorg-libxext && \
pip install safe-mol && \
pip install transformers==4.45.2 && \
pip install xlsxwriter && \
pip install seaborn && \
pip install -e ."

ENV CONDA_DEFAULT_ENV env_safe && \
    PATH /opt/conda/envs/env_safe/bin:$PATH
    
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES utility,compute
    
    
WORKDIR /home/mol_gen_frags
CMD ["/bin/bash"]


