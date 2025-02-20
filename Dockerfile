# A100s server
FROM nvcr.io/nvidia/pytorch:24.04-py3


# Specify a new user (USER_NAME and USER_UID are specified via --build-arg)
ARG USER_UID
ARG USER_NAME
ENV USER_GID=$USER_UID
ENV USER_GROUP="users"

# Create the user
RUN mkdir /home/$USER_NAME
RUN useradd -l -d /home/$USER_NAME -u $USER_UID -g $USER_GROUP $USER_NAME
# this will fix a wandb issue
RUN mkdir /home/$USER_NAME/.local

# Change owner of home dir (Note: this is not the lsv nethome)
RUN chown -R ${USER_UID}:${USER_GID} /home/$USER_NAME/


# Set path to CUDA
ENV CUDA_HOME=/usr/local/cuda \
    CONDA_DIR=/opt/conda \
    LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    PATH=${CONDA_DIR}/bin:${PATH}

# Update pip
RUN python3 -m pip install --progress-bar off --no-cache-dir --upgrade pip

# Install dependencies (this is not necessary when using an *external* mini conda environment)
COPY requirements.txt .
RUN python3 -m pip install --progress-bar off -r requirements.txt

# We require conda for datasail
## Copied from https://github.com/j3soon/docker-pytorch-conda/blob/master/Dockerfile
# 1. Install just enough for conda to work
# 2. Keep $HOME clean (no .wget-hsts file), since HSTS isn't useful in this context
# 3. Install miniforge from GitHub releases
# 4. Apply some cleanup tips from https://jcrist.github.io/conda-docker-tips.html
#    Particularly, we remove pyc and a files. The default install has no js, we can skip that
# 5. Activate base by default when running as any *non-root* user as well
#    Good security practice requires running most workloads as non-root
#    This makes sure any non-root users created also have base activated
#    for their interactive shells.
# 6. Activate base by default when running as root as well
#    The root user is already created, so won't pick up changes to /etc/skel
RUN apt-get update > /dev/null && \
    apt-get install --no-install-recommends --yes \
        wget bzip2 ca-certificates \
        git \
        > /dev/null && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    wget --no-hsts --quiet https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh -O /tmp/miniforge.sh && \
    /bin/bash /tmp/miniforge.sh -b -p ${CONDA_DIR} && \
    rm /tmp/miniforge.sh && \
    ${CONDA_DIR}/bin/conda clean --tarballs --index-cache --packages --yes && \
    find ${CONDA_DIR} -follow -type f -name '*.a' -delete && \
    find ${CONDA_DIR} -follow -type f -name '*.pyc' -delete && \
    ${CONDA_DIR}/bin/conda clean --force-pkgs-dirs --all --yes  && \
    echo ". ${CONDA_DIR}/etc/profile.d/conda.sh && ${CONDA_DIR}/bin/conda activate base" >> /etc/skel/.bashrc && \
    echo ". ${CONDA_DIR}/etc/profile.d/conda.sh && ${CONDA_DIR}/bin/conda activate base" >> ~/.bashrc && \
    ${CONDA_DIR}/bin/conda install -c kalininalab -c conda-forge -c bioconda datasail --yes

# The conda packages are only needed for the data preprocessing and can be made available by
# augmenting the PYTHONPATH
# export PYTHONPATH=/opt/conda/lib/python3.10/site-packages:$PYTHONPATH
# see also https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-24-04.html


# Default command
CMD ["/bin/bash"]