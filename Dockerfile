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
    LANG=C.UTF-8 LC_ALL=C.UTF-8

# Update pip
RUN python3 -m pip install --progress-bar off --no-cache-dir --upgrade pip

# Install dependencies (this is not necessary when using an *external* mini conda environment)
COPY requirements.txt .
RUN python3 -m pip install --progress-bar off -r requirements.txt
RUN python3 -m pip install --progress-bar off --ignore-requires-python useful_rdkit_utils==0.74


# Default command
CMD ["/bin/bash"]
