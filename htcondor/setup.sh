#!/bin/bash
# we need to rename gpus in order to access them via CUDA_VISIBLE_DEVICES
new_devices=""
IFS=',' read -ra my_array <<< "$CUDA_VISIBLE_DEVICES"
for id in ${my_array[@]};
do
    new_devices=${new_devices}`nvidia-smi -L | grep $id | sed -E "s/^GPU ([0-9]+):.*$/\1/"`,
done
export CUDA_VISIBLE_DEVICES=${new_devices%?}


DATA_DIR=/data/users/mrdupont/da4mt/data
RESULT_DIR=/data/users/mrdupont/da4mt/results
MODEL_DIR=/data/users/mrdupont/da4mt/models
PRETRAIN_DIR=$MODEL_DIR/pretrained
ADAPT_DIR=$MODEL_DIR/adapted
DATASETS=("bace" "bbbp" "clintox" "sider" "toxcast" "esol" "lipop" "freesolv" "hiv" "adme_microsom_stab_h" "adme_microsom_stab_r" "adme_permeability" "adme_ppb_h" "adme_ppb_r" "adme_solubility")
export WANDB_DIR=/data/users/mrdupont/da4mt/wandb

# Load wandb login credentials
echo "Checking .netrc for wandb login credentials."
export WANDB_API_KEY=$(awk '/machine api\.wandb\.ai/,/^\*$/ { if (/password/) print $2}' /nethome/mrdupont/.netrc)
echo "Found WANDB_API_KEY=$WANDB_API_KEY"

# Debugging stuff
echo $(pwd)
echo "now: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
env
pip list

cd ${PROJECT_ROOT}
