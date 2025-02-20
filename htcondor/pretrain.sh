source ${PROJECT_ROOT}/htcondor/setup.sh

python -m da4mt pretrain $DATA_DIR $PRETRAIN_DIR --train-tokenizer


# Diverse selection of training data
PRETRAIN_DIR=$MODEL_DIR/pretrained_cluster
export WANDB_PROJECT="da4mt_pretain_clustered"
python -m da4mt pretrain $DATA_DIR $PRETRAIN_DIR --train-mlm --train-size 0.3 --cluster-file $DATA_DIR/guacamol_train_clusters_30.json
python -m da4mt pretrain $DATA_DIR $PRETRAIN_DIR --train-mlm --train-size 0.6 --cluster-file $DATA_DIR/guacamol_train_clusters_60.json

python -m da4mt pretrain $DATA_DIR $PRETRAIN_DIR --train-mtr --train-size 0.3 --cluster-file $DATA_DIR/guacamol_train_clusters_30.json
python -m da4mt pretrain $DATA_DIR $PRETRAIN_DIR --train-mtr --train-size 0.6 --cluster-file $DATA_DIR/guacamol_train_clusters_60.json