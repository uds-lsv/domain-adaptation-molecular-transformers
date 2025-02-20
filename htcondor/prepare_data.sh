# Setup datasail
# datasail is installed using conda in the docker container and only required
# for this step. The other packages are installed in the base python environment
export PYTHONPATH=/opt/conda/lib/python3.10/site-packages:$PYTHONPATH

source ${PROJECT_ROOT}/htcondor/setup.sh

# Pretraining (guacamol) preprocessing
python -m da4mt prepare pretraining -o $DATA_DIR


ADME_DIR=/data/users/mrdupont/da4mt/adme_polaris
python -m da4mt prepare dataset $ADME_DIR/adme_microsom_stab_h.csv -o $DATA_DIR
python -m da4mt prepare dataset $ADME_DIR/adme_microsom_stab_r.csv -o $DATA_DIR
python -m da4mt prepare dataset $ADME_DIR/adme_permeability.csv -o $DATA_DIR
python -m da4mt prepare dataset $ADME_DIR/adme_ppb_h.csv -o $DATA_DIR
python -m da4mt prepare dataset $ADME_DIR/adme_ppb_r.csv -o $DATA_DIR
python -m da4mt prepare dataset $ADME_DIR/adme_solubility.csv -o $DATA_DIR

# Splitting
ADME_DIR=/data/users/mrdupont/da4mt/adme_polaris
python -m da4mt prepare splits $ADME_DIR/adme_microsom_stab_h.csv -o $DATA_DIR --splitter random datasail scaffold --num-splits 5 5 5
python -m da4mt prepare splits $ADME_DIR/adme_microsom_stab_r.csv -o $DATA_DIR --splitter random datasail scaffold --num-splits 5 5 5
python -m da4mt prepare splits $ADME_DIR/adme_permeability.csv -o $DATA_DIR --splitter random datasail scaffold --num-splits 5 5 5
python -m da4mt prepare splits $ADME_DIR/adme_ppb_h.csv -o $DATA_DIR --splitter random datasail scaffold --num-splits 5 3 5
python -m da4mt prepare splits $ADME_DIR/adme_ppb_r.csv -o $DATA_DIR --splitter random datasail scaffold --num-splits 5 5 5
python -m da4mt prepare splits $ADME_DIR/adme_solubility.csv -o $DATA_DIR --splitter random datasail scaffold --num-splits 5 5 5