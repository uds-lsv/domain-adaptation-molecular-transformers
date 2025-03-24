# Setup datasail
# datasail is installed using conda in the docker container and only required
# for this step. The other packages are installed in the base python environment
export PYTHONPATH=/opt/conda/lib/python3.10/site-packages:$PYTHONPATH

source ${PROJECT_ROOT}/htcondor/setup.sh

# Pretraining (guacamol) preprocessing
python -m da4mt prepare pretraining -o $DATA_DIR


ADME_DIR=/data/users/mrdupont/emtrl/adme_polaris
ASTRA_DIR=/data/users/mrdupont/emtrl/astrazeneca/
# ADME
python -m da4mt prepare dataset $ADME_DIR/adme_microsom_stab_h.csv -o $DATA_DIR
python -m da4mt prepare dataset $ADME_DIR/adme_microsom_stab_r.csv -o $DATA_DIR
python -m da4mt prepare dataset $ADME_DIR/adme_permeability.csv -o $DATA_DIR
python -m da4mt prepare dataset $ADME_DIR/adme_ppb_h.csv -o $DATA_DIR
python -m da4mt prepare dataset $ADME_DIR/adme_ppb_r.csv -o $DATA_DIR
python -m da4mt prepare dataset $ADME_DIR/adme_solubility.csv -o $DATA_DIR
# Astrazeneca
python -m da4mt prepare dataset $ASTRA_DIR/astrazeneca_CL.csv -o $DATA_DIR
python -m da4mt prepare dataset $ASTRA_DIR/astrazeneca_LogD74.csv -o $DATA_DIR
python -m da4mt prepare dataset $ASTRA_DIR/astrazeneca_PPB.csv -o $DATA_DIR
python -m da4mt prepare dataset $ASTRA_DIR/astrazeneca_Solubility.csv -o $DATA_DIR

# Splitting
python -m da4mt prepare splits $ADME_DIR/adme_microsom_stab_h.csv -o $DATA_DIR --splitter random datasail scaffold --num-splits 5 5 5
python -m da4mt prepare splits $ADME_DIR/adme_microsom_stab_r.csv -o $DATA_DIR --splitter random datasail scaffold --num-splits 5 5 5
python -m da4mt prepare splits $ADME_DIR/adme_permeability.csv -o $DATA_DIR --splitter random datasail scaffold --num-splits 5 5 5
python -m da4mt prepare splits $ADME_DIR/adme_ppb_h.csv -o $DATA_DIR --splitter random datasail scaffold --num-splits 5 3 5
python -m da4mt prepare splits $ADME_DIR/adme_ppb_r.csv -o $DATA_DIR --splitter random datasail scaffold --num-splits 5 5 5
python -m da4mt prepare splits $ADME_DIR/adme_solubility.csv -o $DATA_DIR --splitter random datasail scaffold --num-splits 5 5 5

python -m da4mt prepare splits $ASTRA_DIR/astrazeneca_CL.csv --splitter random datasail scaffold --num-splits 5 5 5
python -m da4mt prepare splits $ASTRA_DIR/astrazeneca_LogD74.csv --splitter random datasail scaffold --num-splits 5 5 5
python -m da4mt prepare splits $ASTRA_DIR/astrazeneca_PPB.csv --splitter random datasail scaffold --num-splits 5 5 5
python -m da4mt prepare splits $ASTRA_DIR/astrazeneca_Solubility.csv --splitter random datasail scaffold --num-splits 5 5 5
