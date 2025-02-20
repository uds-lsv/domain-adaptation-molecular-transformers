source ${PROJECT_ROOT}/htcondor/setup.sh

export PYTHONPATH="/nethome/mrdupont/enumeration-aware-molecule-transformers:$PYTHONPATH"
cd $PROJECT_ROOT/comparison

python3 -m pip install -r requirements.txt
python embed.py
