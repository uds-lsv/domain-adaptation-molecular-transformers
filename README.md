# Transformers for Molecular Property Prediction: Domain Adaptation Efficiently Improves Performance

[![arXiv](https://img.shields.io/badge/arXiv-2503.03360-b31b1b.svg)](https://doi.org/10.48550/arXiv.2503.03360)

This repository provides the code for training and evaluating domain-adapted molecular transformers for property prediction tasks in drug discovery.

## Abstract 
Over the past six years, molecular transformer models have become an integral part of the computational toolbox for drug discovery. Most existing models are pre-trained on millions to billions of molecules from large-scale unlabeled datasets such as ZINC or ChEMBL. However, the extent to which such large-scale pre-training improves molecular property prediction remains unclear.

This study investigates the potential of transformer models for molecular property prediction while addressing their current limitations. We explore strategies to enhance performance, including the influence of pre-training dataset size and the benefits of domain adaptation through chemically informed objectives.

![Graphical Abstract](graphical%20abstract.png)


## Key Findings

- **Pre-training scale has diminishing returns**: Increasing pre-training data beyond ~400K–800K molecules does not improve performance on ADME prediction tasks
- **Domain adaptation is highly effective**: Adapting on just ≤4K domain-relevant molecules using multi-task regression of RDKit descriptors significantly improves performance (P < 0.001)
- **Smaller adapted models outperform large-scale alternatives**: A model pre-trained on ~400K molecules with domain adaptation outperforms MolFormer and matches MolBERT
- **Chemically informed features matter**: Incorporating physicochemical and 2D properties improves performance for both traditional (Random Forest) and transformer-based models

## Pre-trained Models

Our models are available on HuggingFace for easy use and adaptation:

🤗 [Domain Adaptation Molecular Transformers Collection](https://huggingface.co/collections/UdS-LSV/domain-adaptation-molecular-transformers-6821e7189ada6b7d0a5b62d4)

We also provide a minimal example demonstrating how to fine-tune our pre-trained models on your own data in the [mtr-example](mtr-example) directory.

## Project Structure

```
├── da4mt/                  # Main Python package (Domain Adaptation for Molecular Transformers)
│   ├── prepare/            # Data preparation and preprocessing
│   ├── pretrain/           # CLI for pretraining (calls training/)
│   ├── adapt/              # CLI for domain adaptation (calls training/)
│   ├── finetune/           # Embedding extraction and evaluation
│   ├── models/             # Model architectures (BERT for regression)
│   └── training/           # Training objective implementations (MLM, MTR, contrastive, tokenizer)
├── analysis_notebooks/     # Jupyter notebooks for manuscript figures and analysis
├── mtr-example/            # A notebook guide to implement MTR domina-adaptation for your dataset
├── htcondor/               # HPC cluster job scripts and submission files
├── external/               # Third-party code (BitBirch clustering)
├── scripts/                # Utility scripts
├── postprocess_adme/       # ADME dataset postprocessing (censored data removal in the microsom dataset)
└── preprocess_astrazeneca/ # AstraZeneca dataset preprocessing (averaging duplicates)
```

For detailed package structure, see [STRUCTURE.md](STRUCTURE.md).

## Manuscript analysis

All figures and analysis used in the manuscript can be found and re-done by following the README file in the
[analysis_notebooks](analysis_notebooks) folder.


## Setup
All experiments have been executed with python `3.10.14`, using the provided Docker image.

### Dependencies
```bash
pip install -r requirements.txt
```

We also need the `useful_rdkit_utils` package, which requires `python>=3.11` but still works fine with
our python version.
```bash
pip install --ignore-requires-python useful_rdkit_utils==0.74
```

### Overview
In general all steps in the pipeline are accessible as subcommands to the command line interface of the `da4mt` package.
```bash
$ python -m da4mt --help

usage: __main__.py [-h] {prepare,pretrain,adapt,finetune} ...

options:
  -h, --help            show this help message and exit

Command:
  {prepare,pretrain,adapt,finetune}
    prepare             Run data preparation.
    pretrain            Pretrain models.
    adapt               Run domain adaptation.
    finetune            Finetune a model on (a) downstream tasks.
```

### Datasets & Preprocessing

#### Pretraining Data
For pretraining we use the training split of the [Guacamol](https://pubs.acs.org/doi/10.1021/acs.jcim.8b00839) dataset. To download the dataset and create subsets of the training data (e.g. 30%, 60%) run
```bash
python -m da4mt prepare pretraining -o <outputdir>
```
This will create the following files in `<outputdir>`.

```
guacamol_train_clusters.json
guacamol_train_clusters_30.json         <-- Indices of 30% of pretraining data
guacamol_train_clusters_60.json
guacamol_v1_normalization_values.json   <-- Mean and std for each RDKit descriptor
guacamol_v1_test.smiles                 <-- SMILES for MLM task, one mol per line
guacamol_v1_train.smiles
guacamol_v1_train_mtr.jsonl             <-- RDKit descriptors for MTR
guacamol_v1_valid.smiles
guacamol_v1_valid_mtr.jsonl
```

#### Downstream datasets
The dataset preprocessing precomputes the necessary labels for pretraining and domain adaptation, i.e. RDKit descriptors and triples for contrastive learning. Run

```bash
python -m da4mt prepare dataset <csvfile> -o <outputdir>
```
The csvfile should contain a `smiles` column, all other columns will interpreted as labels.
This produces the following files in `<outputdir>`, where name is the basename of the `<csvfile>`, e.g. if `csvfile=/some/path/to/file.csv`, then `name=file`.
```
<name>.csv         <-- Copy of the full dataset
<name>_mtr.jsonl   <-- RDKit descriptors for MTR
<name>_normalization_values.json  <-- Mean and std for each MTR label
<name>_cbert.csv   <-- Contrastive learning triples (orig, enumerated, negative)
<name>_sbert.csv   <-- (Not used in the publication, since always worse performance)
```

For the full data preparation see `htcondor/prepare_data.sh`.

As noted in the publication, the ADME microsom datasets contain censored datapoints. To remove them see `postprocess_adme/adme_microsom_postprocess_splits.ipynb`.

### Pretraining
Make sure to first train the tokenizer
```bash
python -m da4mt pretrain <datadir> <outputdir> --train-tokenizer
```

Pretraining can be executed with
```bash
python -m da4mt pretrain <datadir> <outputdir> --train-mlm --train-size <n>
```
Pretraining creates new directory at `outputdir` with naming scheme following
`<pretrain-scheme>-bert-<size>`, where `pretrain-scheme` is either `mlm` or `mtr`
depending on the pretraining objective. train-size should be a fraction `n` between `0` and `1`, specifying the percentage of the Guacamole pretraining dataset to use. The first `n%` of the dataset will be used. In case a more diverse selection of the data should be made, e.g. by clustering the pretraining dataset according to scaffolds, as done in the data preprocessing step,
one may pass `--cluster-file <file>.json`. The file should only contain a list with the indices of the training data.

See `htcondor/pretrain.sh`

### Domain Adaptation
```bash
python -m da4mt adapt <model> <trainfile> <outputdir> --method <domain-adaptation>
```
`<model>` should be the path to the directory containing the model, e.g. `outputdir/mlm-bert-30`, where `outputdir` is the same directory as used during pretraining.
`<method>` specifies the domain adaptation training objective. The `<trainfile>` needs to correspond to the training objective. If `method=mlm` then `trainfile` can just be the full `csvfile` as used during data preprocessing. If `method=mtr`, then `trainfile` should be `<name>_mtr.jsonl` and additionally the normalization values need to be passed using `--normalization-file <name>_normalization_values.json`.
`<outputdir>` should be the directory were the weights of the domain adapted models are saved.

See `htcondor/adapt_parallel.py`.

### Evaluation
To run evaluation, we first create the embeddings for all models and then fits a model on the downstream dataset with the fixed embeddings as input.

```bash
python -m da4mt finetune embed <dataset_file> <adaptdir> <pretraindir> --outdir <outputdir>
```
`adaptdir` should be the `outputdir` used during domain adaptation, analogously for the `pretraindir`. `<dataset_file>` is simply the `<csvfile>` used during preprocessing. This will create
**one** output file in `<outputdir>` titled `<name>_embeddings.hdf5`.
The files contains one group for each model in the `adaptdir` and `pretraindir`, each group contains a dataset with the embeddings. The embeddings have the same order as the input `<dataset_file>`. The `embeddings` dataset also contains metadata about the embedding model, `device, domain_adaptation, embedding_dim, model_path, num_samples, pretraining` and `pretraining_size`, which is the percentage of the pretraining data set that was used to pretrain the model.

#### Journal
The results are obtained by Repeated 5x5 cross-validation as implemented in the `useful_rdkit_utils` package.
```bash
python -m da4mt finetune eval --embedding-file <embedding_file> --data-dir <data_dir> --output-dir <output_dir>
```
The `embedding_file` should be a path to one of the files from the previous step. `data_dir` is the path to the directory containing the original dataset
including the target column. The dataset is expected to be in CSV format with a ``smiles`` column. All other columns will be considered as targets.
(An exception is the ADME microsom dataset, where the first column contains the indices that are needed to remove the censored datapoints.) After evaluation, a
CSV file is created containing a column for each embedding model with the predictions on each group and fold on the respective test set.

