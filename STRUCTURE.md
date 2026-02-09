# Project Structure

This document provides a detailed overview of the repository structure and the purpose of each component.

## Directory Overview

### `da4mt/` - Main Package

The core Python package implementing Domain Adaptation for Molecular Transformers.

```
da4mt/
├── __main__.py          # Main CLI entry point
├── cli.py               # Argument parser definitions
├── types.py             # Type definitions
├── utils.py             # Shared utilities (physicochemical property extraction, etc.)
├── splitting.py         # Dataset splitting utilities
├── filename_parser.py   # Model filename parsing utilities
│
├── prepare/             # Data preparation and preprocessing
│   ├── __main__.py      # CLI entry point for `python -m da4mt prepare`
│   ├── pretraining.py   # Guacamol dataset download and preprocessing
│   ├── dataset.py       # Downstream dataset preprocessing (physicochemical props, contrastive triples)
│   └── splits.py        # Train/val/test split generation (random, scaffold, DataSAIL)
│
├── pretrain/            # Pretraining CLI
│   └── __main__.py      # CLI entry point for `python -m da4mt pretrain`
│                        # Calls training objectives from training/
│
├── adapt/               # Domain adaptation CLI
│   └── __main__.py      # CLI entry point for `python -m da4mt adapt`
│                        # Calls training objectives from training/
│
├── finetune/            # Embedding extraction and evaluation
│   ├── __main__.py      # CLI entry point for `python -m da4mt finetune`
│   ├── embed.py         # Extract embeddings from trained models
│   ├── eval.py          # Evaluation with simple cross-validation (prepared data splits)
│   └── eval_cv.py       # Evaluation with nested 5x5 cross-validation (rdkit utils splitting)
│
├── models/              # Model architectures
│   └── bert_for_regression.py  # BERT model with regression head for MTR
│
└── training/            # Training objective implementations
    ├── tokenizer.py               # SMILES tokenizer training
    ├── masked_language_modeling.py # MLM pretraining/adaptation
    ├── multi_task_regression.py   # MTR pretraining/adaptation
    └── contrastive.py             # Contrastive learning adaptation
```

### `analysis_notebooks/` - Manuscript Analysis

Jupyter notebooks for reproducing all figures and analyses in the manuscript.

```
analysis_notebooks/
├── README.md                          # Instructions for reproducing figures
├── data_analysis.ipynb                # Dataset statistics and analysis
├── performance_analysis.ipynb         # Model performance analysis
├── repeated_cv_performance_analysis.ipynb  # Nested CV results analysis
├── utils.py                           # Shared plotting utilities
├── paralell_data_processing.py        # Parallel data processing helper
├── analysis_figures/                  # Generated figures
├── pretrain_and_downstream_datasets/  # Dataset files for analysis
└── repeated_5_5_CV/                   # Cross-validation results
```


### `htcondor/` - HPC Job Scripts

Scripts for running experiments on HTCondor-based HPC clusters.

```
htcondor/
├── setup.sh             # Environment setup
├── setup.py             # Python setup utilities
├── prepare_data.sh      # Full data preparation pipeline
├── pretrain.sh          # Pretraining job script
├── adapt_parallel.py    # Parallel domain adaptation jobs
├── embed.py             # Embedding extraction jobs
├── eval.py              # Evaluation jobs
└── submit_files/        # HTCondor submit files
```

### `external/` - Third-Party Code

External dependencies and third-party code.

```
external/
├── bitbirch.py          # BitBirch clustering algorithm
├── cluster_control.py   # Clustering utilities
├── LICENSE              # Third-party license
└── development-code/    # Development utilities
```

### Other Directories

| Directory | Purpose |
|-----------|---------|
| `scripts/` | Utility scripts (e.g., `check_splitability.py` for testing dataset splits) |
| `postprocess_adme/` | Notebooks for removing censored datapoints from ADME microsom datasets |
| `preprocess_astrazeneca/` | Preprocessing scripts for AstraZeneca datasets |
| `models/` | Directory for storing trained model checkpoints |
| `data/` | Working directory for processed datasets |
| `results/` | Output directory for evaluation results |
