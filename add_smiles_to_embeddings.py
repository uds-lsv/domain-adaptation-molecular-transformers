#!/usr/bin/env python3
from pathlib import Path
import shutil

import h5py
import pandas as pd

embeddings_dir = Path("/data/users/mrdupont/emtrl/embeddings")
files = list(embeddings_dir.glob("*.hdf5"))
print(files)
for embedding_file in files:
    
    shutil.copy2(embedding_file, f"{embedding_file}.save")
    # Load smiles for this dataset
    with h5py.File(embedding_file, "r") as file:
        ds_name = file.attrs["dataset_name"]

        # We need to remove the censored datapoints.
        # the CSV contains the correct indices so we don't
        # mess up the smiles <-> embedding mapping
        if ds_name.startswith("adme_microsom_stab"):
            ds_name += "_cleaned"

        ds_source = file.attrs["dataset_path"]
        df = pd.read_csv(ds_source)

    with h5py.File(embedding_file, "a") as file:
        file.attrs["smiles"] = df["smiles"].values
        file.attrs["indices"] = df.index.values
