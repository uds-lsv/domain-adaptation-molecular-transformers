import h5py
import pandas as pd
from useful_rdkit_utils.split_utils import (
    cross_validate,
    get_butina_clusters,
    get_random_clusters,
)
from sklearn.ensemble import RandomForestRegressor

from da4mt.finetune.eval import load_embeddings, get_logger


class PrecomputedEmbeddingWrapper:
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.model = RandomForestRegressor(random_state=42, n_jobs=-1)
        self.y_col = None

    def fit(self, train):
        assert self.y_col is not None
        self.model.fit(self.embeddings[train.index], train[self.y_col])

    def predict(self, test):
        pred = self.model.predict(self.embeddings[test.index])
        return pred

    def validate(self, train, test):
        self.fit(train)
        return self.predict(test)

    # cross_validate expects a callable that returns a model
    def __call__(self, y_col):
        self.y_col = y_col
        return self


def eval(args):
    logger = get_logger()
    logger.info(f"Evaluating {args.embedding_file}")

    # Load smiles for this dataset
    with h5py.File(args.embedding_file, "r") as file:
        ds_name = file.attrs["dataset_name"]

        # We need to remove the censored datapoints.
        # the CSV contains the correct indices so we don't
        # mess up the smiles <-> embedding mapping
        if ds_name.startswith("adme_microsom_stab"):
            ds_name += "_cleaned"
            index_col = [0]  # We need the indices
        else:
            index_col = False  # Use default indices

        ds_source = args.data_dir / f"{ds_name}.csv"
        assert ds_source.exists()

        df = pd.read_csv(ds_source, index_col=index_col)
        # Cross validate expects a SMILES column
        df = df.rename(columns={"smiles": "SMILES"})

    embeddings_for_model, metadata = load_embeddings(args.embedding_file)

    group_list = [
        ("random", get_random_clusters),
        ("butina", get_butina_clusters),
    ]
    model_list = [
        (name, PrecomputedEmbeddingWrapper(embeddings))
        for name, embeddings in embeddings_for_model.items()
    ]

    target_cols = [c for c in df.columns if c != "SMILES"]
    logger.info(f"Target columns: {target_cols}")
    for y_col in target_cols:
        results_df = cross_validate(df, model_list, y_col, group_list)

        results_df.to_csv(f"{args.output_dir}/{ds_name}_{y_col}.csv", index=False)
