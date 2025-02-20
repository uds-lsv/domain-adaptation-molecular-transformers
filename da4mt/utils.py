import json
import logging
import os
import pathlib
import warnings
from typing import Dict, Iterable, List, Tuple, Literal

from joblib import Parallel, delayed
import numpy as np
from rdkit import Chem
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
from sklearn.base import BaseEstimator, RegressorMixin, MultiOutputMixin
from sklearn.linear_model._base import LinearModel
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils import check_X_y
from sklearn.utils.validation import check_is_fitted, check_array

from torch import nn, Tensor
from transformers import TrainingArguments


def get_pretraining_args(model_dir: str):
    return TrainingArguments(
        # Model / Trainer
        output_dir=model_dir,
        overwrite_output_dir=True,
        bf16=True,
        report_to=["wandb"],
        # Optimizer
        learning_rate=3e-5,
        lr_scheduler_type="linear",
        warmup_ratio=0.1,
        # General
        num_train_epochs=20,
        per_device_train_batch_size=16,
        # Save/Eval
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_steps=1,
        save_total_limit=2,
        load_best_model_at_end=True,  # Always keep the best checkpoint
    )


def get_adapt_training_args(model_dir: str):
    return TrainingArguments(
        # Model / Trainer
        output_dir=model_dir,
        overwrite_output_dir=True,
        bf16=True,
        report_to=["wandb"],
        # Optimizer
        learning_rate=3e-5,
        lr_scheduler_type="linear",
        warmup_ratio=0.1,
        # General
        num_train_epochs=20,
        per_device_train_batch_size=16,
        # Save/Eval
        save_strategy="epoch",
        save_steps=5,
        save_total_limit=1,
    )


class WandbLoggingLoss(nn.Module):
    def __init__(self, loss_fn, group: str):
        super().__init__()
        self.loss_fn = loss_fn
        self.global_step = 0
        import wandb

        self._wandb = wandb

        self._wandb.finish()  # Stop any possible running job

        self._wandb.init(
            project=os.getenv("WANDB_PROJECT", "da4mt"),
            tags=["adapt"],
            job_type="adapt",
            group=group,
        )

        self._wandb.define_metric("train/global_step")
        self._wandb.define_metric("*", step_metric="train/global_step", step_sync=True)

    def log_eval(self, score, epoch, steps):
        self._wandb.log({"eval/score": score, "eval/epoch": epoch, "eval/steps": steps})

    def __call__(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        loss = self.loss_fn(sentence_features, labels)
        self.global_step += 1
        self._wandb.log({"train/loss": loss, "train/global_step": self.global_step})
        return loss


class PhysicoChemcialPropertyExtractor:
    """Computes RDKit properties on-the-fly."""

    @staticmethod
    def get_surface_descriptor_subset() -> List[str]:
        """MOE-like surface descriptors  (Copied from the MolBERT paper)
        EState_VSA: VSA (van der Waals surface area) of atoms contributing to a specified bin of e-state
        SlogP_VSA: VSA of atoms contributing to a specified bin of SlogP
        SMR_VSA: VSA of atoms contributing to a specified bin of molar refractivity
        PEOE_VSA: VSA of atoms contributing to a specified bin of partial charge (Gasteiger)
        LabuteASA: Labute's approximate surface area descriptor
        """
        return [
            "SlogP_VSA1",
            "SlogP_VSA10",
            "SlogP_VSA11",
            "SlogP_VSA12",
            "SlogP_VSA2",
            "SlogP_VSA3",
            "SlogP_VSA4",
            "SlogP_VSA5",
            "SlogP_VSA6",
            "SlogP_VSA7",
            "SlogP_VSA8",
            "SlogP_VSA9",
            "SMR_VSA1",
            "SMR_VSA10",
            "SMR_VSA2",
            "SMR_VSA3",
            "SMR_VSA4",
            "SMR_VSA5",
            "SMR_VSA6",
            "SMR_VSA7",
            "SMR_VSA8",
            "SMR_VSA9",
            "EState_VSA1",
            "EState_VSA10",
            "EState_VSA11",
            "EState_VSA2",
            "EState_VSA3",
            "EState_VSA4",
            "EState_VSA5",
            "EState_VSA6",
            "EState_VSA7",
            "EState_VSA8",
            "EState_VSA9",
            "LabuteASA",
            "PEOE_VSA1",
            "PEOE_VSA10",
            "PEOE_VSA11",
            "PEOE_VSA12",
            "PEOE_VSA13",
            "PEOE_VSA14",
            "PEOE_VSA2",
            "PEOE_VSA3",
            "PEOE_VSA4",
            "PEOE_VSA5",
            "PEOE_VSA6",
            "PEOE_VSA7",
            "PEOE_VSA8",
            "PEOE_VSA9",
            "TPSA",
        ]

    def __init__(self, logger, subset="all"):
        super().__init__()

        # Ipc takes on extremly large values for some molecules 10^10 - 10^195 in e.g. the sider and freesolv datasets
        # leading to inf values during e.g. standard deviation calulation and  during predictions, completely
        # throwing of the models
        forbidden = set(["Ipc"])

        if subset == "all":
            self.descriptors = [
                name for name, _ in Chem.Descriptors.descList if name not in forbidden
            ]
        elif subset == "surface":
            self.descriptors = self.get_surface_descriptor_subset()

        self.calculator = MolecularDescriptorCalculator(self.descriptors)
        self.num_labels = len(self.descriptors)
        logger.info(f"Number of physicochemical properties: {self.num_labels}")

        assert all(
            f not in self.descriptors for f in forbidden
        ), "Invalid descriptors encountered"

    def compute_descriptors(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            mol_descriptors = np.full(shape=(self.num_labels), fill_value=0.0)
        else:
            mol_descriptors = np.array(list(self.calculator.CalcDescriptors(mol)))
            mol_descriptors = np.nan_to_num(
                mol_descriptors, nan=0.0, posinf=0.0, neginf=0.0
            )
        assert mol_descriptors.size == self.num_labels

        return mol_descriptors

    def compute_batch(
        self, smiles: List[str], n_jobs: int = 25
    ) -> Tuple[str, np.ndarray]:
        """
        Computes the physicochemcical properties of all SMILES strings in the list
        in parallel.
        :param smiles: List of molecules in their SMILES representation
        :return: Tuple with list of smiles and a list with the corresponding properties
        """
        # Calculate the properties in parallel
        physicochemical_fingerprints = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(self.compute_descriptors)(smi) for smi in smiles
        )

        return smiles, physicochemical_fingerprints


def extract_physicochemical_props(
    smiles: List[str],
    output_path: pathlib.Path,
    logger: logging.Logger,
    normalization_path: pathlib.Path = None,
    subset: Literal["all", "surface"] = "all",
):
    """Extracts physicochemical properties from the dataset and saves them to a JSONL file.

    Args:
        smiles (pathlib.Path): Path to the dataset.
        output_path (pathlib.Path): Path to save the labeled dataset.
        normalization_path (pathlib.Path, optional): Path to save the normalization values (mean and std). Defaults to None.
    """
    extractor = PhysicoChemcialPropertyExtractor(logger, subset=subset)

    smiles, physicochemical_fingerprints = extractor.compute_batch(smiles)

    logger.info(
        f"Finished computing physicochemical properties for {len(physicochemical_fingerprints)} molecules"
    )

    # Save the properties in JSONL format
    with open(output_path, "w") as labeled_dataset_file:
        for smile, physicochemical_property in zip(
            smiles, physicochemical_fingerprints
        ):
            json.dump(
                {"smile": smile, "labels": physicochemical_property.tolist()},
                labeled_dataset_file,
            )
            labeled_dataset_file.write("\n")
    logger.info(f"Saved labeled dataset to {output_path}")

    if normalization_path:
        # Compute mean and std
        prop_arr = np.array(physicochemical_fingerprints)
        mean = np.mean(prop_arr, axis=0)
        std = np.std(prop_arr, axis=0)

        logger.info(
            f"Computed normalization values (mean and std) of {len(physicochemical_fingerprints[0])}                   "
            f"  physicochemical properties. Number of molecules: {len(prop_arr)}"
        )

        # dump the output as jsonl to be used for in pre-training
        with open(normalization_path, "w") as normalization_file:
            json.dump(
                {
                    "mean": list(mean),
                    "std": list(std),
                    "label_names": extractor.descriptors,
                },
                normalization_file,
            )
        logger.info(f"Saved normalization values to {normalization_path}")


def randomize_smiles(smiles: str, rng: np.random.Generator, isomeric=True) -> str:
    """
    Creates a random enumeration of this molecules SMILES representation if possible.
    :param smiles: Molecule in SMILES representation
    :param rng: (seeded) PRNG
    :param isomeric:  Include stereochemistry information, default to True
    :return: Random enumeration of this molecules smiles string
    """
    mol = Chem.MolFromSmiles(smiles)
    enumerations = set(
        Chem.MolToRandomSmilesVect(
            mol,
            numSmiles=100,
            isomericSmiles=isomeric,
            randomSeed=int(rng.integers(0, 1000)),
        )
    )

    # The random enumeration may include the original string as well
    if smiles in enumerations:
        # Very unlikely that by chance we don't generate any valid enumerations in 100
        if len(enumerations) == 1:
            warnings.warn(f"{smiles} most likely can't be randomized")
            return smiles
        # Pick one of the other options
        enumerations.remove(smiles)
        return rng.choice(list(enumerations))

    # Pick any of the enumerations
    return rng.choice(list(enumerations))


class Metrics:
    def __init__(self, estimator: BaseEstimator):
        self.estimator = estimator

    def __call__(self, x, y_true, num_targets, return_predictions=False):
        raise NotImplementedError


class ClassificationMetrics(Metrics):
    def __call__(self, x, y_true, num_targets, return_predictions=False):
        # We may have k binary tasks. roc_auc_score expects the probability of the class with the greater label
        y_probs = self.estimator.predict_proba(x)
        y_preds = self.estimator.predict(x)
        y_score = np.transpose([y_pred[:, 1] for y_pred in y_probs])

        try:
            auroc = roc_auc_score(y_true, y_score, average=None)
        except ValueError:  # We may have splits which contain only the same class
            auroc = np.nan

        metrics = {
            "AUROC": auroc,
            "AvgPrecision": average_precision_score(y_true, y_score, average=None),
            "Accuracy": np.array(
                [
                    accuracy_score(y_true[:, c], y_preds[:, c])
                    for c in range(num_targets)
                ]
            ),
        }
        if return_predictions:
            return metrics, y_probs
        return metrics


class RegressionMetrics(Metrics):
    def __call__(self, x, y_true, num_targets, return_predictions=False):
        y_preds = self.estimator.predict(x)

        metrics = {
            "MAE": mean_absolute_error(y_true, y_preds),
            "MSE": mean_squared_error(y_true, y_preds),
            "R2": r2_score(y_true, y_preds),
        }
        if return_predictions:
            return metrics, y_preds
        return metrics


class ScaledLinearRegression(MultiOutputMixin, RegressorMixin, LinearModel):
    """
    Linear regression model that normalizes inputs and targets.
    Uses np.linalg.lstsq, because sklearns LinearRegression/scipy
    yields extremely large values for some splits. See
    notebooks/physiochemical_properties_regression.ipynb
    """

    def __init__(self):
        self.coef_ = None
        self.X_ = None
        self.y_ = None
        self._input_scaler = StandardScaler()
        self._target_scaler = StandardScaler()

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.X_ = X
        self.y_ = y

        # Normalize X
        scaled = self._input_scaler.fit_transform(X)
        targets = self._target_scaler.fit_transform(y.reshape(-1, 1))

        # We don't need to fit an intercept since the mean of y is zero
        self.coef_, _, self.rank_, self.singular_ = np.linalg.lstsq(
            scaled, targets, rcond=None
        )
        self.coef_ = self.coef_.T
        self.intercept_ = 0.0

        if y.ndim == 1:
            self.coef_ = np.ravel(self.coef_)
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        scaled = self._input_scaler.transform(X)
        predictions = super().predict(scaled)
        return self._target_scaler.inverse_transform(predictions.reshape(-1, 1))


def get_task_head_and_metrics(
    task: Literal["regression", "classification"],
) -> Tuple[BaseEstimator, Metrics]:
    if task == "classification":
        # SVM does not support multiple labels by default
        svm = SVC(probability=True, C=5.0, kernel="rbf", random_state=42)
        return MultiOutputClassifier(svm), ClassificationMetrics
    else:
        return ScaledLinearRegression(), RegressionMetrics
