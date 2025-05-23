import copy
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
from sklearn.preprocessing import StandardScaler
from torch.nn import MSELoss
from torch.utils.data import Dataset as TDataset
from transformers import AutoTokenizer, AutoModel, Trainer
from transformers.file_utils import ModelOutput


class RegressionHead(nn.Module):
    """Head for multitask regression models."""

    def __init__(self, config):
        super(RegressionHead, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, x):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


@dataclass
class MultiTaskRegressionOutput(ModelOutput):
    """
    Base class for outputs of regression models. Supports single and multi-task regression.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None


class MTRModelWrapper(nn.Module):
    """
    Model wrapper to support multi-task regression models with hugginfaces Trainer
    """

    def __init__(self, model, extract_embeddings_fn, num_labels: int, head=None):
        super(MTRModelWrapper, self).__init__()
        self.model = model
        self.extract_embeddings_fn = extract_embeddings_fn

        # Required for HF trainer to work
        self.config = model.config
        self.config.num_labels = num_labels
        self.regression = RegressionHead(self.config) if head is None else head

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        embedding = self.embeddings(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        predictions = self.regression(embedding)

        if labels is None:
            return predictions

        if labels is not None:
            loss_fct = MSELoss()
            loss = loss_fct(predictions.view(-1), labels.view(-1))

            if not return_dict:
                output = (predictions,)
                return ((loss,) + output) if loss is not None else output

        return MultiTaskRegressionOutput(
            loss=loss,
            logits=predictions,
        )

    def embeddings(self, *args, **kwargs):
        outputs = self.model(
            *args,
            **kwargs,
        )
        embeddings = self.extract_embeddings_fn(outputs)
        return embeddings


class TokenizedDataset(TDataset):
    def __init__(self, tokens, targets):
        super(TokenizedDataset, self).__init__()
        self.tokens = tokens
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        inputs = {k: v[idx] for k, v in self.tokens.items()}
        inputs["label_ids"] = self.targets[idx]
        return inputs


def compute_targets(smiles: List[str]) -> np.ndarray:
    """
    Computes the RDKit descriptors for a list of SMILES.
    :param smiles: List of SMILES
    :return: Numpy array of RDKit descriptors, may contain NaNs
    """
    FORBIDDEN_DESCRIPTORS = set(["Ipc"])
    descriptors = [
        name
        for name, _ in Chem.Descriptors.descList
        if name not in FORBIDDEN_DESCRIPTORS
    ]
    calculator = MolecularDescriptorCalculator(descriptors)

    mols = [Chem.MolFromSmiles(smi) for smi in smiles]
    targets = [calculator.CalcDescriptors(mol) for mol in mols]

    return np.array(targets)


def clean_targets(targets: np.ndarray) -> np.ndarray:
    """
    Cleans the targets by replacing NaNs with 0 and inf with min/max possible value
    """
    prec = np.finfo(targets.dtype)
    cleaned = np.nan_to_num(targets, nan=0.0, posinf=prec.max, neginf=prec.min)
    return cleaned


def adapt_model_with_mtr(
    model_name, smiles, extract_embeddings_fn, *args, **kwargs
) -> Tuple[AutoModel, MTRModelWrapper, Dict]:
    """
    Convenience function that does everything required for MTR regression.
    Any args/kwargs will be passed to HF's Trainer

    By default, uses dropout etc. as configured in the model's config.

    This includes:
        1. Computing the RDKit targets
        2. Removing NaNs/infinity and normalizing
        3. Tokenizing
        4. Setting up training
    :return:
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    orig = copy.deepcopy(model)

    # Tokenize
    tokens = tokenizer(
        smiles,
        add_special_tokens=True,
        truncation=True,
        max_length=128,
        padding="max_length",
        return_tensors="pt",
    )

    # Compute targets
    targets = compute_targets(smiles)
    targets = clean_targets(
        targets
    )  # Replace nan values with 0.0 and inf values with max/min float32

    # We also normalize such that we don't give more weight to naturally larger targets
    scaler = StandardScaler()
    targets = scaler.fit_transform(targets)

    # Train model
    train_data = TokenizedDataset(tokens, targets)
    wrapper = MTRModelWrapper(
        model, num_labels=targets.shape[1], extract_embeddings_fn=extract_embeddings_fn
    )

    trainer = Trainer(model=wrapper, train_dataset=train_data, *args, **kwargs)
    trainer.train()

    return (
        orig,
        wrapper.to("cpu"),
        tokens,
    )


if __name__ == "__main__":
    import pandas as pd

    df = pd.read_csv("astrazeneca_LogD74.csv")

    def first_token(outputs):
        token_embeddings = outputs.last_hidden_state
        summary = token_embeddings[:, 0, :]
        return summary

    adapt_model_with_mtr(
        model_name="UdS-LSV/da4mt-mlm-60",
        smiles=df["smiles"].tolist()[:10],
        extract_embeddings_fn=first_token,
    )
