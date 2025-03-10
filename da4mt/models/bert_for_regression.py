from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch.nn import MSELoss
from transformers import BertModel
from transformers.file_utils import ModelOutput
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertConfig


class BertForRegressionConfig(BertConfig):
    def __init__(
        self,
        norm_mean=None,
        norm_std=None,
        num_labels: int = None,
        property_subset: str = "all",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        if num_labels is None and norm_mean is not None:
            self.num_labels = len(norm_mean)
        else:
            self.num_labels = 0
        self.property_subset = property_subset


class BertForRegression(BertPreTrainedModel):
    _keys_to_ignore_on_load_missing = ["position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.register_buffer("norm_mean", torch.tensor(config.norm_mean))
        # Replace any 0 stddev norms with 1
        self.register_buffer(
            "norm_std",
            torch.tensor(
                [label_std if label_std != 0 else 1 for label_std in config.norm_std]
            ),
        )

        self.bert = BertModel(config, add_pooling_layer=False)
        self.regression = BertRegressionHead(config)

        self.post_init()

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

        outputs = self.bert(
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

        sequence_output = (
            outputs.last_hidden_state
        )  # shape = (batch, seq_len, hidden_size)
        logits = self.regression(sequence_output)

        if labels is None:
            return self.unnormalize_logits(logits)

        if labels is not None:
            normalized_labels = self.normalize_logits(labels)
            loss_fct = MSELoss()
            loss = loss_fct(logits.view(-1), normalized_labels.view(-1))

            if not return_dict:
                output = (logits,) + outputs[2:]
                return ((loss,) + output) if loss is not None else output

        return RegressionOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def normalize_logits(self, tensor):
        return (tensor - self.norm_mean) / self.norm_std

    def unnormalize_logits(self, tensor):
        return (tensor * self.norm_std) + self.norm_mean


class BertRegressionHead(nn.Module):
    """Head for multitask regression models."""

    def __init__(self, config):
        super(BertRegressionHead, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


@dataclass
class RegressionOutput(ModelOutput):
    """
    Base class for outputs of regression models. Supports single and multi-task regression.

    Args:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided)
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.num_labels)`):
            Regression scores for each task (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is
        passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or
        when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
