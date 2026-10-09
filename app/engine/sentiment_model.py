"""EXACT custom regression architecture exported by the Kaggle notebook."""
import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput


class FinBertSentimentRegressor(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        prob = getattr(config, 'classifier_dropout', None)
        if prob is None:
            prob = 0.10
        self.dropout = nn.Dropout(prob)
        self.regressor = nn.Linear(config.hidden_size, 1)
        self.tanh = nn.Tanh()
        self.loss_fn = nn.MSELoss()
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                labels=None, **kwargs):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        pooled = self.dropout(outputs.pooler_output)
        bounded_score = self.tanh(self.regressor(pooled))
        loss = None
        if labels is not None:
            loss = self.loss_fn(bounded_score, labels.float().view(-1, 1))
        return SequenceClassifierOutput(
            loss=loss,
            logits=bounded_score,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
