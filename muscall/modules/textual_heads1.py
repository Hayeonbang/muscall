import torch
from torch import nn
from einops import rearrange

from muscall.modules.transformer import Transformer, LayerNorm
from transformers import BertModel, BertTokenizer


bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')


import torch
from torch import nn
from transformers import BertModel

class TextualHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.bert = BertModel.from_pretrained('bert-base-uncased')

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state

class TextualHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config


class TextTransformer(TextualHead):
    def __init__(self, config):
        super().__init__(config)
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')


    def _build_causal_attention_mask(self, batch_size, seq_len):
        """Create causal attention mask.

        This is a triangular matrix with upper diagonal values set to -inf.
        This is because we're using an additive mask in the attention module.
        """
        mask = torch.empty(batch_size, seq_len, seq_len)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        mask = mask.unsqueeze(1)  # expand mask
        return mask

    def forward(self, x: torch.Tensor, mask=None):
        # BERT 모델을 사용하여 텍스트 인코딩
        outputs = self.bert_model(input_ids=x, attention_mask=mask)
        return outputs.last_hidden_state    