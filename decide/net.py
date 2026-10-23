"""
The shared network: one transformer encoder, three small typed heads.

    encoder(prompt + state)  ->  mean-pooled vector h        (shared, ~all params)
    route head     Linear(h) -> 5 logits                      (choice)
    approve head   MLP([h, tabular]) -> 1 logit               (binary)
    grounded head  Linear(h) -> 1 logit                       (binary)

All three heads run on every call (they are a few thousand parameters each);
the prompt in the input decides which one the caller reads. The ONNX export
is therefore one graph with three outputs, which is the literal "one model".
"""

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

from decide.textfmt import TAB_DIM


class DecideNet(nn.Module):
    def __init__(self, encoder_name_or_path: str, dropout: float = 0.1, from_config: bool = False):
        super().__init__()
        if from_config:
            cfg = AutoConfig.from_pretrained(encoder_name_or_path)
            self.encoder = AutoModel.from_config(cfg)
        else:
            self.encoder = AutoModel.from_pretrained(encoder_name_or_path)
        h = self.encoder.config.hidden_size
        self.drop = nn.Dropout(dropout)
        self.route_head = nn.Linear(h, 5)
        self.grounded_head = nn.Linear(h, 1)
        self.approve_head = nn.Sequential(
            nn.Linear(h + TAB_DIM, 256), nn.GELU(), nn.Dropout(dropout), nn.Linear(256, 1))

    def pooled(self, input_ids, attention_mask, token_type_ids=None):
        kw = dict(input_ids=input_ids, attention_mask=attention_mask)
        if token_type_ids is not None:
            kw["token_type_ids"] = token_type_ids
        out = self.encoder(**kw).last_hidden_state
        m = attention_mask.unsqueeze(-1).to(out.dtype)
        return (out * m).sum(1) / m.sum(1).clamp(min=1.0)

    def forward(self, input_ids, attention_mask, tab, token_type_ids=None):
        h = self.drop(self.pooled(input_ids, attention_mask, token_type_ids))
        route = self.route_head(h)
        approve = self.approve_head(torch.cat([h, tab.to(h.dtype)], dim=-1)).squeeze(-1)
        grounded = self.grounded_head(h).squeeze(-1)
        return route, approve, grounded
