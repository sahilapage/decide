"""
The shared network: one transformer encoder, three small typed heads.

    encoder(prompt + state)  ->  mean-pooled vector h        (shared, ~all params)
    route head     Linear(h) -> 5 logits                      (choice)
    approve head   MLP([h, aux]) -> 1 logit                   (binary)
    grounded head  MLP([h, aux]) -> 1 logit                   (binary)

aux = tier one-hot, tier x amount bucket, log amount (read by approve) and
three number-consistency features of draft vs article (read by grounded).

All three heads run on every call (they are a few thousand parameters each);
the prompt in the input decides which one the caller reads. The ONNX export
is therefore one graph with three outputs, which is the literal "one model".
"""

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

from decide.textfmt import APPROVE_DIM, TAB_DIM


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
        self.grounded_head = nn.Sequential(
            nn.Linear(h + TAB_DIM, 256), nn.GELU(), nn.Dropout(dropout), nn.Linear(256, 1))
        # Learned weight on the "draft states a number the article does not"
        # flag, added straight to the grounded logit. Starts strongly negative
        # (the flag never fires on a supported draft in train or dev) so the
        # MLP does not have to rediscover it from 450 inputs; it stays trainable.
        self.grounded_flag_w = nn.Parameter(torch.tensor(-8.0))
        # same idea for the scope-broadening flag ("all plans" vs "all paid plans")
        self.grounded_scope_w = nn.Parameter(torch.tensor(-8.0))
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
        tab = tab.to(h.dtype)
        approve = self.approve_head(torch.cat([h, tab], dim=-1)).squeeze(-1)
        grounded = (self.grounded_head(torch.cat([h, tab], dim=-1)).squeeze(-1)
                    + self.grounded_flag_w.to(h.dtype) * tab[:, APPROVE_DIM]
                    + self.grounded_scope_w.to(h.dtype) * tab[:, APPROVE_DIM + 3])
        return route, approve, grounded
