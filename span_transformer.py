import torch
from torch import nn


class SpanTransformer(nn.Module):

    def __init__(self, span_dim, vocabs):

        super().__init__()

        self.trans_layer = nn.TransformerEncoderLayer(span_dim, num_heads=6)

        #self.linear_is_entity = nn.Linear(span_dim, 2)

        self.linear_entity_type = nn.Linear(span_dim, len(vocabs['entity']))

        #self.linear_is_trigger = nn.Linear(span_dim, 2)

        self.linear_trigger_type = nn.Linear(span_dim, len(vocabs['event']))

        print()

    def forward(self, span_repr):

        span_repr = self.attn_layer(span_repr)

        #is_entity = self.linear_is_entity(span_repr)

        entity_type = self.linear_entity_type(span_repr)

        #is_trigger = self.linear_is_trigger(span_repr)

        trigger_type = self.linear_trigger_type(span_repr)

        return entity_type, trigger_type
