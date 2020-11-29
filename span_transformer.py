import torch
from torch import nn


class SpanTransformer(nn.Module):

    def __init__(self, span_dim, vocabs, num_layers=3, final_pred_embeds=False, et_dim=64, tt_dim=64, p_dropout=0.3):

        super().__init__()

        self.final_pred_embeds = final_pred_embeds

        self.layers = []

        for i in range(num_layers):
            input_dim = span_dim
            if final_pred_embeds:
                input_dim += et_dim + tt_dim

            trans_layer = nn.Sequential(nn.TransformerEncoderLayer(input_dim, nhead=8),
                                        nn.Linear(input_dim, span_dim), nn.Dropout(p_dropout))

            self.layers.append(trans_layer)

        self.layers = nn.ModuleList(self.layers)

        #self.linear_is_entity = nn.Linear(span_dim, 2)

        self.linear_entity_type = nn.Linear(span_dim, len(vocabs['entity']))

        #self.linear_is_trigger = nn.Linear(span_dim, 2)

        self.linear_trigger_type = nn.Linear(span_dim, len(vocabs['event']))

        if self.final_pred_embeds:
            self.et_embed_layer = nn.Embedding(len(vocabs['entity']), et_dim)
            self.tt_embed_layer = nn.Embedding(len(vocabs['event']), tt_dim)


    def forward(self, span_repr):


        for l in self.layers:

            if self.final_pred_embeds:
                entity_type = self.linear_entity_type(span_repr)
                trigger_type = self.linear_trigger_type(span_repr)

                et_embed = self.et_embed_layer(entity_type.argmax(dim=-1))
                tt_embed = self.tt_embed_layer(trigger_type.argmax(dim=-1))

                span_repr = torch.cat((span_repr, et_embed, tt_embed), dim=-1)

            span_repr = l(span_repr)

        entity_type = self.linear_entity_type(span_repr)
        trigger_type = self.linear_trigger_type(span_repr)

        return entity_type, trigger_type
