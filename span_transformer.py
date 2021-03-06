import torch
from torch import nn
import torch.nn.functional as F

from util import get_pairwise_idxs_separate

from util import RegLayer

import math

class SpanTransformer(nn.Module):

    def __init__(self, span_dim, vocabs, num_layers=3, final_pred_embeds=False,
                 et_dim=64, tt_dim=64, single_hid_dim=1024,
                 pair_hid_dim=256, dist_seps=(5, 10, 20, 50, 100), dist_embed_dim=64,
                 span_dim_small=128, coref_embed_dim=128, dropout=0.1, nhead=8):

        super().__init__()


        self.span_dim = span_dim
        self.span_dim_small = span_dim_small
        self.final_pred_embeds = final_pred_embeds

        self.layers = []

        for i in range(num_layers):
            input_dim = span_dim
            #if final_pred_embeds:
            #    input_dim += et_dim + tt_dim

            trans_layer = nn.Sequential(nn.TransformerEncoderLayer(input_dim, nhead=nhead, dropout=dropout))

            self.layers.append(trans_layer)

        self.layers = nn.ModuleList(self.layers)

        self.lin = nn.Sequential(nn.Linear(input_dim * 3, input_dim), RegLayer(input_dim))

        """self.before_rel_compress = nn.Sequential(nn.Linear(span_dim, span_dim * 2), RegLayer(span_dim * 2),
                                                 nn.Linear(span_dim * 2, span_dim_small), RegLayer(span_dim_small))

        self.rel_q_linear = nn.Linear(span_dim, span_dim_small)
        self.rel_k_linear = nn.Linear(span_dim, span_dim_small)
        self.rel_attn_heads = 8

        self.rel_classifier = nn.Sequential(nn.Linear(span_dim_small * 2 + dist_embed_dim, pair_hid_dim),
                                            nn.LayerNorm(pair_hid_dim),#egLayer(pair_hid_dim),
                                            nn.LeakyReLU(), nn.Linear(pair_hid_dim, len(vocabs['relation'])))

        self.coref_embed = nn.Sequential(nn.Linear(span_dim, span_dim * 2), RegLayer(span_dim * 2), nn.LeakyReLU(),
                                         nn.Linear(span_dim * 2, coref_embed_dim))

        #self.coref_classifier = nn.Sequential(nn.Linear(span_dim_small * 2 + dist_embed_dim, pair_hid_dim),
        #                                    nn.LayerNorm(pair_hid_dim),#RegLayer(pair_hid_dim),
        #                                    nn.LeakyReLU(), nn.Linear(pair_hid_dim, 2))

        if self.final_pred_embeds:
            self.et_embed_layer = nn.Embedding(len(vocabs['entity']), span_dim)
            self.tt_embed_layer = nn.Embedding(len(vocabs['event']), span_dim)

            self.et_embed_layer.weight.data.uniform_(-0.1, 0.1)
            self.tt_embed_layer.weight.data.uniform_(-0.1, 0.1)

        self.dist_seps = dist_seps

        self.dist_embed = nn.Embedding(len(dist_seps) + 1, dist_embed_dim)"""


    def forward(self, span_repr, predict=False, true_spans=None, batch=None, span_cand_idxs=None):

        batch_size = span_repr.shape[0]

        span_reprs = [span_repr]

        for l in self.layers:

            if self.final_pred_embeds:
                entity_type = self.linear_entity_type(span_repr)
                trigger_type = self.linear_trigger_type(span_repr)

                et_embed = self.et_embed_layer(entity_type.argmax(dim=-1))
                tt_embed = self.tt_embed_layer(trigger_type.argmax(dim=-1))

                span_repr = span_repr + et_embed + tt_embed#torch.cat((span_repr, et_embed, tt_embed), dim=-1)

            span_repr = l(span_repr)
            span_reprs.append(span_repr)

        span_repr = torch.cat((span_reprs[0], span_reprs[len(self.layers) // 2], span_reprs[-1]), dim=-1)

        span_repr = self.lin(span_repr)

        return span_repr



        """rel_q = self.rel_q_linear(span_repr)
        rel_k = self.rel_k_linear(span_repr)

        sub_dim = self.span_dim_small // self.rel_attn_heads

        rel_q = rel_q.reshape(batch_size, span_num, self.rel_attn_heads, sub_dim) \
                .permute(0, 2, 1, 3) \
                .reshape(batch_size * self.rel_attn_heads, span_num, sub_dim)

        rel_k = rel_k.reshape(batch_size, span_num, self.rel_attn_heads, sub_dim) \
            .permute(0, 2, 1, 3) \
            .reshape(batch_size * self.rel_attn_heads, span_num, sub_dim)

        dk = rel_q.size()[-1]
        scores = rel_q.matmul(rel_k.transpose(-2, -1)) / math.sqrt(dk)

        attention = F.softmax(scores, dim=-1)
        attn_sum = attention.reshape(batch_size, self.rel_attn_heads, span_num, span_num).sum(dim=1)

        print(rel_q.shape, rel_k.shape, scores.shape, attn_sum.shape)

        print(scores.min(), scores.max(), scores.mean(), scores.var())
        print(attn_sum.min(), attn_sum.max(), attn_sum.mean(), attn_sum.var())

        return entity_type, trigger_type, None, None"""




class AggrTransformer(nn.Module):

    def __init__(self, span_dim, num_layers=3, nhead=4, dropout=0.2):

        super().__init__()

        self.span_dim = span_dim

        self.aggr_emb = nn.Parameter(torch.randn(span_dim).cuda() * 0.1)

        self.layers = []

        for i in range(num_layers):
            input_dim = span_dim

            trans_layer = nn.Sequential(nn.TransformerEncoderLayer(input_dim, nhead=nhead, dropout=dropout))

            self.layers.append(trans_layer)

        self.layers = nn.ModuleList(self.layers)

        #self.lin = nn.Sequential(nn.Linear(input_dim * 3, input_dim), RegLayer(input_dim))

    def forward(self, span_repr):


        span_repr[:, 0] += self.aggr_emb

        for l in self.layers:

            span_repr = l(span_repr)

        return span_repr[:, 0]


class ContextTransformer(nn.Module):

    def __init__(self, hid_dim, num_heads=4, num_layers=3):

        super().__init__()


        self.attn, self.norm = [], []
        self.lin = []


        for i in range(num_layers):
            self.attn.append(
                nn.MultiheadAttention(hid_dim, num_heads)
            )

            self.norm.append(
                nn.LayerNorm(hid_dim)
            )
            self.lin.append(
                nn.Linear(hid_dim, hid_dim)
            )

        self.attn = nn.ModuleList(self.attn)
        self.norm = nn.ModuleList(self.norm)
        self.lin = nn.ModuleList(self.lin)

        self.num_layers = num_layers

    def forward(self, context, x):

        #x[-2, :] += self.thr_emb
        #x[-1, :] += self.offload_emb
        #context[-2, :] += self.thr_emb
        #context[-1, :] += self.offload_emb

        attns = []

        for i in range(self.num_layers):

            x1, attn = self.attn[i](x, context, context)
            attns.append(attn)
            #x1 = self.norm[i](x1)

            x = x + x1

            x = self.norm[i](x)

            x = x + self.lin[i](x)

        return x, attns


