import torch
from torch import nn
import torch.nn.functional as F

from util import get_pairwise_idxs_separate

from util import RegLayer

import math

class SpanTransformer(nn.Module):

    def __init__(self, span_dim, vocabs, num_layers=3, final_pred_embeds=False,
                 et_dim=64, tt_dim=64, p_dropout=0.3, single_hid_dim=1024,
                 pair_hid_dim=256, dist_seps=(5, 10, 20, 50, 100), dist_embed_dim=64,
                 span_dim_small=128, coref_embed_dim=128):

        super().__init__()


        self.span_dim = span_dim
        self.span_dim_small = span_dim_small
        self.final_pred_embeds = final_pred_embeds

        self.layers = []

        for i in range(num_layers):
            input_dim = span_dim
            #if final_pred_embeds:
            #    input_dim += et_dim + tt_dim

            trans_layer = nn.Sequential(nn.TransformerEncoderLayer(input_dim, nhead=4))
                                        ##RegLayer(span_dim))

            self.layers.append(trans_layer)

        self.layers = nn.ModuleList(self.layers)

        #self.linear_is_entity = nn.Linear(span_dim, 2)

        self.linear_entity_type = nn.Sequential(nn.Linear(span_dim, single_hid_dim), RegLayer(single_hid_dim),
                                                nn.LeakyReLU(), nn.Linear(single_hid_dim, len(vocabs['entity'])))

        #self.linear_is_trigger = nn.Linear(span_dim, 2)

        self.linear_trigger_type = nn.Sequential(nn.Linear(span_dim, single_hid_dim), RegLayer(single_hid_dim),
                                                nn.LeakyReLU(), nn.Linear(single_hid_dim, len(vocabs['event'])))

        #self.coref_classifier = nn.Sequential(nn.Linear(span_dim * 2, pair_hid_dim), nn.ReLU(),
        #                                    nn.Linear(pair_hid_dim, 2))

        self.before_rel_compress = nn.Sequential(nn.Linear(span_dim, span_dim * 2), RegLayer(span_dim * 2),
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

        self.dist_embed = nn.Embedding(len(dist_seps) + 1, dist_embed_dim)


    def forward(self, span_repr, predict=False, true_spans=None, batch=None, span_cand_idxs=None):

        batch_size = span_repr.shape[0]

        for l in self.layers:

            if self.final_pred_embeds:
                entity_type = self.linear_entity_type(span_repr)
                trigger_type = self.linear_trigger_type(span_repr)

                et_embed = self.et_embed_layer(entity_type.argmax(dim=-1))
                tt_embed = self.tt_embed_layer(trigger_type.argmax(dim=-1))

                span_repr = span_repr + et_embed + tt_embed#torch.cat((span_repr, et_embed, tt_embed), dim=-1)

            span_repr = l(span_repr)



        entity_type = self.linear_entity_type(span_repr)
        trigger_type = self.linear_trigger_type(span_repr)

        return entity_type, trigger_type, None, None

        if predict == False:

            #print(span_repr.shape)

            #print(true_spans.min(), true_spans.max())

            if true_spans.shape[-1] == 0:
                span_repr = span_repr[:, 0].unsqueeze(1)
                print(span_repr.shape)
            else:
                span_repr = torch.gather(span_repr, 1, true_spans.unsqueeze(-1).expand(-1, -1, self.span_dim))

        #print(span_repr.shape)

        span_num = span_repr.shape[1]

        """if span_num > 300:
            span_num = 300
            span_repr = span_repr[:, :span_num]
            #span_repr_orig = span_repr_orig[:, :span_num]"""

        #special embedding method

        coref_embeds = self.coref_embed(span_repr)

        ###

        return entity_type, trigger_type, None, coref_embeds

        #attn_based pair stuff

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

        ###old rel stuff below this

        """span_repr = self.before_rel_compress(span_repr)

        pair_src_idxs, pair_dst_idxs = get_pairwise_idxs_separate(span_num,
                                                                span_num,
                                                                True)

        #print(span_num, len(pair_src_idxs))


        dists = []

        for b in range(batch_size):
            cur_dist_list = []
            for i in range(len(pair_src_idxs)):
                if predict == False:
                    a_pos = batch.pos_entity_offsets[b][pair_src_idxs[i]][0]
                    b_pos = batch.pos_entity_offsets[b][pair_dst_idxs[i]][0]
                    #a_pos = batch.entity_offsets[span_cand_idxs[b, true_spans[b, pair_src_idxs[i]].item()].item()][0]
                    #b_pos = batch.entity_offsets[span_cand_idxs[b, true_spans[b, pair_dst_idxs[i]].item()].item()][0]
                else:
                    a_pos = batch.entity_offsets[span_cand_idxs[b, pair_src_idxs[i]].item()][0]
                    b_pos = batch.entity_offsets[span_cand_idxs[b, pair_dst_idxs[i]].item()][0]
                dist = abs(a_pos - b_pos)
                dist_val = len(self.dist_seps)
                for sep_id, sep in enumerate(self.dist_seps):
                    if dist <= sep:
                        dist_val = sep_id
                        break
                cur_dist_list.append(dist_val)
            dists.append(cur_dist_list)

        dist = torch.cuda.LongTensor(dists)


        pair_src_idxs = (torch.cuda.LongTensor(pair_src_idxs)
                        .unsqueeze(0).unsqueeze(-1)
                        .expand(batch_size, -1, self.span_dim_small))
        pair_dst_idxs = (torch.cuda.LongTensor(pair_dst_idxs)
                        .unsqueeze(0).unsqueeze(-1)
                        .expand(batch_size, -1, self.span_dim_small))

        pair_src_repr = (span_repr
                        .gather(1, pair_src_idxs)
                        .view(batch_size, span_num, span_num - 1, self.span_dim_small))
        pair_dst_repr = (span_repr
                        .gather(1, pair_dst_idxs)
                        .view(batch_size, span_num, span_num - 1, self.span_dim_small))

        del span_repr, pair_src_idxs, pair_dst_idxs

        dist_embeds = self.dist_embed(dist)

        dist_embeds = dist_embeds.view(batch_size, pair_src_repr.shape[1], -1, dist_embeds.shape[-1])

        #diff = pair_dst_repr - pair_src_repr

        #del pair_dst_repr, pair_src_repr

        pair_repr = torch.cat((pair_dst_repr, pair_src_repr, dist_embeds), dim=-1)

        #relation_type = self.rel_classifier(pair_repr)

        #coref_pred = None
        coref_pred = self.coref_classifier(pair_repr)

        return entity_type, trigger_type, None, coref_pred"""
