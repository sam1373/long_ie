import enum
import logging
from typing import Dict, List, Tuple, Any


import dgl
from dgl.utils.internal import relabel
import torch
import torch.nn as nn
import torch.nn.functional as F

import transformers
from data import Batch
from util import label2onehot, elem_max, elem_min, get_pairwise_idxs_separate, RegLayer

from span_transformer import ContextTransformer

import torch_scatter

import random

from sklearn.cluster import DBSCAN, OPTICS, AgglomerativeClustering


logger = logging.getLogger(__name__)


def calculate_enumeration_num(seq_len: int,
                              max_span_len: int) -> int:
    max_span_len = min(seq_len, max_span_len)
    return seq_len * max_span_len - ((max_span_len - 1) * max_span_len) // 2


def get_pairwise_idxs(num1: int, num2: int, skip_diagonal: bool = False, sent_nums = None, sep=True):
    idxs = []
    for i in range(num1):
        for j in range(num2):
            if i == j and skip_diagonal:
                continue
            if sent_nums is not None and sent_nums[i] != sent_nums[j]:
                continue
            if sep:
                idxs.append(i)
                idxs.append(j)
            else:
                idxs.append((i, j))
    return idxs



def transpose_edge_feats(edge_feats: torch.Tensor, src_num: int, dst_num: int):
    idxs = []
    for i in range(dst_num):
        for j in range(src_num):
            idxs.append(i + j * src_num)
    idxs = edge_feats.new_tensor(idxs, dtype=torch.long).unsqueeze(-1).expand(-1, edge_feats.size(-1))
    return edge_feats.gather(0, idxs)



class Linears(nn.Module):
    """Multiple linear layers with Dropout."""
    def __init__(self,
                 dimensions: List[int],
                 activation: str='relu',
                 dropout_prob: float=0.0,
                 bias: bool=True):
        super().__init__()
        assert len(dimensions) > 1
        self.layers = nn.ModuleList([nn.Linear(dimensions[i], dimensions[i + 1], bias=bias)
                                     for i in range(len(dimensions) - 1)])
        self.activation = activation
        self.func = getattr(torch, activation)
        self.dropout = nn.Dropout(dropout_prob, inplace=True)
        self.regs = nn.ModuleList([RegLayer(dimensions[i]) for i in range(1, len(dimensions) - 1)])

        self.init_parameters()

    def forward(self, inputs):
        for i, layer in enumerate(self.layers):
            if i > 0:
                inputs = self.func(inputs)
                #inputs = self.dropout(inputs)
                inputs = self.regs[i - 1](inputs)
            inputs = layer(inputs)
        return inputs

    def init_parameters(self):
        gain = nn.init.calculate_gain(self.activation)
        for layer in self.layers:
            nn.init.xavier_normal_(layer.weight, gain=gain)


class PairLinears(nn.Module):
    def __init__(self,
                 src_dim: int,
                 dst_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 activation: str='relu',
                 dropout_prob: float=0.0,
                 bias: bool=True):
        super().__init__()

        self.src_linear = nn.Linear(src_dim, hidden_dim, bias=bias)
        self.dst_linear = nn.Linear(dst_dim, hidden_dim, bias=bias)
        self.output_linear = nn.Linear(hidden_dim, output_dim, bias=bias)
        self.dropout = nn.Dropout(dropout_prob, inplace=True)

        self.activation = activation
        self.func = getattr(torch, activation)
        self.init_parameters()

    def init_parameters(self):
        gain = nn.init.calculate_gain(self.activation)
        nn.init.xavier_normal_(self.src_linear.weight, gain=gain)
        nn.init.xavier_normal_(self.dst_linear.weight, gain=gain)
        nn.init.xavier_normal_(self.output_linear.weight, gain=gain)

    def forward_src(self, inputs):
        return self.src_linear(inputs)

    def forward_dst(self, inputs):
        return self.dst_linear(inputs)

    def forward_output(self, src_outputs, dst_outputs):
        h = src_outputs + dst_outputs
        h = self.dropout(self.func(h))
        return self.output_linear(h)

    def forward(self, src_inputs, dst_inputs):
        src_outputs = self.src_linear(src_inputs)
        dst_outputs = self.dst_linear(dst_inputs)
        h = src_outputs + dst_outputs
        h = self.dropout(self.func(h))
        return self.output_linear(h)

class ResidLinear(nn.Module):
    """Multiple linear layers with Dropout."""

    def __init__(self,
                 in_dim,
                 out_dim: int = 2,
                 n_resid_layers: int = 1,
                 have_final: bool = True,
                 bias: bool = True):
        super().__init__()
        self.layers = nn.ModuleList([nn.Sequential(nn.Linear(in_dim, in_dim, bias=bias),
                                                   nn.LayerNorm(in_dim))
                                     for i in range(n_resid_layers)])
        self.activation = nn.LeakyReLU()

        self.norm = nn.ModuleList([nn.LayerNorm(in_dim)
                                   for i in range(n_resid_layers)])

        if have_final:
            self.final_lin = nn.Linear(in_dim, out_dim)

        self.have_final = have_final

        self.init_parameters()

    def forward(self, inputs):
        out = inputs
        for i, layer in enumerate(self.layers):
            res = layer[0](out)
            res = self.activation(res)
            out = out + res
            out = layer[1](out)
        if self.have_final:
            out = self.final_lin(out)
        return out

    def init_parameters(self):
        for layer in self.layers:
            layer[0].weight.data.uniform_(-0.01, 0.01)
            layer[0].bias.data.fill_(0)


class LongIE(nn.Module):
    def __init__(self,
                 config,
                 vocabs: Dict[str, Dict[str, int]],
                 encoder: nn.Module,
                 word_embed:nn.Module = None,
                 extra_bert: int = 0,
                 word_embed_dim: int = 0,
                 coref_embed_dim: int = 64,
                 hidden_dim: int = 500,#500
                 span_transformer_layers: int = 10,#10
                 encoder_dropout_prob: float = 0.,
                 type_embed_dim: int = 32,
                 comp_dim: int = 128,
                 ):
        super().__init__()

        self.config = config
        self.vocabs = vocabs
        self.encoder = encoder
        self.encoder_dropout = nn.Dropout(encoder_dropout_prob)

        token_initial_dim = self.encoder.config.hidden_size
        if self.config.get('use_extra_word_embed'):
            token_initial_dim += word_embed_dim

        if self.config.get('use_sent_num_embed'):
            token_initial_dim += self.config.get('sent_num_embed_dim')

        if self.config.get("use_extra_bert"):
            token_initial_dim += self.encoder.config.hidden_size

        self.comp_dim = comp_dim

        #self.token_start_enc = Linears([token_initial_dim, 1024, 512])

        #self.type_project = nn.Linear(token_initial_dim, token_initial_dim)

        self.rel_context_project = nn.Linear(token_initial_dim, comp_dim * 2)

        #token_initial_dim = comp_dim * 2


        self.entity_dim = token_initial_dim
        if self.config.get('use_sent_context'):
            self.entity_dim *= 2

        self.rel_project_1 = nn.Sequential(nn.Linear(self.entity_dim + type_embed_dim, comp_dim),
                                           nn.LayerNorm(comp_dim))
        self.rel_project_2 = nn.Sequential(nn.Linear(self.entity_dim + type_embed_dim, comp_dim),
                                           nn.LayerNorm(comp_dim))


        self.is_start_clf = Linears([token_initial_dim, hidden_dim, 2])
        self.len_from_here_clf = Linears([token_initial_dim, hidden_dim, self.config.max_entity_len * 2])
        self.type_clf = Linears([self.entity_dim, hidden_dim, len(vocabs['entity'])])

        self.is_start_clf_ev = Linears([token_initial_dim, hidden_dim, 2])
        self.len_from_here_clf_ev = Linears([token_initial_dim, hidden_dim, self.config.max_entity_len * 2])
        self.type_clf_ev = Linears([self.entity_dim, hidden_dim, len(vocabs['event'])])

        self.relation_any_clf = Linears([comp_dim * 2, hidden_dim, 2])

        self.relation_clf = Linears([comp_dim * 2, hidden_dim, 1])

        if self.config.get("relation_type_level") == "multitype":
            self.relation_clf = nn.Sequential(self.relation_clf, nn.Sigmoid())

        self.coref_embed = Linears([self.entity_dim, hidden_dim, coref_embed_dim])

        self.type_embed = nn.Embedding(len(vocabs['entity']), type_embed_dim)

        self.entity_importance_weight = Linears([self.entity_dim, hidden_dim, 1])
        """self.is_start_clf = ResidLinear(token_initial_dim, 2)
        self.len_from_here_clf = ResidLinear(token_initial_dim, self.config.max_entity_len * 2)
        self.type_clf = ResidLinear(self.entity_dim, len(vocabs['entity']))

        self.relation_any_clf = ResidLinear(self.entity_dim * 2 + type_embed_dim * 2, 2)

        self.relation_clf = ResidLinear(self.entity_dim * 2 + type_embed_dim * 2, len(vocabs['relation']))

        self.coref_embed = ResidLinear(self.entity_dim, coref_embed_dim)

        self.type_embed = nn.Embedding(len(vocabs['entity']), type_embed_dim)

        self.entity_importance_weight = ResidLinear(self.entity_dim, 1)"""

        self.mention_aggr = ResidLinear(token_initial_dim, have_final=False)

        self.entity_aggr_lin = ResidLinear(self.entity_dim, have_final=False)
        self.trigger_aggr_lin = ResidLinear(self.entity_dim, have_final=False)

        #self.conv_aggr = nn.Conv1d()


        #150 just in case
        self.sent_num_embed = nn.Embedding(150, self.config.get('sent_num_embed_dim'))


        #self.span_candidate_classifier = Linears([node_dim, 200, 200, 2],
        #                    dropout_prob=.2)

        #self.span_compress = nn.Linear(span_repr_dim, span_comp_dim)

        """self.span_transformer = SpanTransformer(span_dim=token_initial_dim,
                                                vocabs=vocabs,
                                                final_pred_embeds=False,
                                                num_layers=span_transformer_layers)"""

        #self.rel_compress = nn.Linear(token_initial_dim * 2 + type_embed_dim * 2, token_initial_dim)

        """self.span_pair_transformer = SpanTransformer(span_dim=token_initial_dim,
                                                     vocabs=vocabs,
                                                     final_pred_embeds=False,
                                                     num_layers=span_transformer_layers,
                                                     nhead=4,
                                                     dropout=0.2)"""

        #attn between rel pairs and enc tokens after init project
        #both are comp_dim * 2
        self.rel_transformer = ContextTransformer(comp_dim * 2, num_layers=3, num_heads=8)

        self.rel_type_embed = nn.Embedding(len(vocabs['relation']), comp_dim * 2)

        """nn.Transformer(num_encoder_layers = 0,
                      num_decoder_layers = 3,
                      d_model=comp_dim * 2,
                      nhead=4)"""

        """self.cluster_aggr_trans = AggrTransformer(span_dim=token_initial_dim,
                                                  num_layers=10,
                                                  nhead=4,
                                                  dropout=0.1)"""


        self.word_embed = word_embed
        self.use_extra_bert = config.get("use_extra_bert")
        self.extra_bert = extra_bert

        entity_weights = torch.ones(len(vocabs['entity'])).cuda()
        entity_weights[0] /= 100.

        event_weights = torch.ones(len(vocabs['event'])).cuda()
        #event_weights[0] /= 3.

        rel_weights = torch.ones(len(vocabs['relation'])).cuda()
        rel_weights[0] /= 5.

        role_weights = torch.ones(len(vocabs['role'])).cuda()
        #role_weights[0] /= len(vocabs['role'])

        self.entity_loss = nn.CrossEntropyLoss(weight=entity_weights)
        self.event_loss = nn.CrossEntropyLoss(weight=event_weights)

        if self.config.get("relation_type_level") == "multitype":
            self.relation_loss = nn.BCELoss()
        else:
            self.relation_loss = nn.CrossEntropyLoss(weight=rel_weights)

        self.role_loss = nn.CrossEntropyLoss(weight=role_weights)

        self.relation_nonzero_loss = nn.CrossEntropyLoss(weight=torch.Tensor([0.05, 1.]).cuda())
        self.criteria = nn.CrossEntropyLoss()

    def label_size(self, key: str):
        return len(self.vocabs.get(key, {}))

    def encode_bert(self,
                    inputs: torch.Tensor,
                    attention_mask: torch.Tensor,
                    token_lens: List[List[int]]
                    ) -> torch.Tensor:
        """Encode sentences using Bert-based encoders.
        Args:
            inputs (torch.Tensor): A batch of input sequences of wordpieces
            indices.
        """

        all_bert_outputs = self.encoder(inputs, attention_mask=attention_mask, output_hidden_states=True)

        bert_outputs = all_bert_outputs[0]

        if self.use_extra_bert:
            extra_bert_outputs = all_bert_outputs[2][self.extra_bert]
            bert_outputs = torch.cat([bert_outputs, extra_bert_outputs], dim=2)

        batch_size, _, bert_dim = bert_outputs.size()

        # Get sequence representations
        token_mask = torch.cat([attention_mask.new_zeros(batch_size, 1),
                                attention_mask[:,2:],
                                attention_mask.new_zeros(batch_size, 1)], dim=1)
        token_mask = ((1.0 - token_mask) * -1e14).softmax(1).unsqueeze(-1)
        seq_repr = (bert_outputs * token_mask).sum(1, keepdim=True)

        # Generate token representations from wordpiece representations
        max_token_num = max([len(x) for x in token_lens])
        max_token_len = max([max(x) for x in token_lens])

        token_idxs, token_mask = [], []
        for seq_token_lens in token_lens:
            seq_token_idxs, seq_token_mask = [], []
            offset = 0
            # Generate indices to select wordpieces of each token
            # Generate mask to average vectors
            for token_len in seq_token_lens:
                seq_token_idxs.extend([i + offset for i in range(token_len)]
                                      + [-1] * (max_token_len - token_len))
                if self.config.get("use_first_wp"):
                    seq_token_mask.extend([1.0] + [0.0] * (max_token_len - 1))
                else:
                    seq_token_mask.extend([1.0 / max(1, token_len)] * token_len
                                      + [0.0] * (max_token_len - token_len))
                offset += token_len
            # Pad the sequence
            pad_num = max_token_len * (max_token_num - len(seq_token_lens))
            seq_token_idxs.extend([-1] * pad_num)
            seq_token_mask.extend([0.0] * pad_num)
            token_idxs.append(seq_token_idxs)
            token_mask.append(seq_token_mask)
        token_idxs = (inputs.new(token_idxs)
                      .unsqueeze(-1)
                      .expand(batch_size, -1, bert_dim)) + 1#<s>
        token_mask = bert_outputs.new(token_mask).unsqueeze(-1)

        #print(token_idxs.shape, token_idxs.min(), token_idxs.max())

        # For each token, select vectors of its wordpieces and average them
        bert_outputs = bert_outputs.gather(1, token_idxs) * token_mask
        bert_outputs = bert_outputs.view(batch_size,
                                         max_token_num,
                                         max_token_len,
                                         bert_dim)
        bert_outputs = bert_outputs.sum(2)

        # Apply output dropout
        bert_outputs = self.encoder_dropout(bert_outputs)

        return bert_outputs

    def get_span_representations(self,
                                 encoder_outputs,
                                 idxs,
                                 mask,
                                 lens,
                                 boundaries,
                                 token_embed_ids,
                                 max_span_len: int):
        batch_size, seq_len, repr_dim = encoder_outputs.size()
        max_span_len = min(seq_len, max_span_len)
        idxs = idxs.unsqueeze(-1).expand(batch_size, -1, repr_dim)
        mask = mask.unsqueeze(-1).expand(batch_size, -1, 1)

        #lens = label2onehot(lens, max_span_len)
        lens = lens.unsqueeze(0).expand(batch_size, -1, -1)#.view(batch_size, -1, max_span_len)
        lens = torch.argmax(lens, dim=-1)
        lens = self.span_len_embed(lens)
        #lens_repr = torch.zeros([lens.shape[0], max_span_len], device=lens.device)
        #lens_repr = lens_repr.scatter(1, lens.unsqueeze(-1), 1).unsqueeze(0).expand(batch_size, -1, -1)

        #print(boundaries)
        #print(encoder_outputs.shape)

        boundaries0 = boundaries.unsqueeze(-1).expand(batch_size, -1, repr_dim)

        #bert
        boundary_repr = encoder_outputs.gather(1, boundaries0)
        boundary_repr = boundary_repr.view(batch_size, -1, repr_dim * (1 + self.config.get('use_end_boundary', True)))



        if self.config.get('use_avg_repr', False):
            avg_repr = ((encoder_outputs.gather(1, idxs) * mask)
                        .view(batch_size, -1, max_span_len, repr_dim)
                        .sum(2))
            span_repr = torch.cat([boundary_repr, avg_repr, lens], dim=2)
        else:
            span_repr = torch.cat([boundary_repr, lens], dim=2)

        if self.config.get('use_extra_word_embed'):
            word_embed_repr = self.word_embed(token_embed_ids).detach()
            word_embed_dim = word_embed_repr.shape[-1]

            boundaries1 = boundaries.unsqueeze(-1).expand(batch_size, -1, word_embed_dim)

            span_word_embeds = word_embed_repr.gather(1, boundaries1)
            span_word_embeds = span_word_embeds.view(batch_size, -1, word_embed_dim * (1 + self.config.get('use_end_boundary', True)))

            span_repr = torch.cat([span_repr, span_word_embeds], dim=2)

        return span_repr



    def forward_nn(self, batch: Batch, predict: bool = False, epoch=0, gold_inputs=False):
        # Run the encoder to get contextualized word representations

        #print(batch.graphs[0].entities)
        #print(batch.pos_entity_offsets)

        #print(batch.tokens)
        #print(batch.pieces_text)
        #print(batch.pieces.shape)

        #print(batch.token_lens)

        #print(batch.pieces)

        encoder_outputs = self.encode_bert(batch.pieces,
                                           attention_mask=batch.attention_mask,
                                           token_lens=batch.token_lens)


        #encoder_outputs = torch.zeros(batch.token_embed_ids.shape[:2] + (self.encoder.config.hidden_size,)).cuda()

        batch_size = encoder_outputs.size(0)

        #print(batch.sent_nums.max())

        if self.config.get('use_sent_num_embed'):
            sent_num_embeds = self.sent_num_embed(batch.sent_nums)
            encoder_outputs = torch.cat((encoder_outputs, sent_num_embeds), dim=-1)

        if self.config.get('use_extra_word_embed'):
            word_embed_repr = self.word_embed(batch.token_embed_ids)#.detach()
            #if epoch >= 6:
            #word_embed_repr = word_embed_repr.detach()
            encoder_outputs = torch.cat((encoder_outputs, word_embed_repr), dim=-1)

        #encoder_outputs = self.initial_project(encoder_outputs)

        #encoded_starts = self.token_start_enc(encoder_outputs)

        #print()

        #print(encoder_outputs.mean(-1))

        is_start_pred = self.is_start_clf(encoder_outputs)
        len_from_here_pred = self.len_from_here_clf(encoder_outputs).view(batch_size, is_start_pred.shape[1], -1, 2)

        is_start_pred_ev = len_from_here_pred_ev = None
        
        if self.config.get("classify_triggers"):
            is_start_pred_ev = self.is_start_clf_ev(encoder_outputs)
            len_from_here_pred_ev = self.len_from_here_clf_ev(encoder_outputs).view(batch_size, is_start_pred.shape[1], -1, 2)
        
        
        #type_from_here_pred = self.type_from_here_clf(encoded_starts)

        #len_from_here_pred[is_start_pred.argmax(-1) == 0][:, 0] = 10000.
        #type_from_here_pred[is_start_pred.argmax(-1) == 0][:, 0] = 10000.

        #importance = self.importance_score(encoder_outputs)

        #mention_aggr = self.mention_aggr(encoder_outputs)

        #print(batch.sent_nums)

        if self.config.get('use_sent_context'):
            sent_context = torch_scatter.scatter_max(encoder_outputs, batch.sent_nums, dim=1)[0]

        entity_span_list = []

        ent_sent_nums = []

        if gold_inputs or (not predict):

            max_entities = batch.len_from_here[batch.is_start == 1].sum()

            entity_spans = torch.zeros(batch_size, max_entities, self.entity_dim).cuda()

            for b in range(batch_size):
                for i in range(max_entities):
                    l, r = batch.pos_entity_offsets[b][i]

                    entity_span_list.append((l, r))
                    if self.config.get("use_sent_context"):
                        entity_spans[b, i] = torch.cat((torch.max(encoder_outputs[b, l:r], dim=0)[0],
                                                        sent_context[b, batch.sent_nums[b, l]]),
                                                       dim=-1)
                    else:
                        entity_spans[b, i] = torch.max(encoder_outputs[b, l:r], dim=0)[0]

                    ent_sent_nums.append(batch.sent_nums[b, l].item())

            if self.config.get("classify_triggers"):

                max_ev = batch.len_from_here_ev[batch.is_start_ev == 1].sum()

                if max_ev == 0:
                    trigger_spans = None

                else:

                    trigger_spans = torch.zeros(batch_size, max_ev, self.entity_dim).cuda()

                    for b in range(batch_size):
                        for i in range(max_ev):
                            l, r = batch.pos_trigger_offsets[b][i]

                            #entity_span_list.append((l, r))
                            if self.config.get("use_sent_context"):
                                trigger_spans[b, i] = torch.cat((torch.max(encoder_outputs[b, l:r], dim=0)[0],
                                                                sent_context[b, batch.sent_nums[b, l]]),
                                                               dim=-1)
                            else:
                                trigger_spans[b, i] = torch.max(encoder_outputs[b, l:r], dim=0)[0]

        else:

            if is_start_pred.argmax(-1).sum() == 0:
                max_entities = 0
            else:
                max_entities = len_from_here_pred[is_start_pred.argmax(-1) == 1].view(-1, 2).argmax(-1).sum()

            if max_entities == 0:
                is_start_pred[:, 0, 1] = 1000.

                len_from_here_pred[:, 0, :, 1] = -1000.
                len_from_here_pred[:, 0, 1, 1] = 1000.

                max_entities = 1

            entity_spans = torch.zeros(batch_size, max_entities, self.entity_dim).cuda()

            cur_ent = 0

            for b in range(batch_size):
                for i in range(is_start_pred.shape[1]):
                    for j in range(self.config.max_entity_len):
                        if is_start_pred[b, i].argmax(-1) == 1 and len_from_here_pred[b, i, j].argmax(-1) == 1:
                            l = i
                            r = l + max(1, j)

                            entity_span_list.append((l, r))

                            if self.config.get("use_sent_context"):
                                entity_spans[b, cur_ent] = torch.cat((torch.max(encoder_outputs[b, l:r], dim=0)[0],
                                                                sent_context[b, batch.sent_nums[b, l]]),
                                                               dim=-1)
                            else:
                                entity_spans[b, cur_ent] = torch.max(encoder_outputs[b, l:r], dim=0)[0]

                            ent_sent_nums.append(batch.sent_nums[b, l].item())

                            cur_ent += 1

                            if cur_ent >= max_entities:
                                break
                                
            if self.config.get("classify_triggers"):

                if is_start_pred_ev.argmax(-1).sum() == 0:
                    max_ev = 0
                else:
                    max_ev = len_from_here_pred_ev[is_start_pred_ev.argmax(-1) == 1].view(-1, 2).argmax(-1).sum()

                """if max_ev == 0:
                    is_start_pred_ev[:, 0, 1] = 1000.

                    len_from_here_pred_ev[:, 0, :, 1] = -1000.
                    len_from_here_pred_ev[:, 0, 1, 1] = 1000.

                    max_ev = 1"""

                trigger_spans = None

                if max_ev > 0:

                    trigger_spans = torch.zeros(batch_size, max_ev, self.entity_dim).cuda()
    
                    cur_ent = 0
    
                    for b in range(batch_size):
                        for i in range(is_start_pred_ev.shape[1]):
                            for j in range(self.config.max_trigger_len):
                                if is_start_pred_ev[b, i].argmax(-1) == 1 and len_from_here_pred_ev[b, i, j].argmax(-1) == 1:
                                    l = i
                                    r = l + max(1, j)
    
                                    #entity_span_list.append((l, r))
    
                                    if self.config.get("use_sent_context"):
                                        trigger_spans[b, cur_ent] = torch.cat((torch.max(encoder_outputs[b, l:r], dim=0)[0],
                                                                              sent_context[b, batch.sent_nums[b, l]]),
                                                                             dim=-1)
                                    else:
                                        trigger_spans[b, cur_ent] = torch.max(encoder_outputs[b, l:r], dim=0)[0]
    
                                    #ent_sent_nums.append(batch.sent_nums[b, l].item())
    
                                    cur_ent += 1
    
                                    if cur_ent >= max_ev:
                                        break


        cluster_labels = None

        relation_pred = None

        relation_any = None

        relation_true_for_cand = None
        evidence_true_for_cand = None

        attn_sum = None

        relation_cand = None

        if self.config.get("do_coref"):

            coref_embed = self.coref_embed(entity_spans.detach())

            if not gold_inputs:

                cluster_labels = torch.zeros(coref_embed.shape[:2]).cuda().long()

                if cluster_labels.shape[1] > 1:
                    for b in range(batch_size):
                        clustering = AgglomerativeClustering(distance_threshold=1.,
                                                             n_clusters=None).\
                            fit(coref_embed[b].detach().cpu().numpy())
                        cluster_labels[b] = torch.LongTensor(clustering.labels_).cuda()

                cluster_num = cluster_labels.max() + 1

            else:

                true_clusters = batch.mention_to_ent_coref

                cluster_num = true_clusters.max() + 1

                cluster_noise = torch.randint(0, cluster_num, true_clusters.shape).cuda()

                cluster_mask = torch.rand(true_clusters.shape).cuda() > 0.8

                noisy_clusters = torch.clone(true_clusters)

                #noisy_clusters[cluster_mask] = cluster_noise[cluster_mask]

                #to not mess up number of clusters
                #noisy_clusters[true_clusters == cluster_num] = cluster_num

                cluster_labels = noisy_clusters

            if self.config.get("classify_triggers") and trigger_spans is not None:

                coref_embed_ev = self.coref_embed(trigger_spans.detach())

                if not gold_inputs:

                    cluster_labels_ev = torch.zeros(coref_embed_ev.shape[:2]).cuda().long()

                    if cluster_labels_ev.shape[1] > 1:
                        for b in range(batch_size):
                            clustering = AgglomerativeClustering(distance_threshold=1.,
                                                                 n_clusters=None). \
                                fit(coref_embed_ev[b].detach().cpu().numpy())
                            cluster_labels_ev[b] = torch.LongTensor(clustering.labels_).cuda()

                    cluster_num_ev = cluster_labels_ev.max() + 1

                else:

                    true_clusters = batch.mention_to_ev_coref

                    cluster_num_ev = true_clusters.max() + 1

                    cluster_noise = torch.randint(0, cluster_num, true_clusters.shape).cuda()

                    cluster_mask = torch.rand(true_clusters.shape).cuda() > 0.8

                    noisy_clusters = torch.clone(true_clusters)

                    # noisy_clusters[cluster_mask] = cluster_noise[cluster_mask]

                    # to not mess up number of clusters
                    # noisy_clusters[true_clusters == cluster_num] = cluster_num

                    cluster_labels_ev = noisy_clusters


            #entity_imp = self.entity_importance_weight(entity_spans)

            ent_appears_sets = [set() for i in range(cluster_num)]

            cluster_lists = [[] for i in range(cluster_num)]



            for b in range(batch_size):
                for i in range(cluster_labels.shape[1]):
                    cluster_lists[cluster_labels[b, i].item()].append(i)
                    ent_appears_sets[cluster_labels[b, i].item()].add(ent_sent_nums[i])

            max_in_cluster = max([len(cl) for cl in cluster_lists])


            #+ 1 for aggregating
            #ent_span_lists = torch.zeros(len(cluster_lists), max_in_cluster + 1, entity_spans.shape[-1]).cuda()
            #attention_mask = torch.zeros(ent_span_lists.shape).cuda()
            #attention_mask[:, 0] = 1.

            entity_spans_for_aggr = self.entity_aggr_lin(entity_spans)

            entity_aggr = torch.zeros(batch_size, cluster_num, entity_spans.shape[-1]).cuda()


            for b in range(batch_size):
                for i in range(cluster_num):
                    entity_spans_cluster = entity_spans_for_aggr[b][cluster_labels[b] == i]

                    entity_aggr[b, i] = torch.max(entity_spans_cluster, dim=0)[0]


            #cluster types pred
            type_pred = self.type_clf(entity_aggr)

            entity_aggr_aligned = torch.gather(entity_aggr, 1,
                                               cluster_labels.unsqueeze(-1).
                                               expand(-1, -1, entity_aggr.shape[-1]))


            if self.config.get("classify_relations"):
                pair_ids = torch.LongTensor(get_pairwise_idxs(cluster_num, cluster_num)).cuda()
                entity_pairs = torch.gather(entity_aggr, 1, pair_ids.view(1, -1, 1).expand(entity_spans.shape[0], -1,
                                                                                            entity_spans.shape[2]))
                type_pred0 = type_pred.argmax(-1)
                if gold_inputs or self.config.get("only_train_g_i"):
                    type_pred0 = batch.entity_labels[batch.entity_labels > 0].view(batch_size, -1)
                type_pred_embed = self.type_embed(type_pred0)
                entity_type_pairs = torch.gather(type_pred_embed, 1,
                                                 pair_ids.view(1, -1, 1).
                                                 expand(entity_spans.shape[0], -1, type_pred_embed.shape[2]))

                entity_pairs = torch.cat((entity_pairs, entity_type_pairs), dim=-1)

                entity_pairs_proj_1 = self.rel_project_1(entity_pairs)

                entity_pairs_proj_1 = entity_pairs_proj_1.view(entity_spans.shape[0], -1, entity_pairs_proj_1.shape[2] * 2)

                entity_pairs_proj_2 = self.rel_project_2(entity_pairs)

                entity_pairs_proj_2 = entity_pairs_proj_2.view(entity_spans.shape[0], -1, entity_pairs_proj_2.shape[2] * 2)

                #entity_pairs = self.span_pair_transformer(entity_pairs)

                relation_any = self.relation_any_clf(entity_pairs_proj_1)

                #relation_any[:, :, 1] = 1000.

                #relation_cand = (relation_any.argmax(-1) == 1)



                if not predict:
                    #relation_cand = elem_max((relation_any.argmax(-1) == 1),
                    #                     (batch.relation_nonzero.view(batch_size, -1)))
                    relation_cand = batch.relation_nonzero.view(batch_size, -1)

                    if elem_max((relation_any.argmax(-1) == 1),
                            (batch.relation_nonzero.view(batch_size, -1))).sum(-1) < 150:
                        relation_cand = elem_max((relation_any.argmax(-1) == 1),
                            (batch.relation_nonzero.view(batch_size, -1)))
                        #print(relation_cand.sum(-1))


                    if relation_cand.sum(-1) < 150:
                        relation_cand_neg_sample_mask = torch.rand(relation_cand.shape).cuda() > 0.95
                        relation_cand[relation_cand_neg_sample_mask] = 1
                        #print(relation_cand.sum(-1))


                else:
                    #want more candidates for predict, later thresholded by build_information_graph
                    thr = 0.1
                    relation_cand = (relation_any[:, :, -1] > thr)
                    while relation_cand.sum(-1) > 150:
                        thr += 0.05
                        relation_cand = (relation_any[:, :, -1] > thr)

                    #print(thr, relation_cand.sum(-1))


                """if not predict:
                    #negative relation sampling
                    relation_cand_neg_sample_mask = torch.rand(relation_cand.shape).cuda() > 0.9
                    relation_cand[relation_cand_neg_sample_mask] = 1
                    #//doesn't work because relation_any is unchanged?
                    #should work now
                    #..."""

                total_rel_cand = relation_cand.sum()


                if total_rel_cand > 500:
                    if predict:
                        relation_cand[:, 500:] = 0.
                        total_rel_cand = relation_cand.sum()
                    if not predict:
                        relation_cand = (batch.relation_nonzero.view(batch_size, -1))
                        total_rel_cand = relation_cand.sum()


                #if only between mentions in the same sentence
                if self.config.get("only_in_sent_rels", False):
                    in_sent = []
                    for i in range(cluster_num):
                        cur_l = []
                        for j in range(cluster_num):
                            if i == j:
                                cur_l.append(False)
                                continue
                            cur_l.append(len(ent_appears_sets[i].intersection(ent_appears_sets[j])) > 0)
                        in_sent.append(cur_l)

                    in_sent = torch.LongTensor(in_sent).cuda()

                    relation_cand = elem_min(relation_cand.long(), in_sent.view(batch_size, -1))

                if total_rel_cand == 0:
                    relation_cand = (relation_any.argmax(-1) == 1).long()
                    relation_cand[:, 0] = 1.
                    total_rel_cand = relation_cand.sum()

                #print(relation_cand)
                #print(relation_cand.shape)

                if not (predict and total_rel_cand > 500):
                    relation_cand_pairs = torch.zeros(batch_size, total_rel_cand, entity_pairs_proj_2.shape[-1]).cuda()
                    if not predict:
                        relation_true_for_cand = torch.zeros((batch_size, total_rel_cand) +
                                                             batch.relation_labels.shape[3:]).cuda().long()
                        evidence_true_for_cand = torch.zeros((batch_size, total_rel_cand) +
                                                             batch.evidence_labels.shape[3:]).cuda()
                    cur_idx = 0
                    for b in range(batch_size):
                        for i in range(cluster_num):
                            for j in range(cluster_num):
                                pair_idx = i * cluster_num + j

                                if relation_cand[b, pair_idx]:#relation_any.argmax(-1)[b, pair_idx] == 1:
                                    relation_cand_pairs[b, cur_idx] = entity_pairs_proj_2[b, pair_idx]
                                    if not predict:
                                        relation_true_for_cand[b, cur_idx] = batch.relation_labels[b, i, j]
                                        evidence_true_for_cand[b, cur_idx] = batch.evidence_labels[b, i, j]
                                    cur_idx += 1
                    if not predict:
                        evidence_true_for_cand = evidence_true_for_cand.transpose(-2, -1)



                    #relation_cand_pairs = self.rel_compress(relation_cand_pairs)

                    #relation_cand_pairs = self.span_pair_transformer(relation_cand_pairs)

                    #relation_true_for_cand = batch.relation_labels.view(batch_size, -1)

                    #relation_cand_pairs = self.rel_transformer(encoder_outputs.transpose(0, 1),
                    #                                           relation_cand_pairs.transpose(0, 1)).transpose(0, 1)

                    encoder_comp = self.rel_context_project(encoder_outputs)

                    if self.config.get("condense_sents"):
                        encoder_comp = torch_scatter.scatter_max(encoder_comp, batch.sent_nums, dim=1)[0]

                    #add additional thing to "offload" attention to
                    encoder_comp = torch.cat((encoder_comp,
                                                   torch.zeros(batch_size, 1, self.comp_dim * 2).cuda()), dim=1)

                    #relation_cand_pairs = self.rel_transformer(encoder_comp.transpose(0, 1),
                    #                                           relation_cand_pairs.transpose(0, 1)).transpose(0, 1)


                    num_rel_types = len(self.vocabs["relation"])

                    all_type_idx = torch.arange(0, num_rel_types).cuda()
                    all_type_embeds = self.rel_type_embed(all_type_idx)\

                    all_type_embeds = all_type_embeds.unsqueeze(0).unsqueeze(0).\
                        repeat(1, relation_cand_pairs.shape[1], 1, 1)

                    relation_cand_pairs = relation_cand_pairs.unsqueeze(2).repeat(1, 1, num_rel_types, 1)

                    #relation_cand_pairs_trans = torch.cat((relation_cand_pairs_trans, all_type_embeds), dim=-1)
                    relation_cand_pairs = relation_cand_pairs + all_type_embeds

                    hidden_size = relation_cand_pairs.shape[-1]


                    relation_cand_pairs_trans, attns = self.rel_transformer(encoder_comp.transpose(0, 1),
                                                               relation_cand_pairs.view(batch_size, -1, hidden_size).transpose(0, 1))


                    relation_cand_pairs_trans = relation_cand_pairs_trans.transpose(0, 1).view(batch_size, -1, num_rel_types, hidden_size)

                    relation_cand_pairs = relation_cand_pairs + relation_cand_pairs_trans

                    attn_sum = torch.sum(torch.stack(attns, dim=0), dim=0)[:, :, :-1].\
                        reshape(batch_size, total_rel_cand, num_rel_types, -1).transpose(-2, -1)

                    if not self.config.get("condense_sents"):
                        attn_sum = torch_scatter.scatter_mean(attn_sum, batch.sent_nums.unsqueeze(1), dim=2)

                    #attn_sum[attn_sum < 1.] = 0.

                    """attn_highest = attn_sum.max()
                    attn_mean = attn_sum.mean()

                    attn_high_list = attn_sum.nonzero().tolist()

                    if len(attn_high_list) > 0:
                        print(attn_high_list)
                        print(attn_highest, attn_mean)"""

                    relation_pred = self.relation_clf(relation_cand_pairs)

                    relation_pred = relation_pred.view(batch_size, -1, num_rel_types)


            entity_spans = entity_aggr_aligned


            if self.config.get("classify_triggers") and trigger_spans is not None:

                cluster_lists = [[] for i in range(cluster_num_ev)]

                for b in range(batch_size):
                    for i in range(cluster_labels_ev.shape[1]):
                        cluster_lists[cluster_labels_ev[b, i].item()].append(i)

                max_in_cluster = max([len(cl) for cl in cluster_lists])

                trig_aggr = torch.zeros(batch_size, cluster_num_ev, trigger_spans.shape[-1]).cuda()

                trig_spans_for_aggr = self.trigger_aggr_lin(trigger_spans)

                for b in range(batch_size):
                    for i in range(cluster_num_ev):
                        trigger_spans_cluster = trig_spans_for_aggr[b][cluster_labels_ev[b] == i]

                        trig_aggr[b, i] = torch.max(trigger_spans_cluster, dim=0)[0]

                # cluster types pred
                #type_pred_ev = self.type_clf(trig_aggr)

                trig_aggr_aligned = torch.gather(trig_aggr, 1,
                                                   cluster_labels_ev.unsqueeze(-1).
                                                   expand(-1, -1, trig_aggr.shape[-1]))

                trigger_spans = trig_aggr_aligned

                #type_pred_ev = self.type_clf_ev(trig_aggr_aligned)

        else:
            coref_embed = None
            cluster_labels_ev = None
            coref_embed_ev = None

        type_pred = self.type_clf(entity_spans)

        if self.config.get("classify_triggers") and trigger_spans is not None:
            type_pred_ev = self.type_clf_ev(trigger_spans)
        else:
            type_pred_ev = None
            coref_embed_ev = None
            cluster_labels_ev = None

        if not self.config.get("do_coref"):
            if self.config.get("classify_relations"):
                pair_ids = torch.LongTensor(get_pairwise_idxs(entity_spans.shape[1], entity_spans.shape[1])).cuda()
                entity_pairs = torch.gather(entity_spans, 1, pair_ids.view(1, -1, 1).expand(entity_spans.shape[0], -1, entity_spans.shape[2]))
                type_pred = type_pred.argmax(-1)
                type_pred_embed = self.type_embed(type_pred)
                entity_type_pairs = torch.gather(type_pred_embed, 1,
                                                 pair_ids.view(1, -1, 1).
                                                 expand(entity_spans.shape[0], -1, type_pred_embed.shape[2]))

                entity_pairs = torch.cat((entity_pairs, entity_type_pairs), dim=-1)

                entity_pairs = entity_pairs.view(entity_spans.shape[0], -1, entity_pairs.shape[2] * 2)
                relation_pred = self.relation_clf(entity_pairs)
            else:
                relation_pred = None

        return is_start_pred, len_from_here_pred, type_pred, cluster_labels,\
               is_start_pred_ev, len_from_here_pred_ev, type_pred_ev, cluster_labels_ev, coref_embed_ev,\
               relation_any, relation_cand, relation_true_for_cand, coref_embed, relation_pred,\
               attn_sum, evidence_true_for_cand



    def forward(self, batch: Batch, last_only: bool = True, epoch=0):
        gold_inputs = self.config.get("only_train_g_i")

        is_start, len_from_here, type_pred, cluster_labels, \
        is_start_pred_ev, len_from_here_pred_ev, type_pred_ev, cluster_labels_ev, coref_embed_ev, \
        relation_any, relation_cand, relation_true_for_cand,\
        coref_embeds, relation_pred,\
        attn_sum, evidence_true_for_cand = self.forward_nn(batch, epoch=epoch, gold_inputs=gold_inputs)

        #span_candidate_score, span_candidates_idxs, entity_type, trigger_type, relation_type, coref_embeds = self.forward_nn(batch)

        loss_names = []

        loss = []

        #TODO: also fix for bs > 1

        if not self.config.get("only_train_g_i"):

            entity_loss_start = self.criteria(is_start.view(-1, 2), batch.is_start.view(-1))

            #can use this if set 0-len to not count
            #gold_or_pred_start_one = elem_max(batch.is_start == 1., is_start.argmax(-1) == 1)

            gold_one = batch.is_start == 1

            entity_loss_len = self.criteria(len_from_here[gold_one].view(-1, 2),
                                            batch.len_from_here[gold_one].view(-1))
            entity_loss_type = self.criteria(type_pred.view(-1, len(self.vocabs["entity"])),
                                             batch.entity_labels[batch.entity_labels > 0])
                                             #batch.type_from_here[:].view(-1))

            loss = loss + [entity_loss_start, entity_loss_len, entity_loss_type]
            loss_names = loss_names + ["entity_start", "entity_len", "entity_type"]

            if self.config.get("classify_triggers"):
                trig_loss_start = self.criteria(is_start_pred_ev.view(-1, 2), batch.is_start_ev.view(-1))

                # can use this if set 0-len to not count
                # gold_or_pred_start_one = elem_max(batch.is_start == 1., is_start.argmax(-1) == 1)

                gold_one = batch.is_start_ev == 1

                trig_loss_len = self.criteria(len_from_here_pred_ev[gold_one].view(-1, 2),
                                              batch.len_from_here_ev[gold_one].view(-1))
                if type_pred_ev is not None:
                    trig_loss_type = self.criteria(type_pred_ev.view(-1, len(self.vocabs["event"])),
                                                   batch.trigger_labels[batch.trigger_labels > 0])
                else:
                    trig_loss_type = 0
                # batch.type_from_here[:].view(-1))

                loss = loss + [trig_loss_start, trig_loss_len, trig_loss_type]
                loss_names = loss_names + ["trig_start", "trig_len", "trig_type"]

        #entity_loss = self.criteria(entity_type.view(-1, len(self.vocabs["entity"])),
        #                               batch.entity_labels.view(1, -1, 1).gather(1, span_candidates_idxs).view(-1))

        """if not torch.isnan(entity_loss):
            loss.append(entity_loss)
            loss_names.append("entity")
        else:
            print('Entity loss is NaN')
            print(batch)"""

        if self.config.get("classify_evidence"):

            #evid_loss = -(evidence_true_for_cand * attn_sum).mean()
            evid_loss = -torch.clamp(attn_sum[evidence_true_for_cand == 1], max=1.2).mean()

            if self.config.get("relation_type_level") == "multitype":
                false_evid_true_rel = (evidence_true_for_cand == 0) * (relation_true_for_cand.unsqueeze(2) > 0)
            else:
                false_evid_true_rel = (evidence_true_for_cand == 0) * (relation_true_for_cand.unsqueeze(-1) > 0)

            evid_loss_neg = attn_sum[false_evid_true_rel].mean()

            #print(-attn_sum.mean(), evid_loss)

            loss.append(evid_loss)
            loss_names.append("evidence")

            loss.append(evid_loss_neg)
            loss_names.append("evidence_neg")


        if self.config.get("do_coref") and not self.config.get("only_train_g_i"):


            """mention_pair_ids = get_pairwise_idxs(coref_embeds.shape[1], coref_embeds.shape[1],
                                                 skip_diagonal=True, sep=False)

            random_pairs = random.sample(mention_pair_ids, min(len(mention_pair_ids),
                                                               coref_embeds.shape[1] * 50))

            are_coref = []

            for a, b in random_pairs:
                if batch.mention_to_ent_coref[0, a] == batch.mention_to_ent_coref[0, b]:
                    are_coref.append(1)
                else:
                    are_coref.append(0)

            are_coref = torch.LongTensor(are_coref).cuda()#.view(-1, 1).expand(-1, coref_embeds.shape[-1] * 2)

            random_pairs = torch.LongTensor(random_pairs).cuda()

            coref_embed_pairs = torch.gather(coref_embeds, 1, random_pairs.view(1, -1, 1).
                                            expand(coref_embeds.shape[0], -1, coref_embeds.shape[2])).\
                                            view(-1, 2 * coref_embeds.shape[-1])

            are_coref_pairs = coref_embed_pairs[are_coref == 1]
            are_not_coref_pairs = coref_embed_pairs[are_coref == 0]

            coref_dim = coref_embeds.shape[-1]

            are_coref_dist = torch.norm(are_coref_pairs[:, :coref_dim] -
                                                    are_coref_pairs[:, coref_dim:], dim=-1)
            #torch.cosine_similarity(are_coref_pairs[:, :coref_dim],
            #                                        are_coref_pairs[:, coref_dim:])
            are_not_coref_dist = torch.norm(are_not_coref_pairs[:, :coref_dim] -
                                                    are_not_coref_pairs[:, coref_dim:], dim=-1)
            #torch.cosine_similarity(are_not_coref_pairs[:, :coref_dim],
            #                                            are_not_coref_pairs[:, coref_dim:])

            coref_loss_attract = are_coref_dist.mean()
            coref_loss_repel = torch.clamp(-are_not_coref_dist, min=-3.).mean()"""

            coref_true_cluster_means = torch_scatter.scatter_mean(coref_embeds, batch.mention_to_ent_coref, dim=1)

            coref_true_cluster_aligned = torch.gather(coref_true_cluster_means, 1,
                                                      batch.mention_to_ent_coref.unsqueeze(-1).
                                                      expand(-1, -1, coref_true_cluster_means.shape[-1]))

            total_clusters = batch.mention_to_ent_coref.max() + 1
            if total_clusters > 1:
                random_shift = torch.randint(1, total_clusters, batch.mention_to_ent_coref.shape).cuda()
                false_clusters = (batch.mention_to_ent_coref + random_shift) % total_clusters

                coref_false_cluster_aligned = torch.gather(coref_true_cluster_means, 1,
                                                      false_clusters.unsqueeze(-1).expand(-1, -1, coref_true_cluster_means.shape[-1]))

                dist_to_false_cluster_center = torch.clamp(
                    5 - torch.norm(coref_embeds - coref_false_cluster_aligned, dim=-1), min=0)
            else:
                dist_to_false_cluster_center = torch.Tensor([0]).cuda()

            dist_to_true_cluster_center = torch.norm(coref_embeds - coref_true_cluster_aligned, dim=-1)

            coref_loss_attract = dist_to_true_cluster_center.mean()
            coref_loss_repel = dist_to_false_cluster_center.mean()

            if not torch.isnan(coref_loss_attract):
                loss.append(coref_loss_attract)
                loss_names.append("coref_attract")
                loss.append(coref_loss_repel)
                loss_names.append("coref_repel")
            else:
                print('Coref loss is NaN')
                print(batch)

            if self.config.get("classify_triggers") and coref_embed_ev is not None:

                coref_true_cluster_means = torch_scatter.scatter_mean(coref_embed_ev, batch.mention_to_ev_coref, dim=1)

                coref_true_cluster_aligned = torch.gather(coref_true_cluster_means, 1,
                                                          batch.mention_to_ev_coref.unsqueeze(-1).
                                                          expand(-1, -1, coref_true_cluster_means.shape[-1]))

                total_clusters = batch.mention_to_ev_coref.max() + 1
                if total_clusters > 1:
                    random_shift = torch.randint(1, total_clusters, batch.mention_to_ev_coref.shape).cuda()
                    false_clusters = (batch.mention_to_ev_coref + random_shift) % total_clusters

                    coref_false_cluster_aligned = torch.gather(coref_true_cluster_means, 1,
                                                               false_clusters.unsqueeze(-1).expand(-1, -1,
                                                                                                   coref_true_cluster_means.shape[
                                                                                                       -1]))

                    dist_to_false_cluster_center = torch.clamp(
                        5 - torch.norm(coref_embed_ev - coref_false_cluster_aligned, dim=-1), min=0)
                else:
                    dist_to_false_cluster_center = torch.Tensor([0]).cuda()

                dist_to_true_cluster_center = torch.norm(coref_embed_ev - coref_true_cluster_aligned, dim=-1)

                coref_loss_attract = dist_to_true_cluster_center.mean()
                coref_loss_repel = dist_to_false_cluster_center.mean()

                loss.append(coref_loss_attract)
                loss_names.append("trig_coref_attract")
                loss.append(coref_loss_repel)
                loss_names.append("trig_coref_repel")


        if self.config.get("classify_relations") and batch.relation_labels.shape[0] > 0:


            """batch_size = span_candidates_idxs.shape[0]
            span_num = span_candidates_idxs.shape[1]
            span_pair_src, span_pair_dest = get_pairwise_idxs_separate(span_candidates_idxs.shape[1], span_candidates_idxs.shape[1])
            span_pair_rel_labels = []

            for i in range(batch_size):
                pos_entity_rev_dict = {}

                for
                for j in range(len(span_pair_src)):
                        cur_rel = 0
                        if batch.entity_labels[span_pair_src] > 0 and batch.entity_labels[span_pair_dest] > 0:
                            cur_rel = 
                            
            span_rel_label_matrix = []"""

            #relation_pred = relation_pred.view(-1, relation_pred.shape[-1])

            relation_nonzero_loss = self.relation_nonzero_loss(relation_any.view(-1, 2),
                                                  (batch.relation_nonzero.view(-1)).long())

            #print("rels")
            #print("candidates:", relation_pred.shape[1])
            #print("predicted as non-zero:", (relation_pred.argmax(-1) > 0).long().sum())
            #print("gold:", (batch.relation_labels.view(-1) > 0).long().sum())

            """relation_any_cont_gold = elem_min(relation_any.argmax(-1).view(-1),
                                              (batch.relation_nonzero.view(-1)).long())

            #print("intersect:", relation_any_cont_gold.sum())

            rel_pred_correct = (relation_pred.view(-1, relation_pred.shape[-1]).argmax(-1) ==
                                relation_true_for_cand.view(-1))

            #print("predicted right:", rel_pred_correct.sum())

            rel_pred_correct_nonzero = elem_min(rel_pred_correct, (relation_true_for_cand.view(-1) > 0))"""

            #print("predicted right non-zero:", rel_pred_correct_nonzero.sum())

            if self.config.get("relation_type_level") == "multitype":
                relation_loss = self.relation_loss(relation_pred.view(-1),
                                          relation_true_for_cand.view(-1).float())
            else:
                relation_loss = self.relation_loss(relation_pred.view(-1, relation_pred.shape[-1]),
                                          relation_true_for_cand.view(-1))

            #relation_loss = self.relation_loss(relation_pred,
            #                                   batch.relation_labels.view(-1)[:relation_pred.shape[0]]) * 5.

            #print(relation_pred.argmax(-1))
            #print(batch.relation_labels.view(-1))
            #print(((relation_pred.argmax(-1) - batch.relation_labels.view(-1)) > 0).sum())

            if not torch.isnan(relation_loss):
                loss.append(relation_nonzero_loss)
                loss_names.append("relation_nonzero")
                loss.append(relation_loss)
                loss_names.append("relation")
            else:
                print('Relation loss is NaN')
                print(batch)

        if self.config.get("classify_roles"):
            if batch.role_labels.size(0):
                role_loss = self.role_loss(local_scores['role'], batch.role_labels)
                if not torch.isnan(role_loss):
                    loss.append(role_loss)
                    loss_names.append("role")
                else:
                    print('Role loss is NaN')
                    print(batch)


        return loss, loss_names

    def predict(self, batch: Batch, epoch=0, gold_inputs=False):
        self.eval()
        with torch.no_grad():
            result = self.forward_nn(batch, predict=True, epoch=epoch, gold_inputs=gold_inputs)

        self.train()
        return result
