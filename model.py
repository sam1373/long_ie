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
from util import label2onehot, elem_max, get_pairwise_idxs_separate, RegLayer

from span_transformer import SpanTransformer

import torch_scatter

import random

from sklearn.cluster import DBSCAN, OPTICS, AgglomerativeClustering


logger = logging.getLogger(__name__)


def calculate_enumeration_num(seq_len: int,
                              max_span_len: int) -> int:
    max_span_len = min(seq_len, max_span_len)
    return seq_len * max_span_len - ((max_span_len - 1) * max_span_len) // 2


def get_pairwise_idxs(num1: int, num2: int, skip_diagonal: bool = False):
    idxs = []
    for i in range(num1):
        for j in range(num2):
            if i == j and skip_diagonal:
                continue
            idxs.append(i)
            idxs.append(j)
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


class OneIEpp(nn.Module):
    """OneIE++ model."""
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
                 encoder_dropout_prob: float = 0.2,
                 ):
        super().__init__()

        self.config = config
        self.vocabs = vocabs
        self.encoder = encoder
        self.encoder_dropout = nn.Dropout(encoder_dropout_prob)

        token_initial_dim = self.encoder.config.hidden_size
        if self.config.get('use_extra_word_embed'):
            token_initial_dim += word_embed_dim

        #self.token_start_enc = Linears([token_initial_dim, 1024, 512])


        self.is_start_clf = Linears([token_initial_dim, hidden_dim, 2])
        self.len_from_here_clf = Linears([token_initial_dim, hidden_dim, self.config.max_entity_len])
        self.type_clf = Linears([token_initial_dim, hidden_dim, len(vocabs['entity'])])

        self.relation_clf = Linears([token_initial_dim * 2, hidden_dim, len(vocabs['relation'])])

        self.coref_embed = Linears([token_initial_dim, hidden_dim, coref_embed_dim])
        #self.type_from_here_clf = Linears([512, 128, len(vocabs['entity'])])

        #self.importance_score = Linears([token_initial_dim, 128, 1])


        #self.span_candidate_classifier = Linears([node_dim, 200, 200, 2],
        #                    dropout_prob=.2)

        #self.span_compress = nn.Linear(span_repr_dim, span_comp_dim)

        self.span_transformer = SpanTransformer(span_dim=token_initial_dim,
                                                vocabs=vocabs,
                                                final_pred_embeds=False,
                                                num_layers=span_transformer_layers)


        self.word_embed = word_embed
        self.use_extra_bert = config.get("use_extra_bert")
        self.extra_bert = extra_bert

        entity_weights = torch.ones(len(vocabs['entity'])).cuda()
        entity_weights[0] /= 100.

        event_weights = torch.ones(len(vocabs['event'])).cuda()
        #event_weights[0] /= 3.

        rel_weights = torch.ones(len(vocabs['relation'])).cuda()
        #rel_weights[0] /= 5.

        role_weights = torch.ones(len(vocabs['role'])).cuda()
        #role_weights[0] /= len(vocabs['role'])

        self.entity_loss = nn.CrossEntropyLoss(weight=entity_weights)
        self.event_loss = nn.CrossEntropyLoss(weight=event_weights)
        self.relation_loss = nn.CrossEntropyLoss(weight=rel_weights)
        self.role_loss = nn.CrossEntropyLoss(weight=role_weights)
        self.span_loss = nn.CrossEntropyLoss(weight=torch.Tensor([0.1, 1.]).cuda())
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
                    seq_token_mask.extend([1.0 / token_len] * token_len
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



    def calculate_loss(self,
                       scores: List[List[Dict[str, torch.Tensor]]],
                       target: torch.Tensor,
                       key: str,
                       last_only: bool = True):
        # Concatenate scores
        # layer_num = len(scores[0])
        # if last_only:
        # scores = [graph_scores[-1][key] for graph_scores in scores if key in graph_scores[-1]]
        # else:
        #     # TODO: check the calculation
        #     scores = [graph_scores[layer][key]
        #               for layer in range(layer_num)
        #               for graph_scores in scores
        #               if key in graph_scores[layer]]
        # scores = torch.cat(scores, dim=0)
        # Repeat the target labels based on the GNN layer number
        # if not last_only:

        if last_only:
            scores_ = []
            for graph_idx, graph_scores in enumerate(scores):
                graph_last_scores = graph_scores[-1]
                if key in graph_last_scores:
                    scores_.append(graph_last_scores[key])
            scores_ = torch.cat(scores_, dim=0)
            return self.criteria(scores_, target)
        else:
            layer_num = len(scores[0])
            scores_ = []
            for layer_idx in range(layer_num):
                for graph_scores in scores:
                    layer_scores= graph_scores[layer_idx]
                    if key in layer_scores:
                        scores_.append(layer_scores[key])
            scores_ = torch.cat(scores_, dim=0)
            target_ = target.repeat(layer_num)
            return self.criteria(scores_, target_)

    def forward_nn(self, batch: Batch, predict: bool = False, epoch=0):
        # Run the encoder to get contextualized word representations

        encoder_outputs = self.encode_bert(batch.pieces,
                                           attention_mask=batch.attention_mask,
                                           token_lens=batch.token_lens)
        #encoder_outputs = torch.zeros(batch.token_embed_ids.shape[:2] + (self.encoder.config.hidden_size,)).cuda()

        batch_size = encoder_outputs.size(0)

        if self.config.get('use_extra_word_embed'):
            word_embed_repr = self.word_embed(batch.token_embed_ids)#.detach()
            #if epoch >= 6:
            #word_embed_repr = word_embed_repr.detach()
            encoder_outputs = torch.cat((encoder_outputs, word_embed_repr), dim=-1)

        #encoded_starts = self.token_start_enc(encoder_outputs)

        is_start_pred = self.is_start_clf(encoder_outputs)
        len_from_here_pred = self.len_from_here_clf(encoder_outputs)
        #type_from_here_pred = self.type_from_here_clf(encoded_starts)

        #len_from_here_pred[is_start_pred.argmax(-1) == 0][:, 0] = 10000.
        #type_from_here_pred[is_start_pred.argmax(-1) == 0][:, 0] = 10000.

        #importance = self.importance_score(encoder_outputs)

        ##try only start tokens
        #V

        if not predict:

            max_entities = batch.is_start.sum(-1).max().item()

            entity_spans = torch.zeros(batch_size, max_entities, encoder_outputs.shape[-1]).cuda()

            for b in range(batch_size):
                for i in range(max_entities):
                    l, r = batch.pos_entity_offsets[b][i]
                    #importance_weight = importance[b, l:r, 0].softmax(-1)
                    #entity_spans[b, i] = torch.sum(torch.mul(encoder_outputs[b, l:r], importance_weight.unsqueeze(-1)), dim=0)
                    entity_spans[b, i] = encoder_outputs[b, l]
        else:

            max_entities = is_start_pred.argmax(-1).sum(-1).max().item()

            entity_spans = torch.zeros(batch_size, max_entities, encoder_outputs.shape[-1]).cuda()

            cur_ent = 0

            for b in range(batch_size):
                for i in range(is_start_pred.shape[1]):
                    if is_start_pred[b, i].argmax(-1) == 1:
                        l = i
                        #r = l + len_from_here_pred[b, i].argmax(-1)
                        #importance_weight = importance[b, l:r, 0].softmax(-1)
                        #entity_spans[b, cur_ent] = torch.sum(torch.mul(encoder_outputs[b, l:r], importance_weight.unsqueeze(-1)),
                        #                           dim=0)
                        entity_spans[b, cur_ent] = encoder_outputs[b, l]

                        cur_ent += 1

                        if cur_ent >= max_entities:
                            break

        entity_spans = self.span_transformer(entity_spans)

        if self.config.get("do_coref"):
            coref_embed = self.coref_embed(entity_spans)

            if predict:

                cluster_labels = torch.zeros(coref_embed.shape[:2]).cuda().long()

                for b in range(batch_size):
                    clustering = AgglomerativeClustering(distance_threshold=1., n_clusters=None).\
                        fit(coref_embed[b].detach().cpu().numpy())
                    cluster_labels[b] = torch.LongTensor(clustering.labels_).cuda()

                entity_means = torch_scatter.scatter_mean(entity_spans, cluster_labels, dim=1)

                entity_aggr_aligned = torch.gather(entity_means, 1,
                                                 cluster_labels.unsqueeze(-1).
                                                 expand(-1, -1, entity_means.shape[-1]))

                cluster_num = entity_means.shape[1]

            else:

                true_clusters = batch.mention_to_ent_coref

                cluster_num = true_clusters.max() + 1

                cluster_noise = torch.randint(0, cluster_num, true_clusters.shape).cuda()

                cluster_mask = torch.rand(true_clusters.shape).cuda() > 0.7

                noisy_clusters = torch.clone(true_clusters)

                noisy_clusters[cluster_mask] = cluster_noise[cluster_mask]

                entity_means = torch_scatter.scatter_mean(entity_spans, noisy_clusters, dim=1)

                if entity_means.shape[1] < cluster_num:
                    entity_means_pad = torch.zeros(batch_size, cluster_num, entity_means.shape[-1]).cuda()
                    entity_means_pad[:, :entity_means.shape[1]] = entity_means
                    entity_means = entity_means_pad

                entity_aggr_aligned = torch.gather(entity_means, 1,
                                                 noisy_clusters.unsqueeze(-1).
                                                 expand(-1, -1, entity_means.shape[-1]))

            #print()
            #cluster_num = entity_means.shape[1]

            if self.config.get("classify_relations"):
                pair_ids = torch.LongTensor(get_pairwise_idxs(cluster_num, cluster_num)).cuda()
                entity_pairs = torch.gather(entity_means, 1, pair_ids.view(1, -1, 1).expand(entity_spans.shape[0], -1,
                                                                                            entity_spans.shape[2]))
                entity_pairs = entity_pairs.view(entity_spans.shape[0], -1, entity_spans.shape[2] * 2)
                relation_pred = self.relation_clf(entity_pairs)
            else:
                relation_pred = None

            entity_spans = entity_aggr_aligned

        else:
            coref_embed = None

        type_pred = self.type_clf(entity_spans)

        if not self.config.get("do_coref"):
            if self.config.get("classify_relations"):
                pair_ids = torch.LongTensor(get_pairwise_idxs(entity_spans.shape[1], entity_spans.shape[1])).cuda()
                entity_pairs = torch.gather(entity_spans, 1, pair_ids.view(1, -1, 1).expand(entity_spans.shape[0], -1, entity_spans.shape[2]))
                entity_pairs = entity_pairs.view(entity_spans.shape[0], -1, entity_spans.shape[2] * 2)
                relation_pred = self.relation_clf(entity_pairs)
            else:
                relation_pred = None

        return is_start_pred, len_from_here_pred, type_pred, coref_embed, relation_pred



    def forward(self, batch: Batch, last_only: bool = True, epoch=0):
        is_start, len_from_here, type_pred, coref_embeds, relation_pred = self.forward_nn(batch, epoch=epoch)
        #span_candidate_score, span_candidates_idxs, entity_type, trigger_type, relation_type, coref_embeds = self.forward_nn(batch)

        loss_names = []

        loss = []

        if self.config.get("classify_triggers"):
            span_candidate_loss = self.span_loss(span_candidate_score.view(-1, 2),
                                                 (elem_max(batch.entity_labels > 0, batch.trigger_labels > 0)).\
                                                 type(torch.LongTensor).cuda())



        #TODO: also fix for bs > 1

        entity_loss_start = self.criteria(is_start.view(-1, 2), batch.is_start.view(-1))

        gold_start_one = (batch.is_start == 1.)

        entity_loss_len = self.criteria(len_from_here[gold_start_one].view(-1, self.config.max_entity_len),
                                        batch.len_from_here[gold_start_one].view(-1))
        entity_loss_type = self.criteria(type_pred.view(-1, len(self.vocabs["entity"])),
                                         batch.entity_labels[batch.entity_labels > 0])
                                         #batch.type_from_here[:].view(-1))

        loss = loss + [entity_loss_start, entity_loss_len, entity_loss_type]
        loss_names = loss_names + ["entity_start", "entity_len", "entity_type"]

        #entity_loss = self.criteria(entity_type.view(-1, len(self.vocabs["entity"])),
        #                               batch.entity_labels.view(1, -1, 1).gather(1, span_candidates_idxs).view(-1))

        """if not torch.isnan(entity_loss):
            loss.append(entity_loss)
            loss_names.append("entity")
        else:
            print('Entity loss is NaN')
            print(batch)"""

        if self.config.get("do_coref"):

            #process coref_embeds
            #
            # get avg for clusters
            # get avg between clusters
            # get dist losses
            # also produce predictions

            coref_true_cluster_means = torch_scatter.scatter_mean(coref_embeds, batch.mention_to_ent_coref, dim=1)

            #coref_true_cluster_scattered = torch.zeros(coref_embeds.shape).cuda()

            coref_true_cluster_aligned = torch.gather(coref_true_cluster_means, 1,
                                                      batch.mention_to_ent_coref.unsqueeze(-1).expand(-1, -1, coref_true_cluster_means.shape[-1]))



            #random_shift = random.randint(0, coref_embeds.shape[1])
            #shifted_clusters =  torch.cat((batch.mention_to_ent_coref[:, random_shift:],
            #                               batch.mention_to_ent_coref[:, random_shift]), dim=1)

            total_clusters = batch.mention_to_ent_coref.max() + 1
            random_shift = torch.randint(1, total_clusters, batch.mention_to_ent_coref.shape).cuda()
            false_clusters = (batch.mention_to_ent_coref + random_shift) % total_clusters

            coref_false_cluster_aligned = torch.gather(coref_true_cluster_means, 1,
                                                      false_clusters.unsqueeze(-1).expand(-1, -1, coref_true_cluster_means.shape[-1]))

            dist_to_true_cluster_center = torch.norm(coref_embeds - coref_true_cluster_aligned, dim=-1)
            dist_to_false_cluster_center = torch.clamp(5 - torch.norm(coref_embeds - coref_false_cluster_aligned, dim=-1), min=0) ** 2

            #avg_of_clusters = torch.mean(coref_true_cluster_means, dim=-2)

            #dist_to_avg_of_clusters = torch.norm(coref_true_cluster_means - avg_of_clusters, dim=-1)

            coref_loss_attract = dist_to_true_cluster_center.mean()
            coref_loss_repel = dist_to_false_cluster_center.mean()
            #coref_cluster_loss = dist_to_true_cluster_center.mean() - dist_to_avg_of_clusters.mean()

            #coref_pred = coref_pred.view(-1, coref_pred.shape[-1])

            #coref_loss = self.span_loss(coref_pred,
            #                                   batch.coref_labels[:coref_pred.shape[0]]) * 10.

            if not torch.isnan(coref_loss_attract):
                loss.append(coref_loss_attract)
                loss_names.append("coref_attract")
                loss.append(coref_loss_repel)
                loss_names.append("coref_repel")
            else:
                print('Coref loss is NaN')
                print(batch)

        if self.config.get("classify_triggers"):
            trigger_loss = self.event_loss(trigger_type.view(-1, len(self.vocabs["event"])),
                                             batch.trigger_labels.view(1, -1, 1).gather(1, span_candidates_idxs).view(-1))

            if not torch.isnan(trigger_loss):
                loss.append(trigger_loss)
                loss_names.append("trigger")
            else:
                print('Trigger loss is NaN')
                print(batch)

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

            relation_pred = relation_pred.view(-1, relation_pred.shape[-1])

            relation_loss = self.criteria(relation_pred,
                                               batch.relation_labels.view(-1)[:relation_pred.shape[0]])

            print(relation_loss)

            if not torch.isnan(relation_loss):
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

    def predict(self, batch: Batch, epoch=0):
        self.eval()
        with torch.no_grad():
            result = self.forward_nn(batch, predict=True, epoch=epoch)

        self.train()
        return result
