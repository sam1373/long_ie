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
            idxs.append(j + num1)
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
                 config: Dict[str, Any],
                 vocabs: Dict[str, Dict[str, int]],
                 encoder: nn.Module,
                 node_dim: int,
                 entity_classifier: nn.Module = None,
                 mention_classifier: nn.Module = None,
                 event_classifier: nn.Module = None,
                 relation_classifier: nn.Module = None,
                 role_classifier: nn.Module = None,
                 span_len_embed:nn.Module = None,
                 word_embed:nn.Module = None,
                 gnn: nn.Module = None,
                 extra_bert: int = 0,
                 span_repr_dim: int = 512,
                 span_comp_dim: int = 512,
                 ):
        super().__init__()

        self.config = config
        self.vocabs = vocabs
        self.encoder = encoder
        self.encoder_dropout = nn.Dropout(0.2)

        self.span_candidate_classifier = Linears([node_dim, 200, 200, 2],
                            dropout_prob=.2)

        self.span_compress = nn.Linear(span_repr_dim, span_comp_dim)

        self.span_transformer = SpanTransformer(span_dim=span_comp_dim,
                                                vocabs=vocabs,
                                                final_pred_embeds=False,
                                                num_layers=3)

        self.entity_classifier = entity_classifier
        self.mention_classifier = mention_classifier
        self.event_classifier = event_classifier
        self.relation_classifier = relation_classifier
        self.role_classifier = role_classifier
        self.span_len_embed = span_len_embed
        self.word_embed = word_embed
        self.gnn = gnn
        self.use_extra_bert = config.get("use_extra_bert")
        self.extra_bert = extra_bert

        entity_weights = torch.ones(len(vocabs['entity'])).cuda()
        entity_weights[0] /= 100.

        event_weights = torch.ones(len(vocabs['event'])).cuda()
        event_weights[0] /= 3.

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
                      .expand(batch_size, -1, bert_dim)) + 1
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

        return bert_outputs, seq_repr

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

    def predict_spans(self,
                      span_scores: torch.Tensor,
                      mask: torch.Tensor):
        batch_size = span_scores.size(0)
        # Predict labels
        span_scores = span_scores.softmax(2)
        _, predicted = span_scores.max(2)
        # Mask invalid spans
        predicted = predicted * mask
        predicted = predicted.nonzero(as_tuple=False)

        predicted_idxs = [[] for _ in range(batch_size)]
        for i, j in predicted.tolist():
            predicted_idxs[i].append(j)
        span_nums = [len(x) for x in predicted_idxs]
        max_span_num = max(1, max(span_nums))
        predicted_idxs = [x + [0] * (max_span_num - len(x))
                          for x in predicted_idxs]
        predicted_idxs = mask.new_tensor(predicted_idxs)

        return predicted_idxs, span_nums

    def get_local_scores(self,
                         encoder_outputs: torch.Tensor,
                         batch,
                         predict: bool = False):
        batch_size, seq_len, _ = encoder_outputs.size()
        # Get entity representations
        entity_repr = self.get_span_representations(encoder_outputs,
                                                    batch.entity_idxs,
                                                    batch.entity_mask,
                                                    batch.entity_lens,
                                                    batch.entity_boundaries,
                                                    batch.max_entity_len)
        repr_dim = entity_repr.size(-1)
        # Calculate entity type scores
        entity_score = self.entity_classifier(entity_repr)
        # Calculate mention type scores
        mention_score = self.mention_classifier(entity_repr)

        # Get trigger representations
        trigger_repr = self.get_span_representations(encoder_outputs,
                                                     batch.trigger_idxs,
                                                     batch.trigger_mask,
                                                     batch.trigger_lens,
                                                     batch.trigger_boundaries,
                                                     batch.max_trigger_len)
        # Calculate event type scores
        event_score = self.event_classifier(trigger_repr)

        # Select positive entities
        if predict:
            pos_entity_idxs, entity_span_nums = self.predict_spans(entity_score,
                                                                   batch.entity_span_mask)
        else:
            pos_entity_idxs = batch.pos_entity_idxs
        pos_entity_idxs_exp = pos_entity_idxs.unsqueeze(-1).expand(-1, -1, repr_dim)
        pos_entity_repr = entity_repr.gather(1, pos_entity_idxs_exp)
        entity_num = pos_entity_repr.size(1)

        # Select positive triggers
        if predict:
            pos_trigger_idxs, trigger_span_nums = self.predict_spans(event_score,
                                                                     batch.trigger_span_mask)
        else:
            pos_trigger_idxs = batch.pos_trigger_idxs
        pos_trigger_idxs_exp = pos_trigger_idxs.unsqueeze(-1).expand(-1, -1, repr_dim)
        pos_trigger_repr = trigger_repr.gather(1, pos_trigger_idxs_exp)
        trigger_num = pos_trigger_repr.size(1)

        # Get relation representation
        # TODO: transform before concatenate
        relation_idxs = get_pairwise_idxs(entity_num, entity_num, True)
        relation_idxs = (pos_entity_idxs.new(relation_idxs)
                         .unsqueeze(0)
                         .unsqueeze(-1)
                         .expand(batch_size, -1, repr_dim))
        relation_repr = (torch.cat([pos_entity_repr, pos_entity_repr], dim=1)
                         .gather(1, relation_idxs)
                         .view(batch_size, -1, 2 * repr_dim))
        # Calculate relation scores
        relation_score = self.relation_classifier(relation_repr)
        # Get arg role representation
        role_idxs = get_pairwise_idxs(trigger_num, entity_num)
        role_idxs = (pos_entity_idxs.new(role_idxs)
                     .unsqueeze(0)
                     .unsqueeze(-1)
                     .expand(batch_size, -1, repr_dim))
        role_repr = (torch.cat([pos_trigger_repr, pos_entity_repr], dim=1)
                     .gather(1, role_idxs)
                     .view(batch_size, -1, 2 * repr_dim))
        # Calculate role scores
        role_score = self.role_classifier(role_repr)

        entity_score = entity_score.view(-1, entity_score.size(-1))
        mention_score = mention_score.view(-1, mention_score.size(-1))
        event_score = event_score.view(-1, event_score.size(-1))
        relation_score = relation_score.view(-1, relation_score.size(-1))
        role_score = role_score.view(-1, role_score.size(-1))

        if predict:
            return (entity_score, event_score, relation_score, role_score,
                    mention_score, pos_entity_idxs, pos_trigger_idxs,
                    entity_span_nums, trigger_span_nums)
        else:
            return (entity_score, event_score, relation_score, role_score,
                    mention_score, pos_entity_idxs, pos_trigger_idxs)

    def build_graphs(self,
                     entity_nums: List[int],
                     trigger_nums: List[int],
                     device):
        # print('build graph entity nums', entity_nums)
        graphs = []
        for idx, (entity_num, trigger_num) in enumerate(
                zip(entity_nums, trigger_nums)):
            # Construct a complete graph
            entity_nodes = torch.arange(entity_num)
            trigger_nodes = torch.arange(trigger_num)

            relation_src_nodes = [i for i in range(entity_num) for _ in range(entity_num - 1)]
            relation_dst_nodes = [j for i in range(entity_num) for j in range(entity_num) if i != j]
            relation_src_nodes = torch.LongTensor(relation_src_nodes)
            relation_dst_nodes = torch.LongTensor(relation_dst_nodes)

            role_src_nodes = [i for i in range(trigger_num) for _ in range(entity_num)]
            role_dst_nodes = [j for _ in range(trigger_num) for j in range(entity_num)]
            role_src_nodes = torch.LongTensor(role_src_nodes)
            role_dst_nodes = torch.LongTensor(role_dst_nodes)

            data_dict = {
                ('entity', 'relation', 'entity'): (relation_src_nodes, relation_dst_nodes),
                ('trigger', 'role', 'entity'): (role_src_nodes, role_dst_nodes),
                ('entity', 'role_rev', 'trigger'): (role_dst_nodes, role_src_nodes),

                # Add self links
                ('entity', 'entity_self', 'entity'): (entity_nodes, entity_nodes),
                ('trigger', 'trigger_self', 'trigger'): (trigger_nodes,
                                                         trigger_nodes)
            }
            graph = dgl.heterograph(data_dict)
            graphs.append(graph.to(device))
        return graphs

    def run_gnn(self,
                graphs: List[dgl.DGLHeteroGraph],
                reprs: Dict[str, torch.Tensor],
                scores: Dict[str, torch.Tensor],
                entity_nums: List[int],
                trigger_nums: List[int],
                predict: bool=False) -> List[List[Dict[str, torch.Tensor]]]:
        """Run the GNN to update node and edge features.

        Args:
            graphs (List[dgl.DGLHeteroGraph]): A list of DGLHeteroGraph objects.
            reprs (Dict[str, torch.Tensor]): A dict of node representation tensors.
            scores (Dict[str, torch.Tensor]): A dict of node and edge score tensors.
            entity_nums (List[int]): A list of entity numbers.
            trigger_nums (List[int]): A list of trigger numbers.

        Returns:
            List[List[Dict[str, torch.Tensor]]]: A list of lists of predicted score dicts. The i-th list is the score
            list for the i-th graph, where the j-th element is a score dict predicted by the j-th GNN layer.
        """
        all_scores = []
        for graph_idx, graph in enumerate(graphs):
            entity_num = entity_nums[graph_idx]
            trigger_num = trigger_nums[graph_idx]
            nfeats, efeats = {}, {}
            if entity_num:
                entity_repr = reprs['entity'][graph_idx][:entity_num]
                entity_score = scores['entity'][graph_idx][:entity_num]
                nfeats['entity'] = torch.cat([entity_repr, entity_score], dim=1)
                # Relation
                if entity_num > 1:
                    if predict:
                        relation_score = scores['relation'][graph_idx][:entity_num, :entity_num - 1]
                        relation_score = relation_score.contiguous().view(-1, relation_score.size(-1))
                    else:
                        relation_score = scores['relation'][graph_idx]
                    efeats['relation'] = relation_score

            if trigger_num:
                trigger_repr = reprs['trigger'][graph_idx][:trigger_num]
                trigger_score = scores['trigger'][graph_idx][:trigger_num]
                nfeats['trigger'] = torch.cat([trigger_repr, trigger_score], dim=1)
                if entity_num:
                    if predict:
                        role_score = scores['role'][graph_idx][:trigger_num, :entity_num]
                        role_score = role_score.contiguous().view(-1, role_score.size(-1))
                    else:
                        role_score = scores['role'][graph_idx]
                    efeats['role'] = role_score
                    efeats['role_rev'] = role_score

            # Edge features
            all_scores.append(self.gnn(graph, nfeats, efeats))
        return all_scores

    def reshape_local_scores(self,
                             entity_span_score,
                             trigger_span_score,
                             relation_score,
                             role_score,
                             entity_nums,
                             trigger_nums):
        # entity_span_score_ = entity_span_score
        # trigger_span_score_ = trigger_span_score
        entity_span_score = entity_span_score.view(-1, entity_span_score.size(-1))
        trigger_span_score = trigger_span_score.view(-1, trigger_span_score.size(-1))
        relation_score_list = []
        role_score_list = []
        if self.config.get("classify_relations") or self.config.get("classify_roles"):
            for idx, (entity_num, trigger_num) in enumerate(zip(entity_nums, trigger_nums)):
                if entity_num:
                    if self.config.get("classify_relations"):
                        inst_relation_score = relation_score[idx][:entity_num, :entity_num - 1]
                        inst_relation_score = inst_relation_score.contiguous().view(-1, inst_relation_score.size(-1))
                        relation_score_list.append(inst_relation_score)
                    if trigger_num:
                        if self.config.get("classify_roles"):
                            inst_role_score = role_score[idx][:trigger_num, :entity_num]
                            inst_role_score = inst_role_score.contiguous().view(-1, inst_role_score.size(-1))
                            role_score_list.append(inst_role_score)
                    else:
                        role_score_list.append(None)
                else:
                    relation_score_list.append(None)
                    role_score_list.append(None)

        relation_score_list_ = [x for x in relation_score_list if x is not None]
        relation_score = torch.cat(relation_score_list_, dim=0) if relation_score_list_ else None

        role_score_list_ = [x for x in role_score_list if x is not None]
        role_score = torch.cat(role_score_list_, dim=0) if role_score_list_ else None

        return {
            'entity': entity_span_score,
            # 'entity_sep': entity_span_score_,
            'trigger': trigger_span_score,
            # 'trigger_sep': trigger_span_score_,
            'relation': relation_score,
            # 'relation_sep': relation_score_list,
            'role': role_score,
            # 'role_sep': role_score_list,
        }

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

    def forward_nn(self, batch: Batch, predict: bool = False):
        # Run the encoder to get contextualized word representations
        encoder_outputs, seq_repr = self.encode_bert(batch.pieces,
                                           attention_mask=batch.attention_mask,
                                           token_lens=batch.token_lens)
        batch_size = encoder_outputs.size(0)
        # Generate entity and trigger span representations
        entity_span_repr = self.get_span_representations(encoder_outputs,
                                                         batch.entity_idxs,
                                                         batch.entity_mask,
                                                         batch.entity_lens,
                                                         batch.entity_boundaries,
                                                         batch.token_embed_ids,
                                                         batch.max_entity_len)

        """trigger_span_repr = self.get_span_representations(encoder_outputs,
                                                          batch.trigger_idxs,
                                                          batch.trigger_mask,
                                                          batch.trigger_lens,
                                                          batch.trigger_boundaries,
                                                          batch.token_embed_ids,
                                                          batch.max_trigger_len)"""
        repr_dim = entity_span_repr.size(-1)

        span_candidate_score = self.span_candidate_classifier(entity_span_repr)

        if not predict:

            span_candidate_score_with_true = span_candidate_score.clone()

            score_shape = span_candidate_score.shape[:2] + (1,)

            scores_for_true_spans = torch.cat((torch.zeros(score_shape), torch.ones(score_shape) * 100.), dim=-1).cuda()

            span_candidate_score_with_true.scatter_(1, batch.pos_entity_idxs.unsqueeze(-1).expand(-1, -1, 2),
                                                   scores_for_true_spans)

            span_candidates_idxs = span_candidate_score_with_true.max(2)[1].nonzero()[:, 1]

        else:
            span_candidates_idxs = span_candidate_score.max(2)[1].nonzero()[:, 1]

            if span_candidates_idxs.shape[-1] == 0:
                span_candidates_idxs = torch.cuda.LongTensor([0])

        span_candidates_idxs = span_candidates_idxs.reshape(1, -1, 1)

        #bs, span_num, 1
        #contains span pos id

        #TODO: change to work with larger batch size

        span_candidate_repr = entity_span_repr.gather(
            1, span_candidates_idxs.expand(-1, -1, repr_dim))

        #bs, span_num, span_dim

        span_candidate_repr = self.span_compress(span_candidate_repr)


        true_spans = None

        if not predict:
            true_entity_labels = batch.entity_labels.view(1, -1, 1).gather(1, span_candidates_idxs).view(-1)

            true_spans = true_entity_labels.nonzero()

            true_spans = true_spans.view(1, -1)

        entity_type, trigger_type, relation_type, coref_pred = self.span_transformer(span_candidate_repr,
                                                                         predict=predict,
                                                                         true_spans=true_spans,
                                                                         batch=batch,
                                                                         span_cand_idxs=span_candidates_idxs)


        return span_candidate_score, span_candidates_idxs, entity_type, trigger_type, relation_type, coref_pred

        # Calculate span label scores
        entity_span_score = self.entity_classifier(entity_span_repr)
        entity_label_size = entity_span_score.size(-1)
        trigger_span_score = self.event_classifier(entity_span_repr)
        trigger_label_size = trigger_span_score.size(-1)

        if predict:
            # In evaluation phase, predict the indices of valid spans
            entity_idxs, entity_nums = self.predict_spans(entity_span_score,
                                                          batch.entity_span_mask)
            trigger_idxs, trigger_nums = self.predict_spans(trigger_span_score,
                                                            batch.trigger_span_mask)
        else:
            # In training phase, using spans in the annotations
            entity_idxs = batch.pos_entity_idxs
            trigger_idxs = batch.pos_trigger_idxs
            entity_nums = [len(x) for x in batch.pos_entity_offsets]
            trigger_nums = [len(x) for x in batch.pos_trigger_offsets]
        entity_idxs_exp = entity_idxs.unsqueeze(-1)
        trigger_idxs_exp = trigger_idxs.unsqueeze(-1)

        # Select entity representation
        entity_repr = entity_span_repr.gather(
            1, entity_idxs_exp.expand(-1, -1, repr_dim))
        entity_score = entity_span_score.gather(
            1, entity_idxs_exp.expand(-1, -1, entity_label_size))
        mention_score = self.mention_classifier(entity_repr)
        entity_num = entity_repr.size(1)

        # Select trigger representation
        trigger_repr = entity_span_repr.gather(
            1, trigger_idxs_exp.expand(-1, -1, repr_dim))
        event_score = trigger_span_score.gather(
            1, trigger_idxs_exp.expand(-1, -1, trigger_label_size))
        trigger_num = trigger_repr.size(1)

        rel_score, role_score = None, None

        if self.config.get("classify_relations"):
            # Generate relation representations
            rel_src_repr = self.relation_classifier.forward_src(entity_repr)
            rel_dst_repr = self.relation_classifier.forward_dst(entity_repr)
            rel_hid_dim = rel_src_repr.size(-1)
            rel_src_idxs, rel_dst_idxs = get_pairwise_idxs_separate(entity_num,
                                                                    entity_num,
                                                                    True)
            rel_src_idxs = (entity_idxs.new_tensor(rel_src_idxs)
                            .unsqueeze(0).unsqueeze(-1)
                            .expand(batch_size, -1, rel_hid_dim))
            rel_dst_idxs = (entity_idxs.new_tensor(rel_dst_idxs)
                            .unsqueeze(0).unsqueeze(-1)
                            .expand(batch_size, -1, rel_hid_dim))
            rel_src_repr = (rel_src_repr
                            .gather(1, rel_src_idxs)
                            .view(batch_size, entity_num, entity_num - 1, rel_hid_dim))
            rel_dst_repr = (rel_dst_repr
                            .gather(1, rel_dst_idxs)
                            .view(batch_size, entity_num, entity_num - 1, rel_hid_dim))
            rel_score = self.relation_classifier.forward_output(rel_src_repr,
                                                                rel_dst_repr)
        if self.config.get("classify_roles"):
            # Generate event-argument link (role) representations
            role_src_repr = self.role_classifier.forward_src(trigger_repr)
            role_dst_repr = self.role_classifier.forward_dst(entity_repr)
            role_hid_dim = role_src_repr.size(-1)
            role_src_idxs, role_dst_idxs = get_pairwise_idxs_separate(trigger_num,
                                                                      entity_num)
            role_src_idxs = (entity_idxs.new_tensor(role_src_idxs)
                             .unsqueeze(0).unsqueeze(-1)
                             .expand(batch_size, -1, role_hid_dim))
            role_dst_idxs = (entity_idxs.new_tensor(role_dst_idxs)
                             .unsqueeze(0).unsqueeze(-1)
                             .expand(batch_size, -1, role_hid_dim))
            role_src_repr = (role_src_repr
                             .gather(1, role_src_idxs)
                             .view(batch_size, trigger_num, entity_num, role_hid_dim))
            role_dst_repr = (role_dst_repr
                             .gather(1, role_dst_idxs)
                             .view(batch_size, trigger_num, entity_num, role_hid_dim))
            role_score = self.role_classifier.forward_output(role_src_repr,
                                                             role_dst_repr)


        local_scores = self.reshape_local_scores(
                entity_span_score, trigger_span_score, rel_score, role_score,
                entity_nums, trigger_nums)

        if predict:
            scores = {'entity': entity_score,
                      'trigger': event_score,
                      'relation': rel_score,
                      'role': role_score}
        else:
            # Use gold label scores
            entity_labels = batch.entity_labels_sep
            trigger_labels = batch.trigger_labels_sep
            relation_labels = batch.relation_labels_sep
            role_labels = batch.role_labels_sep
            entity_label_size = self.label_size('entity')
            trigger_label_size = self.label_size('event')
            relation_label_size = self.label_size('relation')
            role_label_size = self.label_size('role')
            scores = {
                'entity': [label2onehot(x, entity_label_size) for x in entity_labels],
                'trigger': [label2onehot(x, trigger_label_size) for x in trigger_labels],
                'relation': [label2onehot(x, relation_label_size) for x in relation_labels],
                'role': [label2onehot(x, role_label_size) for x in role_labels]
            }

        # Construct graphs
        # Add all node / edge types
        if self.gnn is not None:
            graphs = self.build_graphs(entity_nums=entity_nums,
                                       trigger_nums=trigger_nums,
                                       device=encoder_outputs.device)
            reprs = {'entity': entity_repr,
                     'trigger': trigger_repr,}

            # Run HeteroRGCN
            # Relation score and role score should be transformed
            gnn_scores = self.run_gnn(graphs, reprs, scores, entity_nums, trigger_nums,
                                      predict=predict)

            if predict:
                return gnn_scores, local_scores, entity_idxs, entity_nums, trigger_idxs, trigger_nums, span_candidate_score
            else:
                # local_scores = self.reshape_local_scores(
                #     entity_span_score, trigger_span_score, rel_score, role_score,
                #     entity_nums, trigger_nums)
                return gnn_scores, local_scores
        else:
            if predict:
                return scores, local_scores, entity_idxs, entity_nums, trigger_idxs, trigger_nums, span_candidate_score
            else:
                return scores, local_scores, span_candidate_score

    def forward(self, batch: Batch, last_only: bool = True):
        span_candidate_score, span_candidates_idxs, entity_type, trigger_type, relation_type, coref_pred = self.forward_nn(batch)

        loss_names = []

        loss = []

        span_candidate_loss = self.span_loss(span_candidate_score.view(-1, 2),
                                             #(batch.entity_labels > 0).\
                                             (elem_max(batch.entity_labels > 0, batch.trigger_labels > 0)).\
                                             type(torch.LongTensor).cuda())

        if not torch.isnan(span_candidate_loss):
            loss.append(span_candidate_loss)
            loss_names.append("span_candidate")
        else:
            print('span_candidate_loss is NaN')
            print(batch)

        #TODO: also fix for bs > 1

        entity_loss = self.criteria(entity_type.view(-1, len(self.vocabs["entity"])),
                                       batch.entity_labels.view(1, -1, 1).gather(1, span_candidates_idxs).view(-1))

        if not torch.isnan(entity_loss):
            loss.append(entity_loss)
            loss_names.append("entity")
        else:
            print('Entity loss is NaN')
            print(batch)

        if self.config.get("do_coref"):
            coref_pred = coref_pred.view(-1, coref_pred.shape[-1])

            coref_loss = self.span_loss(coref_pred,
                                               batch.coref_labels[:coref_pred.shape[0]])

            if not torch.isnan(coref_loss):
                loss.append(coref_loss)
                loss_names.append("coref")
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

            relation_type = relation_type.view(-1, relation_type.shape[-1])

            relation_loss = self.relation_loss(relation_type,
                                               batch.relation_labels[:relation_type.shape[0]]) * 5.

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

        if self.gnn is not None:
            if batch.id_entity_labels.size(0):
                gnn_entity_loss = self.calculate_loss(gnn_scores,
                                                      batch.id_entity_labels,
                                                      'entity',
                                                      last_only=last_only)
                if not torch.isnan(gnn_entity_loss):
                    loss.append(gnn_entity_loss)
                else:
                    print('GNN entity loss is NaN')
                    print(batch)
            if batch.relation_labels.size(0):
                gnn_relation_loss = self.calculate_loss(gnn_scores,
                                                        batch.relation_labels,
                                                        'relation',
                                                        last_only=last_only)
                if not torch.isnan(gnn_relation_loss):
                    loss.append(gnn_relation_loss)
                else:
                    print('GNN relation loss is NaN')
                    print(batch)
            if batch.id_trigger_labels.size(0):
                gnn_trigger_loss = self.calculate_loss(gnn_scores,
                                                       batch.id_trigger_labels,
                                                       'trigger',
                                                       last_only=last_only)
                if not torch.isnan(gnn_trigger_loss):
                    loss.append(gnn_trigger_loss)
                else:
                    print('GNN trigger loss is NaN')
                    print(batch)
            if batch.role_labels.size(0):
                gnn_role_loss = self.calculate_loss(gnn_scores,
                                                    batch.role_labels,
                                                    'role',
                                                    last_only=last_only)
                if not torch.isnan(gnn_role_loss):
                    loss.append(gnn_role_loss)
                else:
                    print('GNN role loss is NaN')
                    print(batch)

        #loss = sum(loss) if loss else None
        return loss, loss_names

    def predict(self, batch: Batch):
        self.eval()
        with torch.no_grad():
            result = self.forward_nn(batch, predict=True)

        self.train()
        return result
