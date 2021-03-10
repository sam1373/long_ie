import os
import json
import glob
import lxml.etree as et
from nltk import word_tokenize, sent_tokenize
from copy import deepcopy

import torch
from torch import nn

from graph import Graph
from random import random

from collections import defaultdict

import torch.nn.functional as F


def argmax(lst):
    max_idx = -1
    max_value = -100000
    for i, v in enumerate(lst):
        if v > max_value:
            max_idx = i
            max_value = v
    return max_idx, max_value


def generate_vocabs(datasets, coref=False,
                    relation_directional=False,
                    symmetric_relations=None):
    """Generate vocabularies from a list of data sets
    :param datasets (list): A list of data sets
    :return (dict): A dictionary of vocabs
    """
    entity_type_set = set()
    event_type_set = set()
    relation_type_set = set()
    role_type_set = set()
    for dataset in datasets:
        entity_type_set.update(dataset.entity_type_set)
        event_type_set.update(dataset.event_type_set)
        relation_type_set.update(dataset.relation_type_set)
        role_type_set.update(dataset.role_set)

    # add inverse relation types for non-symmetric relations
    if relation_directional:
        if symmetric_relations is None:
            symmetric_relations = []
        relation_type_set_ = set()
        for relation_type in relation_type_set:
            relation_type_set_.add(relation_type)
            if relation_directional and relation_type not in symmetric_relations:
                relation_type_set_.add(relation_type + '_inv')

    # entity and trigger labels
    """prefix = ['B', 'I']
    entity_label_stoi = {'O': 0}
    trigger_label_stoi = {'O': 0}
    for t in entity_type_set:
        for p in prefix:
            entity_label_stoi['{}-{}'.format(p, t)] = len(entity_label_stoi)
    for t in event_type_set:
        for p in prefix:
            trigger_label_stoi['{}-{}'.format(p, t)] = len(trigger_label_stoi)"""

    entity_type_stoi = {k: i for i, k in enumerate(entity_type_set, 1)}
    entity_type_stoi['O'] = 0

    event_type_stoi = {k: i for i, k in enumerate(event_type_set, 1)}
    event_type_stoi['O'] = 0

    relation_type_stoi = {k: i for i, k in enumerate(relation_type_set, 1)}
    relation_type_stoi['O'] = 0
    if coref:
        relation_type_stoi['COREF'] = len(relation_type_stoi)

    role_type_stoi = {k: i for i, k in enumerate(role_type_set, 1)}
    role_type_stoi['O'] = 0

    mention_type_stoi = {'NAM': 0, 'NOM': 1, 'PRO': 2, 'UNK': 3, 'TIME': 4, 'VALUE': 5}

    return {
        'entity': entity_type_stoi,
        'event': event_type_stoi,
        'relation': relation_type_stoi,
        'role': role_type_stoi,
        'mention': mention_type_stoi,
        # 'entity_label': entity_label_stoi,
        # 'trigger_label': trigger_label_stoi,
    }


def load_valid_patterns(path, vocabs):
    event_type_vocab = vocabs['event_type']
    entity_type_vocab = vocabs['entity_type']
    relation_type_vocab = vocabs['relation_type']
    role_type_vocab = vocabs['role_type']

    # valid event-role
    valid_event_role = set()
    event_role = json.load(
        open(os.path.join(path, 'event_role.json'), 'r', encoding='utf-8'))
    for event, roles in event_role.items():
        if event not in event_type_vocab:
            continue
        event_type_idx = event_type_vocab[event]
        for role in roles:
            if role not in role_type_vocab:
                continue
            role_type_idx = role_type_vocab[role]
            valid_event_role.add(event_type_idx * 100 + role_type_idx)

    # valid relation-entity
    valid_relation_entity = set()
    relation_entity = json.load(
        open(os.path.join(path, 'relation_entity.json'), 'r', encoding='utf-8'))
    for relation, entities in relation_entity.items():
        relation_type_idx = relation_type_vocab[relation]
        for entity in entities:
            entity_type_idx = entity_type_vocab[entity]
            valid_relation_entity.add(
                relation_type_idx * 100 + entity_type_idx)

    # valid role-entity
    valid_role_entity = set()
    role_entity = json.load(
        open(os.path.join(path, 'role_entity.json'), 'r', encoding='utf-8'))
    for role, entities in role_entity.items():
        if role not in role_type_vocab:
            continue
        role_type_idx = role_type_vocab[role]
        for entity in entities:
            entity_type_idx = entity_type_vocab[entity]
            valid_role_entity.add(role_type_idx * 100 + entity_type_idx)

    return {
        'event_role': valid_event_role,
        'relation_entity': valid_relation_entity,
        'role_entity': valid_role_entity
    }


def read_ltf(path):
    root = et.parse(path, et.XMLParser(
        dtd_validation=False, encoding='utf-8')).getroot()
    doc_id = root.find('DOC').get('id')
    doc_tokens = []
    for seg in root.find('DOC').find('TEXT').findall('SEG'):
        seg_id = seg.get('id')
        seg_tokens = []
        seg_start = int(seg.get('start_char'))
        seg_text = seg.find('ORIGINAL_TEXT').text
        for token in seg.findall('TOKEN'):
            token_text = token.text
            start_char = int(token.get('start_char'))
            end_char = int(token.get('end_char'))
            assert seg_text[start_char - seg_start:
                            end_char - seg_start + 1
                   ] == token_text, 'token offset error'
            seg_tokens.append((token_text, start_char, end_char))
        doc_tokens.append((seg_id, seg_tokens))

    return doc_tokens, doc_id


def read_txt(path, language='english'):
    doc_id = os.path.basename(path)
    data = open(path, 'r', encoding='utf-8').read()
    data = [s.strip() for s in data.split('\n') if s.strip()]
    sents = [l for ls in [sent_tokenize(line, language=language) for line in data]
             for l in ls]
    doc_tokens = []
    offset = 0
    for sent_idx, sent in enumerate(sents):
        sent_id = '{}-{}'.format(doc_id, sent_idx)
        tokens = word_tokenize(sent)
        tokens = [(token, offset + i, offset + i + 1)
                  for i, token in enumerate(tokens)]
        offset += len(tokens)
        doc_tokens.append((sent_id, tokens))
    return doc_tokens, doc_id


def read_json(path):
    with open(path, 'r', encoding='utf-8') as r:
        data = [json.loads(line) for line in r]
    doc_id = data[0]['doc_id']
    offset = 0
    doc_tokens = []

    for inst in data:
        tokens = inst['tokens']
        tokens = [(token, offset + i, offset + i + 1)
                  for i, token in enumerate(tokens)]
        offset += len(tokens)
        doc_tokens.append((inst['sent_id'], tokens))
    return doc_tokens, doc_id


def read_json_single(path):
    with open(path, 'r', encoding='utf-8') as r:
        data = [json.loads(line) for line in r]
    doc_id = os.path.basename(path)
    doc_tokens = []
    for inst in data:
        tokens = inst['tokens']
        tokens = [(token, i, i + 1) for i, token in enumerate(tokens)]
        doc_tokens.append((inst['sent_id'], tokens))
    return doc_tokens, doc_id


def save_result(output_file, gold_graphs, pred_graphs, sent_ids, tokens=None):
    with open(output_file, 'w', encoding='utf-8') as w:
        for i, (gold_graph, pred_graph, sent_id) in enumerate(
                zip(gold_graphs, pred_graphs, sent_ids)):
            output = {'sent_id': sent_id,
                      'gold': gold_graph.to_dict(),
                      'pred': pred_graph.to_dict()}
            if tokens:
                output['tokens'] = tokens[i]
            w.write(json.dumps(output) + '\n')


def mention_to_tab(start, end, entity_type, mention_type, mention_id, tokens, token_ids, score=1):
    tokens = tokens[start:end]
    token_ids = token_ids[start:end]
    span = '{}:{}-{}'.format(token_ids[0].split(':')[0],
                             token_ids[0].split(':')[1].split('-')[0],
                             token_ids[1].split(':')[1].split('-')[1])
    mention_text = tokens[0]
    previous_end = int(token_ids[0].split(':')[1].split('-')[1])
    for token, token_id in zip(tokens[1:], token_ids[1:]):
        start, end = token_id.split(':')[1].split('-')
        start, end = int(start), int(end)
        mention_text += ' ' * (start - previous_end) + token
        previous_end = end
    return '\t'.join([
        'json2tab',
        mention_id,
        mention_text,
        span,
        'NIL',
        entity_type,
        mention_type,
        str(score)
    ])


def json_to_mention_results(input_dir, output_dir, file_name,
                            bio_separator=' '):
    mention_type_list = ['nam', 'nom', 'pro', 'nam+nom+pro']
    file_type_list = ['bio', 'tab']
    writers = {}
    for mention_type in mention_type_list:
        for file_type in file_type_list:
            output_file = os.path.join(output_dir, '{}.{}.{}'.format(file_name,
                                                                     mention_type,
                                                                     file_type))
            writers['{}_{}'.format(mention_type, file_type)
            ] = open(output_file, 'w')

    json_files = glob.glob(os.path.join(input_dir, '*.json'))
    for f in json_files:
        with open(f, 'r', encoding='utf-8') as r:
            for line in r:
                result = json.loads(line)
                doc_id = result['doc_id']
                tokens = result['tokens']
                token_ids = result['token_ids']
                bio_tokens = [[t, tid, 'O']
                              for t, tid in zip(tokens, token_ids)]
                # separate bio output
                for mention_type in ['NAM', 'NOM', 'PRO']:
                    tokens_tmp = deepcopy(bio_tokens)
                    for start, end, enttype, mentype in result['graph']['entities']:
                        if mention_type == mentype:
                            tokens_tmp[start] = 'B-{}'.format(enttype)
                            for token_idx in range(start + 1, end):
                                tokens_tmp[token_idx] = 'I-{}'.format(
                                    enttype)
                    writer = writers['{}_bio'.format(mention_type.lower())]
                    for token in tokens_tmp:
                        writer.write(bio_separator.join(token) + '\n')
                    writer.write('\n')
                # combined bio output
                tokens_tmp = deepcopy(bio_tokens)
                for start, end, enttype, _ in result['graph']['entities']:
                    tokens_tmp[start] = 'B-{}'.format(enttype)
                    for token_idx in range(start + 1, end):
                        tokens_tmp[token_idx] = 'I-{}'.format(enttype)
                writer = writers['nam+nom+pro_bio']
                for token in tokens_tmp:
                    writer.write(bio_separator.join(token) + '\n')
                writer.write('\n')
                # separate tab output
                for mention_type in ['NAM', 'NOM', 'PRO']:
                    writer = writers['{}_tab'.format(mention_type.lower())]
                    mention_count = 0
                    for start, end, enttype, mentype in result['graph']['entities']:
                        if mention_type == mentype:
                            mention_id = '{}-{}'.format(doc_id, mention_count)
                            tab_line = mention_to_tab(
                                start, end, enttype, mentype, mention_id, tokens, token_ids)
                            writer.write(tab_line + '\n')
                # combined tab output
                writer = writers['nam+nom+pro_tab']
                mention_count = 0
                for start, end, enttype, mentype in result['graph']['entities']:
                    mention_id = '{}-{}'.format(doc_id, mention_count)
                    tab_line = mention_to_tab(
                        start, end, enttype, mentype, mention_id, tokens, token_ids)
                    writer.write(tab_line + '\n')
    for w in writers:
        w.close()


def normalize_score(scores):
    min_score, max_score = min(scores), max(scores)
    if min_score == max_score:
        return [0] * len(scores)
    return [(s - min_score) / (max_score - min_score) for s in scores]


def best_score_by_task(log_file, task, max_epoch=1000):
    with open(log_file, 'r', encoding='utf-8') as r:
        config = r.readline()

        best_scores = []
        best_dev_score = 0
        for line in r:
            record = json.loads(line)
            dev = record['dev']
            test = record['test']
            epoch = record['epoch']
            if epoch > max_epoch:
                break
            if dev[task]['f'] > best_dev_score:
                best_dev_score = dev[task]['f']
                best_scores = [dev, test, epoch]

        print('Epoch: {}'.format(best_scores[-1]))
        tasks = ['entity', 'mention', 'relation', 'trigger_id', 'trigger',
                 'role_id', 'role']
        for t in tasks:
            print('{}: dev: {:.2f}, test: {:.2f}'.format(t,
                                                         best_scores[0][t][
                                                             'f'] * 100.0,
                                                         best_scores[1][t][
                                                             'f'] * 100.0))


def enumerate_spans(seq_len: int,
                    max_span_len: int):
    """Enumerate possible spans given the length of the sequence and max span
    length.
    Args:
        seq_len (int): Sequence length.
        max_span_len (int): Max span length.

    Returns:
        Tuple[List, List, List]: #TODO
    """
    idxs, mask, offsets, lens, boundaries = [], [], [], [], []

    # max_span_len = min(seq_len, max_span_len)
    for start in range(seq_len):
        for span_len in range(1, max_span_len + 1):
            if start + span_len > seq_len:
                break
            pad_len = max_span_len - span_len
            end = start + span_len
            offsets.append((start, end))
            idxs.extend([i for i in range(start, end)] + [0] * pad_len)
            mask.extend([1.0 / span_len] * span_len + [0] * pad_len)
            boundaries.append(start)
            boundaries.append(end - 1)
            span_len_vec = [0 for i in range(max_span_len)]
            span_len_vec[span_len - 1] = 1
            lens.append(span_len_vec)

    return idxs, mask, offsets, lens, boundaries


def label2onehot(tensor: torch.Tensor, label_size: int):
    tensor_onehot = tensor.new_zeros(tensor.size(0), label_size)
    tensor_onehot.scatter_(1, tensor.unsqueeze(-1), 1)
    return tensor_onehot


from sklearn.cluster import DBSCAN, OPTICS, AgglomerativeClustering


def build_information_graph(batch,
                            # entities
                            is_start,
                            len_from_here,
                            type_pred,
                            cluster_labels,
                            # trigger
                            is_start_pred_ev,
                            len_from_here_pred_ev,
                            type_pred_ev,
                            cluster_labels_ev,
                            coref_embed_ev,
                            # relation
                            relation_any,
                            relation_cand,
                            relation_true_for_cand,
                            coref_embeds,
                            relation_pred,
                            attn_sum,
                            evidence_true_for_cand,
                            vocabs,
                            gold_inputs=False,
                            config=None,
                            extra=0,
                            rel_type_thr=None):
    entity_itos = {i: s for s, i in vocabs['entity'].items()}
    trigger_itos = {i: s for s, i in vocabs['event'].items()}
    relation_itos = {i: s for s, i in vocabs['relation'].items()}
    role_itos = {i: s for s, i in vocabs['role'].items()}

    symmetric_rel = config.get("symmetric_relations")

    graphs = []
    for graph_idx in range(batch.batch_size):

        # coref_preds = None
        # TODO remove from here
        coref_matrix = None
        if coref_embeds is not None:
            coref_embeds_cur = coref_embeds[graph_idx].cpu().numpy()

            # coref_matrix = np.linalg.norm(coref_embeds_cur[:, None, :] - coref_embeds_cur[None, :, :], axis=-1)

            ##clustering = AgglomerativeClustering(distance_threshold=1., n_clusters=None).fit(coref_embeds_cur)
            # clustering = OPTICS(min_samples=2).fit(coref_embeds_cur)
            # clustering = DBSCAN(min_samples=2).fit(coref_embeds_cur)

            # coref_preds = []
            coref_matrix = []

            for i in range(coref_embeds_cur.shape[0]):
                new_l = []
                for j in range(coref_embeds_cur.shape[0]):

                    if i == j:
                        new_l.append(1)
                    elif cluster_labels[graph_idx, i] == cluster_labels[graph_idx, j] and cluster_labels[
                        graph_idx, i] != -1:
                        # coref_preds.append(1)
                        new_l.append(1)
                    else:
                        # coref_preds.append(0)
                        new_l.append(0)
                coref_matrix.append(new_l)

        entities = []

        cur_ent = 0

        is_start_pred_cur = is_start[graph_idx].argmax(-1).tolist()
        len_from_here_pred_cur = len_from_here[graph_idx].view(is_start.shape[1], -1, 2).argmax(-1).tolist()

        cur_clusters = None

        if cluster_labels is not None:
            cur_clusters = cluster_labels[graph_idx]

        if gold_inputs:
            is_start_pred_cur = batch.is_start[graph_idx].tolist()
            len_from_here_pred_cur = batch.len_from_here[graph_idx].tolist()

            cur_clusters = batch.mention_to_ent_coref[graph_idx]

        for j in range(is_start.shape[1]):
            if is_start_pred_cur[j] == 1:
                for i in range(len(len_from_here_pred_cur[j])):
                    if len_from_here_pred_cur[j][i]:
                        start = j
                        end = j + i
                        # type = type_from_here[graph_idx, j].argmax().item()
                        type = type_pred[graph_idx, cur_ent].argmax().item()
                        cur_ent += 1
                        # if type != 0:
                        entities.append((start, end, entity_itos[type]))

        entity_num = cur_ent

        triggers = []

        if cluster_labels_ev is None:
            cur_cluster_labels_ev = None
        else:
            cur_cluster_labels_ev = cluster_labels_ev[graph_idx].tolist()

        if type_pred_ev is not None:

            cur_ent = 0

            is_start_pred_cur = is_start_pred_ev[graph_idx].argmax(-1).tolist()
            len_from_here_pred_cur = len_from_here_pred_ev[graph_idx]. \
                view(is_start_pred_ev.shape[1], -1, 2).argmax(-1).tolist()

            # cur_clusters = cluster_labels_ev[graph_idx]

            if gold_inputs:
                is_start_pred_cur = batch.is_start_ev[graph_idx].tolist()
                len_from_here_pred_cur = batch.len_from_here_ev[graph_idx].tolist()

                cur_cluster_labels_ev = batch.mention_to_ev_coref[graph_idx]

            for j in range(is_start.shape[1]):
                if is_start_pred_cur[j] == 1:
                    for i in range(len(len_from_here_pred_cur[j])):
                        if len_from_here_pred_cur[j][i]:
                            start = j
                            end = j + i
                            # type = type_from_here[graph_idx, j].argmax().item()
                            # why out of bounds???
                            type = type_pred_ev[graph_idx, cur_ent].argmax().item()
                            cur_ent += 1
                            # if type != 0:
                            triggers.append((start, end, trigger_itos[type]))

        relations = []
        evidence = []
        evidence_class = []
        evid_scores = []

        if relation_pred is not None:

            if rel_type_thr is None:
                rel_type_thr = torch.ones(len(vocabs['relation'])).cuda() * 0.5
            else:
                rel_type_thr = torch.Tensor(rel_type_thr).cuda()

            cluster_num = cur_clusters.max() + 1

            rel_cand = relation_cand[graph_idx].view(cluster_num, cluster_num)

            rel_matrix_nonzero = relation_any[graph_idx].view(cluster_num, cluster_num, -1)
            cur_idx = 0

            # print(rel_matrix_nonzero.argmax(-1).sum())

            nonzero_final_probs = []

            # if not(rel_matrix_nonzero.argmax(-1).sum() > 200 or rel_matrix_nonzero.argmax(-1).sum() == 0):

            # all_cand_rels = []

            for i in range(cluster_num):
                for j in range(cluster_num):

                    if i == j:
                        continue

                    if symmetric_rel and i > j:
                        continue

                    any_probs = rel_matrix_nonzero[i, j]

                    if any_probs[-1] > extra[1] and rel_cand[i, j]:

                        predicted_not_zero = False

                        if config.get("relation_type_level") != "multitype":

                            rel_pred = relation_pred[graph_idx, cur_idx].argmax().item()
                            if rel_pred > 0:
                                predicted_not_zero = True
                            rel_pred = relation_itos[rel_pred]

                        else:

                            rel_pred_scores = relation_pred[graph_idx, cur_idx]
                            rel_pred_types = (rel_pred_scores > rel_type_thr).float().nonzero().view(-1).tolist()
                            if len(rel_pred_types) > 0:
                                predicted_not_zero = True
                            rel_pred = [relation_itos[i] for i in rel_pred_types]

                        attn_cur = attn_sum[graph_idx, cur_idx].tolist()

                        if predicted_not_zero:
                            relations.append((i, j, rel_pred))
                            nonzero_final_probs.append(relation_pred[graph_idx, cur_idx])

                            cur_evid = []
                            cur_evid_class = dict()
                            for k in range(len(attn_cur[0])):
                                if k not in rel_pred_types:
                                    continue
                                k_s = relation_itos[k]
                                cur_evid_class[k_s] = []
                                for l in range(len(attn_cur)):
                                    if attn_cur[l][k] > extra[0]:
                                        cur_evid.append(l)
                                        cur_evid_class[k_s].append(l)
                            evidence.append(cur_evid)
                            evidence_class.append(cur_evid_class)
                            evid_scores.append(attn_cur)

                    if rel_cand[i, j]:
                        cur_idx += 1

                    # if rel_cand[i, j]:

        cur_graph = Graph(entities, triggers, relations, [], evidence, evidence_class,
                          coref_matrix, cur_clusters, cur_cluster_labels_ev)

        if relation_pred is not None:
            cur_graph.rel_probs = nonzero_final_probs
            cur_graph.evid_scores = evid_scores

        graphs.append(cur_graph)

    return graphs


def load_model(path, model, device=0, gpu=True):
    print('Loading the model from {}'.format(path))
    map_location = 'cuda:{}'.format(device) if gpu else 'cpu'
    state = torch.load(path, map_location=map_location)

    model.load_state_dict(state['model'])
    model.cuda(device)

    vocabs = state['vocabs']

    optimizer = state['optimizer']

    return model, vocabs, optimizer


def get_coref_clusters(coref_matrix):
    if not isinstance(coref_matrix, list):
        coref_matrix = coref_matrix.tolist()

    entity_num = len(coref_matrix)

    coref_entities = list(range(entity_num))

    for i in range(entity_num):
        for j in range(entity_num):

            if coref_matrix[i][j] == 1:
                coref_entities[j] = min(coref_entities[j], i)

    coref_dict = dict()

    coref_cluster_lists = []

    for i in range(entity_num):
        if coref_entities[i] not in coref_dict:
            coref_dict[coref_entities[i]] = len(coref_dict)
            coref_cluster_lists.append([])
        coref_entities[i] = coref_dict[coref_entities[i]]

        coref_cluster_lists[coref_entities[i]].append(i)

    return coref_entities, coref_cluster_lists


def clusters_from_cluster_labels(cluster_labels):
    num_clusters = max(cluster_labels, default=-1) + 1
    clusters = [[] for i in range(num_clusters)]

    for i, cl in enumerate(cluster_labels):
        clusters[cl].append(i)

    return clusters


def align_pred_to_gold(true_entities, pred_entities):
    rev_true = defaultdict(lambda: -1)

    for i, (s, e, t) in enumerate(true_entities):
        rev_true[(s, e)] = i

    alignment = []
    total_al_ent = len(true_entities)

    for i, (s, e, t) in enumerate(pred_entities):
        aligned_to = rev_true[(s, e)]
        if aligned_to == -1:
            aligned_to = total_al_ent
            total_al_ent += 1
        alignment.append(aligned_to)

    return alignment


import matplotlib.pyplot as plt
import numpy as np
from highlight_text import ax_text, fig_text
import matplotlib.colors as mcolors
from sklearn.manifold import TSNE


def summary_graph(pred_graph, true_graph, batch,
                  writer, global_step, prefix, vocabs, coref_embeds=None, id=None, create_images=False):
    if coref_embeds is not None:
        coref_embeds = coref_embeds[0]

    tokens = batch.tokens[0]

    offsets = batch.entity_offsets

    rev_pos_offsets = dict()

    for i in range(len(batch.pos_entity_offsets[0])):
        rev_pos_offsets[batch.pos_entity_offsets[0][i]] = i

    predicted_entities = []
    pred_to_true = []
    for (s, e, t) in pred_graph.entities:
        if (s, e) in rev_pos_offsets:
            pred_to_true.append(rev_pos_offsets[(s, e)])
        else:
            pred_to_true.append(-1)
        predicted_entities.append(("|".join(tokens[s:e]), t, s, e, pred_to_true[-1]))

    true_entities = []
    for i, (s, e, t) in enumerate(true_graph.entities):
        true_entities.append(("|".join(tokens[s:e]), t, s, e, i))

    writer.add_text(prefix + "predicted_entities", " ".join(str(predicted_entities)), global_step)
    writer.add_text(prefix + "true_entities", " ".join(str(true_entities)), global_step)
    writer.add_text(prefix + "full_text", " ".join(tokens).replace("</s>", "\n\n"), global_step)

    predicted_entities_set = set(predicted_entities)

    true_entities_set = set(true_entities)

    predicted_false = predicted_entities_set - true_entities_set

    not_predicted = true_entities_set - predicted_entities_set

    writer.add_text(prefix + "predicted_false", " ".join(str(predicted_false)), global_step)
    writer.add_text(prefix + "not_predicted", " ".join(str(not_predicted)), global_step)

    ###triggers

    rev_pos_offsets = dict()

    for i in range(len(batch.pos_trigger_offsets[0])):
        rev_pos_offsets[batch.pos_trigger_offsets[0][i]] = i

    predicted_triggers = []
    pred_to_true_ev = []
    for (s, e, t) in pred_graph.triggers:
        if (s, e) in rev_pos_offsets:
            pred_to_true_ev.append(rev_pos_offsets[(s, e)])
        else:
            pred_to_true_ev.append(-1)
        predicted_triggers.append(("|".join(tokens[s:e]), t, s, e, pred_to_true_ev[-1]))

    true_triggers = []
    for i, (s, e, t) in enumerate(true_graph.triggers):
        true_triggers.append(("|".join(tokens[s:e]), t, s, e, i))

    writer.add_text(prefix + "predicted_triggers", " ".join(str(predicted_triggers)), global_step)
    writer.add_text(prefix + "true_triggers", " ".join(str(true_triggers)), global_step)
    #writer.add_text(prefix + "full_text", "|".join(tokens), global_step)

    predicted_trig_set = set(predicted_triggers)

    true_trig_set = set(true_triggers)

    predicted_false = predicted_trig_set - true_trig_set

    not_predicted = true_trig_set - predicted_trig_set

    writer.add_text(prefix + "predicted_false_trig", " ".join(str(predicted_false)), global_step)
    writer.add_text(prefix + "not_predicted_trig", " ".join(str(not_predicted)), global_step)

    ###

    all_cols = list(mcolors.TABLEAU_COLORS) + list(mcolors.BASE_COLORS)

    true_ent_text = ""

    tok_dict_t = dict()
    tok_dict_idx = dict()

    for idx, (s, e, t) in enumerate(true_graph.entities):
        for i in range(s, e):
            tok_dict_t[i] = t
            tok_dict_idx[i] = idx

    entity_num = len(true_entities)

    coref_entities = [i for i in range(entity_num)]

    # cur_idx = 0

    # print(entity_num, entity_num ** 2)
    # print(len(batch.pos_entity_offsets[0]))
    # print(batch.coref_labels.shape)

    coref_entities, true_clusters = get_coref_clusters(batch.coref_labels[0])

    total_coref_ent = max(coref_entities, default=-1)

    true_coref_text = ""

    for i in range(total_coref_ent + 1):
        cur_corefd = 0
        coref_ents = []
        for j in range(entity_num):
            if coref_entities[j] == i:
                cur_corefd += 1
                s, e, _ = true_graph.entities[j]
                coref_ents.append(("|".join(tokens[s:e]), s, e))
        if cur_corefd > 1:
            true_coref_text += " ".join(str(coref_ents)) + "\n"

    writer.add_text(prefix + "true_coref_text", true_coref_text, global_step)

    col_list = []

    cur_len = 0

    for i, tok in enumerate(tokens[:500]):
        tok = tok.replace('$', 'dol')
        tok = tok.replace('>', 'gt')
        tok = tok.replace('<', 'lt')
        if len(tok) == 0:
            continue
        col = None
        if i in tok_dict_t and tok != '\n':  # and len(col_list) < 50:
            col = all_cols[(vocabs['entity'][tok_dict_t[i]] - 1) % len(all_cols)]
            col_list.append(col)
        if col:
            true_ent_text += '<'
        true_ent_text += tok
        cur_len += len(tok)
        if tok[-1] == '\n':
            cur_len = 0
        if col:
            true_ent_text += '>'
        true_ent_text += ' '
        if cur_len > 50:
            cur_len = 0
            true_ent_text += '\n'

    if create_images:
        fig, ax = plt.subplots()
        plt.tight_layout()
        plt.axis('off')
        # print(true_ent_text)
        ax_text(x=0, y=1.,
                s=true_ent_text,
                color='k', highlight_colors=col_list, va='top')
        # plt.show()
        fig.canvas.draw()
        # plt.show()
        img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        img1 = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        writer.add_image(prefix + "true_entities_type", img1, global_step, dataformats='HWC')

    ###

    true_ent_text = ""

    col_list = []

    cur_len = 0

    for i, tok in enumerate(tokens[:500]):
        tok = tok.replace('$', 'dol')
        tok = tok.replace('>', 'gt')
        tok = tok.replace('<', 'lt')
        if len(tok) == 0:
            continue
        col = None
        if i in tok_dict_idx and tok != '\n':  # and coref_entities[tok_dict_idx[i]] < len(all_cols):
            col = all_cols[coref_entities[tok_dict_idx[i]] % len(all_cols)]
            col_list.append(col)
        if col:
            true_ent_text += '<'
        true_ent_text += tok
        cur_len += len(tok)
        if tok[-1] == '\n':
            cur_len = 0
        if col:
            true_ent_text += '>'
        true_ent_text += ' '
        if cur_len > 50:
            cur_len = 0
            true_ent_text += '\n'

    if create_images:
        fig, ax = plt.subplots()
        plt.tight_layout()
        plt.axis('off')
        # print(true_ent_text)
        ax_text(x=0, y=1.,
                s=true_ent_text,
                color='k', highlight_colors=col_list, va='top')
        # plt.show()
        fig.canvas.draw()
        # plt.show()
        img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        img1 = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        writer.add_image(prefix + "true_entities_coref", img1, global_step, dataformats='HWC')

    ###

    pred_ent_text = ""

    tok_dict_t = dict()
    tok_dict_idx = dict()

    for idx, (s, e, t) in enumerate(pred_graph.entities):
        for i in range(s, e):
            tok_dict_t[i] = t
            tok_dict_idx[i] = idx

    entity_num = len(predicted_entities)

    coref_entities = [i for i in range(entity_num)]

    cur_idx = 0

    pred_coref_pairs = []
    notpred_coref_pairs = []

    if pred_graph.coref_matrix is not None:

        for i in range(entity_num):
            for j in range(i + 1, entity_num):

                true_ent_i = pred_to_true[i]
                true_ent_j = pred_to_true[j]

                is_actually_coref = False
                if true_ent_i != -1 and true_ent_j != -1:
                    is_actually_coref = true_graph.coref_matrix[true_ent_i][true_ent_j]

                if pred_graph.coref_matrix[i][j] == 1:
                    coref_entities[j] = min(coref_entities[j], i)

                    if coref_embeds is not None:
                        pred_coref_pairs.append((predicted_entities[i], predicted_entities[j],
                                                 torch.norm(coref_embeds[i] - coref_embeds[j]).item(),
                                                 is_actually_coref))
                else:
                    if coref_embeds is not None:
                        notpred_coref_pairs.append((predicted_entities[i], predicted_entities[j],
                                                    torch.norm(coref_embeds[i] - coref_embeds[j]).item(),
                                                    is_actually_coref))

                cur_idx += 1

        pred_coref_pairs_notactually = sorted(pred_coref_pairs, key=lambda x: x[2] + x[3] * 999999999)[:100]
        notpred_coref_pairs_actually = sorted(notpred_coref_pairs, key=lambda x: x[2] - x[3] * 999999999)[:100]

        writer.add_text(prefix + "pred_coref_pairs", "\n".join(map(str, pred_coref_pairs_notactually)), global_step)
        writer.add_text(prefix + "notpred_coref_pairs", "\n".join(map(str, notpred_coref_pairs_actually)), global_step)

        coref_dict = dict()
        pred_clusters = clusters_from_cluster_labels(pred_graph.cluster_labels)

        for i in range(entity_num):
            if coref_entities[i] not in coref_dict:
                coref_dict[coref_entities[i]] = len(coref_dict)
                # pred_clusters.append([])
            coref_entities[i] = coref_dict[coref_entities[i]]
            # pred_clusters[coref_entities[i]].append(i)

        pred_coref_ent = max(coref_entities, default=-1)

        pred_coref_text = ""

        for i in range(pred_coref_ent + 1):
            cur_corefd = 0
            coref_ents = []
            for j in range(entity_num):
                if coref_entities[j] == i:
                    cur_corefd += 1
                    s, e, _ = pred_graph.entities[j]
                    coref_ents.append(("|".join(tokens[s:e]), s, e))
            if cur_corefd > 1:
                pred_coref_text += " ".join(str(coref_ents)) + "\n"

        writer.add_text(prefix + "pred_coref_text", pred_coref_text, global_step)

        if coref_embeds is not None:
            if coref_embeds.shape[-1] != 2 and coref_embeds.shape[0] > 1:
                coref_embeds = coref_embeds.cpu().numpy()
                coref_embeds = TSNE(n_components=2).fit_transform(coref_embeds)

            if coref_embeds.shape[0] == 1:
                coref_embeds = np.array([[0, 0]])

            y = coref_embeds[:, 0].tolist()
            z = coref_embeds[:, 1].tolist()

            n = [ent[0] for ent in predicted_entities]
            c = [all_cols[coref_entities[i] % len(all_cols)] for i in range(entity_num)]

            if create_images:
                fig, ax = plt.subplots()
                ax.scatter(z, y, c=c)

                for i, txt in enumerate(n):
                    ax.annotate(txt, (z[i], y[i]))

                fig.canvas.draw()
                img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
                img1 = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                writer.add_image(prefix + "pred_coref_embeds", img1, global_step, dataformats='HWC')

    col_list = []

    cur_len = 0

    for i, tok in enumerate(tokens[:500]):
        tok = tok.replace('$', 'dol')
        tok = tok.replace('>', 'gt')
        tok = tok.replace('<', 'lt')
        if len(tok) == 0:
            continue
        col = None
        if i in tok_dict_t and tok != '\n':  # and len(col_list) < 50:
            col = all_cols[(vocabs['entity'][tok_dict_t[i]] - 1) % len(all_cols)]
            col_list.append(col)
        if col:
            pred_ent_text += '<'
        pred_ent_text += tok
        cur_len += len(tok)
        if tok[-1] == '\n':
            cur_len = 0
        if col:
            pred_ent_text += '>'
        pred_ent_text += ' '
        if cur_len > 50:
            cur_len = 0
            pred_ent_text += '\n'

    if create_images:
        fig, ax = plt.subplots()
        plt.tight_layout()
        plt.axis('off')
        # print(pred_ent_text)
        ax_text(x=0, y=1.,
                s=pred_ent_text,
                color='k', highlight_colors=col_list, va='top')
        # plt.show()
        fig.canvas.draw()
        # plt.show()
        img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        # img0 = img.reshape((3,) + fig.canvas.get_width_height()[::-1])
        img1 = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        # writer.add_image(prefix+"true_entities_text", img0, global_step)
        writer.add_image(prefix + "pred_ent_type", img1, global_step, dataformats='HWC')

    ###

    if pred_graph.coref_matrix is not None and create_images:
        true_ent_text = ""

        col_list = []

        cur_len = 0

        for i, tok in enumerate(tokens[:500]):
            tok = tok.replace('$', 'dol')
            tok = tok.replace('>', 'gt')
            tok = tok.replace('<', 'lt')
            if len(tok) == 0:
                continue
            col = None
            if i in tok_dict_idx and tok != '\n':  # and coref_entities[tok_dict_idx[i]] < len(all_cols):
                col = all_cols[coref_entities[tok_dict_idx[i]] % len(all_cols)]
                col_list.append(col)
            if col:
                true_ent_text += '<'
            true_ent_text += tok
            cur_len += len(tok)
            if tok[-1] == '\n':
                cur_len = 0
            if col:
                true_ent_text += '>'
            true_ent_text += ' '
            if cur_len > 50:
                cur_len = 0
                true_ent_text += '\n'

        fig, ax = plt.subplots()
        plt.tight_layout()
        plt.axis('off')
        # print(true_ent_text)
        ax_text(x=0, y=1.,
                s=true_ent_text,
                color='k', highlight_colors=col_list, va='top')
        # plt.show()
        fig.canvas.draw()
        # plt.show()
        img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        img1 = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        writer.add_image(prefix + "pred_entities_coref", img1, global_step, dataformats='HWC')

    ##############################

    rel_pair_dict = defaultdict(lambda : defaultdict(lambda : set()))

    predicted_relations = []

    ###want:
    #correctly predicted with comparison of evidence
    #not predicted
    #predicted false

    for i, (a, b, t) in enumerate(pred_graph.relations[:50]):
        if len(pred_clusters[a]) == 0:
            arg1 = "None"
        else:
            arg1 = predicted_entities[pred_clusters[a][0]][0]
        if len(pred_clusters[b]) == 0:
            arg2 = "None"
        else:
            arg2 = predicted_entities[pred_clusters[b][0]][0]
        if arg1 > arg2:
            arg1, arg2 = arg2, arg1
        prob_pred = F.log_softmax(pred_graph.rel_probs[i], dim=-1).tolist()
        prob_pred = max(prob_pred)
        prob_pred = round(prob_pred, 2)
        """evid = []
        if len(pred_graph.evidence) > i:
            evid = pred_graph.evidence[i]
            evid_scores = pred_graph.evid_scores[i]
            for j in range(len(evid_scores)):
                evid_scores[j] = round(evid_scores[j], 2)"""
        predicted_relations.append((arg1, arg2, t, prob_pred))

        rel_pair_dict[(arg1, arg2)]['pred_types'] = set(t)
        rel_pair_dict[(arg1, arg2)]['pred_evid'] = dict()
        for type in t:
            rel_pair_dict[(arg1, arg2)]['pred_evid'][type] = pred_graph.evidence_class[i][type]

    true_relations = []

    for i, (a, b, t) in enumerate(true_graph.relations):
        arg1 = true_entities[true_clusters[a][0]][0]
        arg2 = true_entities[true_clusters[b][0]][0]
        if arg1 > arg2:
            arg1, arg2 = arg2, arg1
        evid = []
        if len(true_graph.evidence) > i:
            evid = true_graph.evidence[i]
        true_relations.append((arg1, arg2, t, evid))


        rel_pair_dict[(arg1, arg2)]['true_types'] = set(t)
        rel_pair_dict[(arg1, arg2)]['true_evid']  = dict()
        for type in t:
            rel_pair_dict[(arg1, arg2)]['true_evid'][type] = true_graph.evidence_class[i][type]

    writer.add_text(prefix + "predicted_relations", " ".join(str(predicted_relations)), global_step)
    writer.add_text(prefix + "true_relations", " ".join(str(true_relations)), global_step)

    rel_analysis_text = ""
    for (a, b), d in rel_pair_dict.items():
        rel_analysis_text += a + " - " + b + "\n\n"
        rel_analysis_text += "Correctly predicted types:\n\n"
        for type in d['pred_types'].intersection(d['true_types']):
            rel_analysis_text += type + "\n\n"
            rel_analysis_text += "  Predicted evidence: " + str(d['pred_evid'][type]) + "\n\n"
            rel_analysis_text += "  True evidence: " + str(d['true_evid'][type]) + "\n\n"
        rel_analysis_text += "Incorrect types:\n\n"
        for type in d['pred_types'] - d['true_types']:
            rel_analysis_text += type + "\n\n"
            rel_analysis_text += "  Predicted evidence: " + str(d['pred_evid'][type]) + "\n\n"
        rel_analysis_text += "Not predicted types:\n\n"
        for type in d['true_types'] - d['pred_types']:
            rel_analysis_text += type + "\n\n"
            rel_analysis_text += "  True evidence: " + str(d['true_evid'][type]) + "\n\n"
        rel_analysis_text += "\n\n"

    writer.add_text(prefix + "rel_analysis", rel_analysis_text, global_step)


    """predicted_relations = set(predicted_relations)

    true_relations = set(true_relations)

    predicted_false = predicted_relations - true_relations

    not_predicted = true_relations - predicted_relations

    writer.add_text(prefix + "rels_predicted_false", " ".join(str(predicted_false)), global_step)
    writer.add_text(prefix + "rels_not_predicted", " ".join(str(not_predicted)), global_step)"""

    draw_network(true_entities, true_clusters, true_graph.relations,
                 writer, prefix + "true", global_step, id)

    draw_network(predicted_entities, pred_clusters, pred_graph.relations,
                 writer, prefix + "pred", global_step, id)

    plt.close('all')


def load_word_embed(path: str,
                    dimension: int,
                    *,
                    skip_first: bool = False,
                    freeze: bool = False,
                    sep: str = ' '
                    ):
    """Load pre-trained word embeddings from file.

    Args:
        path (str): Path to the word embedding file.
        skip_first (bool, optional): Skip the first line. Defaults to False.
        freeze (bool, optional): Freeze embedding weights. Defaults to False.

    Returns:
        Tuple[nn.Embedding, Dict[str, int]]: The first element is an Embedding
        object. The second element is a word vocab, where the keys are words and
        values are word indices.
    """
    vocab = {'$$$UNK$$$': 0}
    embed_matrix = [[0.0] * dimension]
    with open(path) as r:
        if skip_first:
            r.readline()
        for line in r:
            segments = line.rstrip('\n').rstrip(' ').split(sep)
            word = segments[0]
            vocab[word] = len(vocab)
            embed = [float(x) for x in segments[1:]]
            embed_matrix.append(embed)
    print('Loaded %d word embeddings' % (len(embed_matrix) - 1))

    embed_matrix = torch.FloatTensor(embed_matrix)

    word_embed = torch.nn.Embedding.from_pretrained(embed_matrix,
                                                    freeze=freeze,
                                                    padding_idx=0)
    return word_embed, vocab


def elem_max(t1, t2):
    combined = torch.cat((t1.unsqueeze(-1), t2.unsqueeze(-1)), dim=-1)
    return torch.max(combined, dim=-1)[0]  # .squeeze(-1)


def elem_min(t1, t2):
    combined = torch.cat((t1.unsqueeze(-1), t2.unsqueeze(-1)), dim=-1)
    return torch.min(combined, dim=-1)[0]  # .squeeze(-1)


def get_pairwise_idxs_separate(num1: int, num2: int, skip_diagonal: bool = False):
    idxs1, idxs2 = [], []
    for i in range(num1):
        for j in range(num2):
            if i == j and skip_diagonal:
                continue
            idxs1.append(i)
            idxs2.append(j)
    return idxs1, idxs2


def augment(tokens, mask_prob, ws_tokenizer, ws_model):
    masked = set()

    tokens_masked = []

    for i in range(len(tokens)):
        r = random()
        if r < mask_prob:
            tokens_masked.append("[MASK]")
            masked.add(i + 1)
        else:
            tokens_masked.append(str.lower(tokens[i]))

    # print(tokens)

    # print("After masking:")
    # print(tokens_masked)

    # print(sorted(list(masked)))

    # inputs = tokenizer(text, return_tensors="pt")

    tokens_orig = tokens

    tokens = ws_tokenizer.encode(tokens_masked, return_tensors="pt").cuda()

    # print("Encoded text:")
    ##print("|".join([tokenizer.decode(tok) for tok in inputs['input_ids'].view(-1).tolist()]))
    # print("|".join([ws_tokenizer.decode(tok) for tok in tokens.view(-1).tolist()]))

    with torch.no_grad():
        outputs = ws_model(tokens)

    logits = outputs.logits

    logits = logits.squeeze(0)

    """max_str = outputs.logits.argmax(dim=-1).view(-1).tolist()

    for j in range(len(max_str)):
        if j not in masked:
            max_str[j] = tokens[0, j]

    #print("Best prediction:")

    #print("|".join([ws_tokenizer.decode(tok) for tok in max_str]))
    """
    samples = torch.multinomial((logits * 1.5).softmax(dim=-1), num_samples=1, replacement=True).T

    ##print("Samples:")

    # for i in range(samples.shape[0]):
    ##print(samples[i])
    i = 0
    sample_list = samples[i].tolist()
    for j in range(len(sample_list)):
        if j not in masked:
            sample_list[j] = tokens[0, j]
    ##print("|".join([ws_tokenizer.decode(tok) for tok in sample_list]))
    ##print()

    res = []

    for j in range(1, len(sample_list) - 1):
        if j in masked:
            res.append(ws_tokenizer.decode(sample_list[j]).replace(" ", ""))
        else:
            res.append(tokens_orig[j - 1].replace(" ", ""))

    # print(res)
    # print()

    return res


class RegLayer(nn.Module):

    def __init__(self, hid_dim, s=0.1, d=0.5):
        super(RegLayer, self).__init__()

        self.drop = nn.Dropout(d)
        self.norm = nn.LayerNorm(hid_dim)
        # self.norm = nn.InstanceNorm1d(hid_dim, affine=True, track_running_stats=True)

        self.s = s

    def forward(self, x):
        if self.training:
            r = torch.randn(x.shape).cuda() * self.s
            x = x * (r + 1.)
            del r

        # x = self.drop(x)

        # print(x.shape)
        # x = x.transpose(-2, -1)
        # print(x.shape)
        x = self.norm(x)
        # x = x.transpose(-2, -1)

        return x


import networkx as nx


def draw_network(entities, clusters, relations, writer=None, prefix="ex", global_step=None, id=0, create_images=False):
    ents = []

    for cl_id, cl in enumerate(clusters):
        cur_ent = []
        for i in cl:
            cur_ent.append(entities[i][0].replace("|", " "))
        ents.append((",".join(list(set(cur_ent))), {"col": cl_id, "size": len(cur_ent), "type": entities[cl[0]][1]}))

    rels = []

    for a, b, t in relations:
        rels.append((ents[a][0], ents[b][0], {"label": str(t)}))

    G = nx.Graph()

    G.add_nodes_from(ents)

    G.add_edges_from(rels)

    if create_images:
        fig, ax = plt.subplots()
        plt.tight_layout()
        plt.axis('off')

        nx.draw(G, node_size=50)

        fig.canvas.draw()

        img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        img1 = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        writer.add_image(prefix + "_graph", img1, global_step, dataformats='HWC')

    nx.write_gexf(G, "output/" + prefix + "_" + str(id) + ".gexf")


def get_facts(graphs, titles, rev_dict):
    facts = []

    for (title, graph) in zip(titles, graphs):
        for i, (h, t, r_type) in enumerate(graph.relations):
            # r_sep = r_text.split("|")
            r_rev = [rev_dict[r_i] for r_i in r_type]
            for r in r_rev:
                facts.append({
                    "title": title,
                    "h_idx": h,
                    "t_idx": t,
                    "r": r,
                    "evidence": graph.evidence[i]
                })

    return facts


def get_rev_dict(rel_info_path):
    rel_info = open(rel_info_path, 'r', encoding='utf-8')

    rel_info = json.loads(rel_info.readline())

    rev_dict = dict()

    for k, v in rel_info.items():
        rev_dict[v] = k

    return rev_dict

def get_adjustment(prec, rec):

    diff = abs(prec - rec)

    if diff < 0.03:
        thr_delta = 0
    elif diff < 0.06:
        thr_delta = 0.01
    elif diff < 0.1:
        thr_delta = 0.05
    elif diff < 0.3:
        thr_delta = 0.1
    else:
        thr_delta = 0.2

    if prec > rec:
        thr_delta *= -1

    if prec < 0.01:
        thr_delta = 0.05

    return thr_delta

def adjust_thresholds(thr, stats, vocabs, ep=0):
    # new_thr = [i for i in thr]

    for type, metrics in stats.items():
        idx = vocabs[type]
        prec, rec = metrics["prec"], metrics["rec"]

        thr_delta = get_adjustment(prec, rec)

        thr[idx] += thr_delta

        thr[idx] = max(thr[idx], 0.01)
        thr[idx] = min(thr[idx], 0.99)
