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
    prefix = ['B', 'I']
    entity_label_stoi = {'O': 0}
    trigger_label_stoi = {'O': 0}
    for t in entity_type_set:
        for p in prefix:
            entity_label_stoi['{}-{}'.format(p, t)] = len(entity_label_stoi)
    for t in event_type_set:
        for p in prefix:
            trigger_label_stoi['{}-{}'.format(p, t)] = len(trigger_label_stoi)

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
        'entity_label': entity_label_stoi,
        'trigger_label': trigger_label_stoi,
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
    for span_len in range(1, max_span_len + 1):
        pad_len = max_span_len - span_len
        # rel_span_len = span_len / max_span_len
        for start in range(seq_len - span_len + 1):
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


def build_information_graph(batch,
                            all_scores,
                            local_scores,
                            entity_idxs,
                            entity_nums,
                            trigger_idxs,
                            trigger_nums,
                            vocabs):
    entity_itos = {i: s for s, i in vocabs['entity'].items()}
    trigger_itos = {i: s for s, i in vocabs['event'].items()}
    relation_itos = {i: s for s, i in vocabs['relation'].items()}
    role_itos = {i: s for s, i in vocabs['role'].items()}

    # relation_size = (len(relation_itos) - 1) // 2 + 1

    graphs = []
    for graph_idx in range(batch.batch_size):
        entity_num = entity_nums[graph_idx]
        trigger_num = trigger_nums[graph_idx]
        entity_idx_map = {}
        trigger_idx_map = {}

        inst_scores = all_scores[graph_idx]
        inst_entity_idxs = entity_idxs[graph_idx][:entity_num]
        inst_trigger_idxs = trigger_idxs[graph_idx][:trigger_num]
        inst_entity_offsets = [batch.entity_offsets[i] for i in inst_entity_idxs]
        inst_trigger_offsets = [batch.trigger_offsets[i] for i in inst_trigger_idxs]

        # Predict entities
        entities = []
        entity_scores = inst_scores[-1].get('entity', None)
        if entity_scores is not None:
            entity_scores = entity_scores[:entity_num]
            entity_type_idxs = entity_scores.max(1)[1].tolist()

            for idx, entity_type_idx in enumerate(entity_type_idxs):
                if entity_type_idx > 0:
                    entity_type = entity_itos[entity_type_idx]
                    start, end = inst_entity_offsets[idx]
                    entities.append((start, end, entity_type))
                    entity_idx_map[idx] = len(entity_idx_map)

        # Predict triggers
        triggers = []
        trigger_scores = inst_scores[-1].get('trigger', None)
        if trigger_scores is not None:
            trigger_scores = trigger_scores[:trigger_num]
            trigger_type_idxs = trigger_scores.max(1)[1].tolist()

            for idx, trigger_type_idx in enumerate(trigger_type_idxs):
                if trigger_type_idx > 0:
                    trigger_type = trigger_itos[trigger_type_idx]
                    start, end = inst_trigger_offsets[idx]
                    triggers.append((start, end, trigger_type))
                    trigger_idx_map[idx] = len(trigger_idx_map)

        # Predict relation
        relations = []
        relation_scores = inst_scores[-1].get('relation', None)
        if relation_scores is not None:
            for entity_i in range(entity_num):
                for entity_j in range(entity_i + 1, entity_num):

                    if entity_i not in entity_idx_map or entity_j not in entity_idx_map:
                        # Skip entity i or entity j is NULL
                        continue

                    relation_score_1 = relation_scores[entity_i * (entity_num - 1) + entity_j - 1].tolist()
                    relation_score_2 = relation_scores[entity_j * (entity_num - 1) + entity_i].tolist()
                    relation_score = [max(i, j) for i, j in zip(relation_score_1, relation_score_2)]

                    max_idx, max_value = argmax(relation_score)
                    if max_idx != 0:
                        entity_i_idx = entity_idx_map[entity_i]
                        entity_j_idx = entity_idx_map[entity_j]
                        relation_type = relation_itos[max_idx]

                        relations.append((entity_i_idx,
                                          entity_j_idx,
                                          relation_type))

        # Predict role
        roles = []
        role_scores = inst_scores[-1].get('role', None)
        if role_scores is not None:
            for trigger_idx in range(trigger_num):
                for entity_idx in range(entity_num):

                    if trigger_idx not in trigger_idx_map or entity_idx not in entity_idx_map:
                        # Skip if the trigger or entity is NULL
                        continue

                    role_score = role_scores[trigger_idx * entity_num + entity_idx].tolist()
                    max_idx, max_value = argmax(role_score)
                    if max_idx != 0:
                        trigger_idx_new = trigger_idx_map[trigger_idx]
                        entity_idx_new = entity_idx_map[entity_idx]
                        role_type = role_itos[max_idx]

                        # if ontology.is_valid_role(triggers[trigger_idx_new][-1],
                        #                          role_type):
                        roles.append((trigger_idx_new,
                                      entity_idx_new,
                                      role_type))

        graphs.append(Graph(entities, triggers, relations, roles))
    return graphs


def build_local_information_graph(batch,
                                  gnn_scores,
                                  all_scores,
                                  entity_idxs,
                                  entity_nums,
                                  trigger_idxs,
                                  trigger_nums,
                                  candidate_span_scores,
                                  vocabs):
    entity_itos = {i: s for s, i in vocabs['entity'].items()}
    trigger_itos = {i: s for s, i in vocabs['event'].items()}
    relation_itos = {i: s for s, i in vocabs['relation'].items()}
    role_itos = {i: s for s, i in vocabs['role'].items()}

    candidates = []
    candidate_scores = []

    graphs = []
    for graph_idx in range(batch.batch_size):
        entity_num = entity_nums[graph_idx]
        trigger_num = trigger_nums[graph_idx]
        entity_idx_map = {}
        trigger_idx_map = {}

        inst_entity_idxs = entity_idxs[graph_idx]  # [:entity_num]
        inst_trigger_idxs = trigger_idxs[graph_idx]  # [:trigger_num]
        inst_entity_offsets = [batch.entity_offsets[i] for i in inst_entity_idxs]
        inst_trigger_offsets = [batch.trigger_offsets[i] for i in inst_trigger_idxs]

        # Predict candidate spans
        candidates = []
        # candidate_span_scores = all_scores['candidate']  # [graph_idx]
        # candidate_span_scores = candidate_span_scores[:entity_num]
        # candidate_is_predicted = candidate_span_scores[graph_idx].max(1)[1].tolist()
        candidate_scores = candidate_span_scores[graph_idx].softmax(1)[:, 1].tolist()
        for idx in range(len(candidate_scores)):
            if candidate_scores[idx] > 0.5:
                start, end = batch.entity_offsets[idx]
                candidates.append((start, end))

        # Predict entities
        entities = []
        entity_scores = all_scores['entity']  # [graph_idx]
        if entity_scores.size(0) > 0 and entity_num:
            entity_scores = entity_scores[:entity_num]
            entity_type_idxs = entity_scores.max(1)[1].tolist()
            for idx, entity_type_idx in enumerate(entity_type_idxs):
                if entity_type_idx > 0:
                    entity_type = entity_itos[entity_type_idx]
                    start, end = inst_entity_offsets[idx]
                    entities.append((start, end, entity_type))
                    entity_idx_map[idx] = len(entity_idx_map)

        # Predict triggers
        triggers = []
        trigger_scores = all_scores['trigger']  # [graph_idx]
        if trigger_scores.size(0) > 0 and trigger_num:
            trigger_scores = trigger_scores[:trigger_num]
            trigger_type_idxs = trigger_scores.max(1)[1].tolist()

            for idx, trigger_type_idx in enumerate(trigger_type_idxs):
                if trigger_type_idx > 0:
                    trigger_type = trigger_itos[trigger_type_idx]
                    start, end = inst_trigger_offsets[idx]
                    triggers.append((start, end, trigger_type))
                    trigger_idx_map[idx] = len(trigger_idx_map)

        # Predict relations
        relations = []
        if entity_num > 1:
            relation_scores = all_scores['relation']  # [:entity_num, :entity_num - 1]
            if relation_scores is not None and relation_scores.size(0) > 0:
                relation_scores = relation_scores.view(entity_num, entity_num - 1, -1)
                for entity_i in range(entity_num):
                    for entity_j in range(entity_i + 1, entity_num):
                        if entity_i not in entity_idx_map or entity_j not in entity_idx_map:
                            # Skip entity i or entity j is NULL
                            continue

                        # relation_score_1 = relation_scores[entity_i * (entity_num - 1) + entity_j - 1].tolist()
                        # relation_score_2 = relation_scores[entity_j * (entity_num - 1) + entity_i].tolist()
                        relation_score_1 = relation_scores[entity_i, entity_j - 1].tolist()
                        relation_score_2 = relation_scores[entity_j, entity_i].tolist()
                        relation_score = [max(i, j) for i, j in zip(relation_score_1, relation_score_2)]

                        max_idx, max_value = argmax(relation_score)
                        if max_idx != 0:
                            entity_i_idx = entity_idx_map[entity_i]
                            entity_j_idx = entity_idx_map[entity_j]
                            relation_type = relation_itos[max_idx]

                            relations.append((entity_i_idx,
                                              entity_j_idx,
                                              relation_type))

        # Predict role
        roles = []
        if trigger_num and entity_num:
            role_scores = all_scores['role']  # [graph_idx]
            if role_scores is not None and role_scores.size(0) > 0:
                role_scores = role_scores.view(trigger_num, entity_num, -1)
                for trigger_idx in range(trigger_num):
                    for entity_idx in range(entity_num):

                        if trigger_idx not in trigger_idx_map or entity_idx not in entity_idx_map:
                            # Skip if the trigger or entity is NULL
                            continue

                        # role_score = role_scores[trigger_idx * entity_num + entity_idx].sigmoid().tolist()
                        role_score = role_scores[trigger_idx, entity_idx].tolist()
                        max_idx, max_value = argmax(role_score)
                        if max_idx != 0 and max_value > 0:
                            trigger_idx_new = trigger_idx_map[trigger_idx]
                            entity_idx_new = entity_idx_map[entity_idx]
                            role_type = role_itos[max_idx]

                            # if ontology.is_valid_role(triggers[trigger_idx_new][-1],
                            #                        role_type):
                            roles.append((trigger_idx_new,
                                          entity_idx_new,
                                          role_type))

        graphs.append(Graph(entities, triggers, relations, roles))
    return graphs, candidates, candidate_scores


from sklearn.cluster import DBSCAN, OPTICS


def build_information_graph(batch,
                            is_start,
                            len_from_here,
                            type_from_here,
                            vocabs):
    entity_itos = {i: s for s, i in vocabs['entity'].items()}
    trigger_itos = {i: s for s, i in vocabs['event'].items()}
    relation_itos = {i: s for s, i in vocabs['relation'].items()}
    role_itos = {i: s for s, i in vocabs['role'].items()}


    graphs = []
    for graph_idx in range(batch.batch_size):

        entities = []

        for j in range(is_start.shape[1]):
            if is_start[graph_idx, j].argmax().item() == 1:
                start = j
                end = j + len_from_here[graph_idx, j].argmax().item()
                type = type_from_here[graph_idx, j].argmax().item()
                if type != 0:
                    entities.append((start, end, entity_itos[type]))

        graphs.append(Graph(entities, [], [], []))

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


import matplotlib.pyplot as plt
import numpy as np
from highlight_text import ax_text, fig_text
import matplotlib.colors as mcolors


def summary_graph(pred_graph, true_graph, batch,
                  writer, global_step, prefix, vocabs):
    tokens = batch.tokens[0]

    offsets = batch.entity_offsets

    rev_offsets = dict()

    for i in range(len(offsets)):
        rev_offsets[offsets[i]] = i

    predicted_entities = []
    for (s, e, t) in pred_graph.entities:
        predicted_entities.append(("|".join(tokens[s:e]), t, s, e))

    true_entities = []
    for (s, e, t) in true_graph.entities:
        true_entities.append(("|".join(tokens[s:e]), t, s, e))

    writer.add_text(prefix + "predicted_entities", " ".join(str(predicted_entities)), global_step)
    writer.add_text(prefix + "true_entities", " ".join(str(true_entities)), global_step)
    writer.add_text(prefix + "full_text", "|".join(tokens), global_step)

    predicted_entities_set = set(predicted_entities)

    true_entities_set = set(true_entities)

    predicted_false = predicted_entities_set - true_entities_set

    not_predicted = true_entities_set - predicted_entities_set

    writer.add_text(prefix + "predicted_false", " ".join(str(predicted_false)), global_step)
    writer.add_text(prefix + "not_predicted", " ".join(str(not_predicted)), global_step)

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

    cur_idx = 0

    print(entity_num, entity_num * (entity_num - 1), batch.coref_labels.shape)

    for i in range(entity_num):
        for j in range(entity_num):

            if i == j:
                continue

            if i > j:
                cur_idx += 1
                continue

            if cur_idx >= batch.coref_labels.shape[0]:
                break

            if batch.coref_labels[cur_idx] == 1:
                coref_entities[j] = min(coref_entities[j], i)

            cur_idx += 1

    coref_dict = dict()

    for i in range(entity_num):
        if coref_entities[i] not in coref_dict:
            coref_dict[coref_entities[i]] = len(coref_dict)
        coref_entities[i] = coref_dict[coref_entities[i]]

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
            for j in range(entity_num):

                if i == j:
                    continue

                if i > j:
                    cur_idx += 1
                    continue

                if cur_idx >= batch.coref_labels.shape[0]:
                    break

                if pred_graph.coref_matrix[cur_idx] == 1:
                    coref_entities[j] = min(coref_entities[j], i)

                    pred_coref_pairs.append((predicted_entities[i], predicted_entities[j], torch.norm(coref_embeds[i] - coref_embeds[j]).item()))
                else:
                    notpred_coref_pairs.append((predicted_entities[i], predicted_entities[j],
                                             torch.norm(coref_embeds[i] - coref_embeds[j]).item()))

                cur_idx += 1

        pred_coref_pairs = sorted(pred_coref_pairs, key=lambda x: x[2])[:100]
        notpred_coref_pairs = sorted(notpred_coref_pairs, key=lambda x: x[2])[:100]

        writer.add_text(prefix + "pred_coref_pairs", "\n".join(map(str, pred_coref_pairs)), global_step)
        writer.add_text(prefix + "notpred_coref_pairs", "\n".join(map(str, notpred_coref_pairs)), global_step)

        coref_dict = dict()

        for i in range(entity_num):
            if coref_entities[i] not in coref_dict:
                coref_dict[coref_entities[i]] = len(coref_dict)
            coref_entities[i] = coref_dict[coref_entities[i]]

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

    if pred_graph.coref_matrix is not None:
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

    predicted_relations = []

    for (a, b, t) in pred_graph.relations:
        arg1 = predicted_entities[a][0]
        arg2 = predicted_entities[b][0]
        if arg1 > arg2:
            arg1, arg2 = arg2, arg1
        dist = abs(predicted_entities[a][2] - predicted_entities[b][2])
        predicted_relations.append((arg1, arg2, t, dist))

    true_relations = []

    for (a, b, t) in true_graph.relations:
        arg1 = true_entities[a][0]
        arg2 = true_entities[b][0]
        if arg1 > arg2:
            arg1, arg2 = arg2, arg1
        dist = abs(true_entities[a][2] - true_entities[b][2])
        true_relations.append((arg1, arg2, t, dist))

    writer.add_text(prefix + "predicted_relations", " ".join(str(predicted_relations)), global_step)
    writer.add_text(prefix + "true_relations", " ".join(str(true_relations)), global_step)

    predicted_relations = set(predicted_relations)

    true_relations = set(true_relations)

    predicted_false = predicted_relations - true_relations

    not_predicted = true_relations - predicted_relations

    writer.add_text(prefix + "rels_predicted_false", " ".join(str(predicted_false)), global_step)
    writer.add_text(prefix + "rels_not_predicted", " ".join(str(not_predicted)), global_step)

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
    return torch.max(combined, dim=-1)[0].squeeze(-1)


def get_pairwise_idxs_separate(num1: int, num2: int, skip_diagnol: bool = False):
    idxs1, idxs2 = [], []
    for i in range(num1):
        for j in range(num2):
            if i == j and skip_diagnol:
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

    tokens = ws_tokenizer.encode(tokens_masked, return_tensors="pt")

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
    samples = torch.multinomial((logits * 1.2).softmax(dim=-1), num_samples=1, replacement=True).T

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

    def __init__(self, hid_dim, s=0.1, d=0.3):
        super(RegLayer, self).__init__()

        # self.drop = nn.Dropout(d)
        # self.norm = nn.LayerNorm(hid_dim)
        self.norm = nn.InstanceNorm1d(hid_dim, affine=True, track_running_stats=True)

        self.s = s

    def forward(self, x):
        """if self.training:
            r = torch.randn(x.shape).cuda() * self.s
            x = x * (r + 1.)
            del r"""

        # x = self.drop(x)

        # print(x.shape)
        x = x.transpose(-2, -1)
        # print(x.shape)
        x = self.norm(x)
        x = x.transpose(-2, -1)

        return x
