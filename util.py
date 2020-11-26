import os
import json
import glob
import lxml.etree as et
from nltk import word_tokenize, sent_tokenize
from copy import deepcopy
import torch
from graph import Graph

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

    #max_span_len = min(seq_len, max_span_len)
    for span_len in range(1, max_span_len + 1):
        pad_len = max_span_len - span_len
        #rel_span_len = span_len / max_span_len
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

                        #if ontology.is_valid_role(triggers[trigger_idx_new][-1],
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

        inst_entity_idxs = entity_idxs[graph_idx]#[:entity_num]
        inst_trigger_idxs = trigger_idxs[graph_idx]#[:trigger_num]
        inst_entity_offsets = [batch.entity_offsets[i] for i in inst_entity_idxs]
        inst_trigger_offsets = [batch.trigger_offsets[i] for i in inst_trigger_idxs]

        # Predict candidate spans
        candidates = []
        #candidate_span_scores = all_scores['candidate']  # [graph_idx]
        #candidate_span_scores = candidate_span_scores[:entity_num]
        #candidate_is_predicted = candidate_span_scores[graph_idx].max(1)[1].tolist()
        candidate_scores = candidate_span_scores[graph_idx].softmax(1)[:, 1].tolist()
        for idx in range(len(candidate_scores)):
            if candidate_scores[idx] > 0.5:
                start, end = batch.entity_offsets[idx]
                candidates.append((start, end))

        # Predict entities
        entities = []
        entity_scores = all_scores['entity']#[graph_idx]
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
        trigger_scores = all_scores['trigger']#[graph_idx]
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
            relation_scores = all_scores['relation']#[:entity_num, :entity_num - 1]
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
            role_scores = all_scores['role']#[graph_idx]
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

                            #if ontology.is_valid_role(triggers[trigger_idx_new][-1],
                            #                        role_type):
                            roles.append((trigger_idx_new,
                                        entity_idx_new,
                                        role_type))

        graphs.append(Graph(entities, triggers, relations, roles))
    return graphs, candidates, candidate_scores


def build_information_graph(batch,
                            span_candidate_score,
                            span_candidates_idxs,
                            entity_type,
                            trigger_type,
                            vocabs):

    return


def load_model(path, model, device=0, gpu=True):
    print('Loading the model from {}'.format(path))
    map_location = 'cuda:{}'.format(device) if gpu else 'cpu'
    state = torch.load(path, map_location=map_location)

    model.load_state_dict(state['model'])
    model.cuda(device)

    vocabs = state['vocabs']

    optimizer = state['optimizer']

    return model, vocabs, optimizer


def summary_entities(pred_graph, true_graph, batch, candidates, candidate_scores,
                     writer, global_step, prefix):

    tokens = batch.tokens[0]

    offsets = batch.entity_offsets


    rev_offsets = dict()

    for i in range(len(offsets)):
        rev_offsets[offsets[i]] = i

    span_candidates = []
    for (s, e) in candidates:
        span_candidates.append(("|".join(tokens[s:e]), s, e, round(candidate_scores[rev_offsets[(s, e)]], 2)))

    true_entities = []
    for (s, e, t) in true_graph.entities:
        score = -1
        if (s, e) in rev_offsets:
            score = round(candidate_scores[rev_offsets[(s, e)]], 2)
        true_entities.append(("|".join(tokens[s:e]), s, e, score))

    writer.add_text(prefix + "span_candidates", " ".join(str(span_candidates)), global_step)
    writer.add_text(prefix + "true_spans", " ".join(str(true_entities)), global_step)
    #writer.add_text(prefix + "full_text", "|".join(tokens), global_step)

    span_candidates = set(span_candidates)

    true_entities = set(true_entities)

    predicted_false = span_candidates - true_entities

    not_predicted = true_entities - span_candidates

    writer.add_text(prefix + "candidates_false", " ".join(str(predicted_false)), global_step)
    writer.add_text(prefix + "candidates_not_predicted", " ".join(str(not_predicted)), global_step)

    ################################

    predicted_entities = []
    for (s, e, t) in pred_graph.entities:
        predicted_entities.append(("|".join(tokens[s:e]), t, s, e))

    true_entities = []
    for (s, e, t) in true_graph.entities:
        true_entities.append(("|".join(tokens[s:e]), t, s, e))

    writer.add_text(prefix+"predicted_entities", " ".join(str(predicted_entities)), global_step)
    writer.add_text(prefix+"true_entities", " ".join(str(true_entities)), global_step)
    writer.add_text(prefix+"full_text", "|".join(tokens), global_step)

    predicted_entities = set(predicted_entities)

    true_entities = set(true_entities)

    predicted_false = predicted_entities - true_entities

    not_predicted = true_entities - predicted_entities

    writer.add_text(prefix+"predicted_false", " ".join(str(predicted_false)), global_step)
    writer.add_text(prefix+"not_predicted", " ".join(str(not_predicted)), global_step)


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