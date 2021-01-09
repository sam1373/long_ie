"""Our scorer is adapted from: https://github.com/dwadden/dygiepp"""

from util import get_coref_clusters, align_pred_to_gold
import math
from statistics import harmonic_mean


def safe_div(num, denom):
    if denom > 0:
        return num / denom
    else:
        return 0


def compute_f1(predicted, gold, matched):
    precision = safe_div(matched, predicted)
    recall = safe_div(matched, gold)
    f1 = harmonic_mean((precision, recall))
    return precision, recall, f1


# b_cubed modified to account for system mentions
# see https://www.aclweb.org/anthology/W10-4305.pdf

import copy


def b_cubed_modified(key, response, true_entity_num, max_pred_ent, not_predicted_set):
    response_mod = []

    for r in response:
        if len(r) == 1 and r[0] >= true_entity_num:
            continue
        response_mod.append(set(r))

    for ent in not_predicted_set:
        response_mod.append({ent})

    key = [set(k) for k in key]
    key_precision = copy.copy(key)
    for ent in range(true_entity_num, max_pred_ent + 1):
        key_precision.append({ent})

    if sum(len(r) for r in response_mod) == 0:
        P = 0.0
    else:
        P = math.fsum(
            len(r.intersection(k)) ** 2 / len(r) for r in response_mod for k in key_precision
        ) / sum(len(r) for r in response_mod)

    response_recall = []
    for r in response_mod:
        new_r = set([ent for ent in r if ent < true_entity_num])
        response_recall.append(new_r)

    if sum(len(k) for k in key) == 0:
        R = 0.0
    else:
        R = math.fsum(
            len(k.intersection(r)) ** 2 / len(k) for k in key for r in response_recall
        ) / sum(len(k) for k in key)

    F = harmonic_mean((R, P))

    return R, P, F


def convert_arguments(triggers, entities, roles):
    args = set()
    for trigger_idx, entity_idx, role in roles:
        arg_start, arg_end, _ = entities[entity_idx]
        trigger_label = triggers[trigger_idx][-1]
        args.add((arg_start, arg_end, trigger_label, role))
    return args


def span_match(span_1, span_2, check_type=True):
    if check_type and span_1[-1] != span_2[-1]:
        return 0
    sa, ea, _ = span_1
    sb, eb, _ = span_2
    iou = (min(ea, eb) - max(sa, sb)) / (max(eb, ea) - min(sa, sb))
    return iou


def overlap_score(cluster_1, cluster_2, predicted_entities, gold_entities):
    matched = 0
    for s1 in cluster_1:
        matched += 1 if any([span_match(predicted_entities[s1], gold_entities[s2]) > 0.5 for s2 in cluster_2]) else 0

    return min(matched / len(cluster_1), matched / len(cluster_2))


# this way matching is 1-to-1 as clusters match when more than half is present in the other


def compute_cluster_metrics(predicted_clusters, gold_clusters, predicted_entities, gold_entities):
    matched_predicted = []
    matched_gold = []
    for i, p in enumerate(predicted_clusters):
        for j, g in enumerate(gold_clusters):
            if overlap_score(p, g, predicted_entities, gold_entities) > 0.5:
                matched_predicted.append(i)
                matched_gold.append(j)

    matched_predicted = set(matched_predicted)
    matched_gold = set(matched_gold)

    # metrics = {
    #    "p": len(matched_predicted) / (len(predicted_clusters) + 1e-7),
    #    "r": len(matched_gold) / (len(gold_clusters) + 1e-7),
    # }
    # metrics["f1"] = 2 * metrics["p"] * metrics["r"] / (metrics["p"] + metrics["r"] + 1e-7)

    return len(matched_gold)


def score_graphs(gold_graphs, pred_graphs,
                 relation_directional=False):
    gold_arg_num = pred_arg_num = arg_idn_num = arg_class_num = 0
    gold_trigger_num = pred_trigger_num = trigger_idn_num = trigger_class_num = 0
    gold_ent_num = pred_ent_num = ent_match_num = ent_overlap_match_num = 0
    gold_rel_num = pred_rel_num = rel_match_num = 0
    gold_men_num = pred_men_num = men_match_num = 0
    # gold_cluster_matched = gold_cluster_total = pred_cluster_total = 0
    cluster_p = cluster_r = 0

    for gold_graph, pred_graph in zip(gold_graphs, pred_graphs):
        # Entity
        gold_entities = gold_graph.entities
        pred_entities = pred_graph.entities
        gold_ent_num += len(gold_entities)
        pred_ent_num += len(pred_entities)
        ent_match_num += len([entity for entity in pred_entities
                              if entity in gold_entities])
        ent_overlap_match_num += len([entity for entity in pred_entities
                                      if any([span_match(entity, s2) > 0.5 for s2 in gold_entities])])

        alignment = align_pred_to_gold(gold_entities, pred_entities)

        not_predicted_idx = set(range(len(gold_entities))) - set(alignment)

        max_pred_ent = max(alignment)

        """# Mention
        gold_mentions = gold_graph.mentions
        pred_mentions = pred_graph.mentions
        gold_men_num += len(gold_mentions)
        pred_men_num += len(pred_mentions)
        men_match_num += len([mention for mention in pred_mentions
                              if mention in gold_mentions])"""

        pred_entity_coref, pred_clusters = get_coref_clusters(pred_graph.coref_matrix)
        gold_entity_cored, gold_clusters = get_coref_clusters(gold_graph.coref_matrix)

        pred_clusters_ment_aligned = [list(map(lambda x: alignment[x], c)) for c in pred_clusters]

        pred_clusters_aligned = []

        num_matched = 0

        for p_cl in pred_clusters_ment_aligned:
            found = False
            for g_id, g_cl in enumerate(gold_clusters):
                matched = set(p_cl).intersection(set(g_cl))
                if len(matched) > len(p_cl) // 2 and len(matched) > len(g_cl) // 2:
                    pred_clusters_aligned.append(g_id)
                    found = True
                    num_matched += 1
                    break
            if not found:
                pred_clusters_aligned.append(-1)

        matched_prec = num_matched / len(pred_clusters)
        matched_rec = num_matched / len(gold_clusters)

        # g_c_m = compute_cluster_metrics(pred_clusters, gold_clusters, pred_entities, gold_entities)
        r, p, f = b_cubed_modified(gold_clusters, pred_clusters_ment_aligned, len(gold_entities), max_pred_ent,
                                   not_predicted_idx)

        # print("pred clusters:", pred_clusters)
        # print("gold clusters:", gold_clusters)
        # print("matched:", g_c_m)

        # gold_cluster_matched += g_c_m
        # pred_cluster_matched += p_c_m

        cluster_p += p
        cluster_r += r

        # gold_cluster_total += len(gold_clusters)
        # pred_cluster_total += len(pred_clusters)

        # Relation
        gold_relations = gold_graph.relations
        pred_relations = pred_graph.relations
        gold_rel_num += len(gold_relations)
        pred_rel_num += len(pred_relations)
        for arg1, arg2, rel_type in pred_relations:
            # arg1_start, arg1_end, _ = pred_entities[arg1]
            # arg2_start, arg2_end, _ = pred_entities[arg2]
            arg1 = pred_clusters_aligned[arg1]
            arg2 = pred_clusters_aligned[arg2]
            if arg1 == -1 or arg2 == -1:
                continue
            for arg1_gold, arg2_gold, rel_type_gold in gold_relations:
                # arg1_start_gold, arg1_end_gold, _ = gold_entities[arg1_gold]
                # arg2_start_gold, arg2_end_gold, _ = gold_entities[arg2_gold]
                if relation_directional:
                    if (arg1 == arg1_gold and arg2 == arg2_gold
                    ) and rel_type == rel_type_gold:
                        rel_match_num += 1
                        break
                else:
                    if ((arg1 == arg1_gold and arg2 == arg2_gold) or (
                            arg1 == arg2_gold and arg1 == arg2_gold
                    )) and rel_type == rel_type_gold:
                        rel_match_num += 1
                        break

        # Trigger
        gold_triggers = gold_graph.triggers
        pred_triggers = pred_graph.triggers
        gold_trigger_num += len(gold_triggers)
        pred_trigger_num += len(pred_triggers)
        for trg_start, trg_end, event_type in pred_triggers:
            matched = [item for item in gold_triggers
                       if item[0] == trg_start and item[1] == trg_end]
            if matched:
                trigger_idn_num += 1
                if matched[0][-1] == event_type:
                    trigger_class_num += 1

        # Argument
        gold_args = convert_arguments(gold_triggers, gold_entities,
                                      gold_graph.roles)
        pred_args = convert_arguments(pred_triggers, pred_entities,
                                      pred_graph.roles)
        gold_arg_num += len(gold_args)
        pred_arg_num += len(pred_args)
        for pred_arg in pred_args:
            arg_start, arg_end, event_type, role = pred_arg
            gold_idn = {item for item in gold_args
                        if item[0] == arg_start and item[1] == arg_end
                        and item[2] == event_type}
            if gold_idn:
                arg_idn_num += 1
                gold_class = {item for item in gold_idn if item[-1] == role}
                if gold_class:
                    arg_class_num += 1

    entity_prec, entity_rec, entity_f = compute_f1(
        pred_ent_num, gold_ent_num, ent_match_num)
    entity_overlap_prec, entity_overlap_rec, entity_overlap_f = compute_f1(
        pred_ent_num, gold_ent_num, ent_overlap_match_num)
    # mention_prec, mention_rec, mention_f = compute_f1(
    #    pred_men_num, gold_men_num, men_match_num)
    trigger_id_prec, trigger_id_rec, trigger_id_f = compute_f1(
        pred_trigger_num, gold_trigger_num, trigger_idn_num)
    trigger_prec, trigger_rec, trigger_f = compute_f1(
        pred_trigger_num, gold_trigger_num, trigger_class_num)
    relation_prec, relation_rec, relation_f = compute_f1(
        pred_rel_num, gold_rel_num, rel_match_num)
    role_id_prec, role_id_rec, role_id_f = compute_f1(
        pred_arg_num, gold_arg_num, arg_idn_num)
    role_prec, role_rec, role_f = compute_f1(
        pred_arg_num, gold_arg_num, arg_class_num)
    # cluster_prec, cluster_rec, cluster_f = compute_f1(
    #    pred_cluster_total, gold_cluster_total, gold_cluster_matched
    # )
    cluster_prec = cluster_p / len(gold_graphs)
    cluster_rec = cluster_r / len(gold_graphs)
    cluster_f = harmonic_mean((cluster_prec, cluster_rec))

    print('Entity: P: {:.2f}, R: {:.2f}, F: {:.2f}'.format(
        entity_prec * 100.0, entity_rec * 100.0, entity_f * 100.0))
    print('Entity Overlap 0.5: P: {:.2f}, R: {:.2f}, F: {:.2f}'.format(
        entity_overlap_prec * 100.0, entity_overlap_rec * 100.0, entity_overlap_f * 100.0))
    # print('Mention: P: {:.2f}, R: {:.2f}, F: {:.2f}'.format(
    #    mention_prec * 100.0, mention_rec * 100.0, mention_f * 100.0))
    print('Trigger identification: P: {:.2f}, R: {:.2f}, F: {:.2f}'.format(
        trigger_id_prec * 100.0, trigger_id_rec * 100.0, trigger_id_f * 100.0))
    print('Trigger: P: {:.2f}, R: {:.2f}, F: {:.2f}'.format(
        trigger_prec * 100.0, trigger_rec * 100.0, trigger_f * 100.0))
    print('Relation: P: {:.2f}, R: {:.2f}, F: {:.2f}'.format(
        relation_prec * 100.0, relation_rec * 100.0, relation_f * 100.0))
    print('Role identification: P: {:.2f}, R: {:.2f}, F: {:.2f}'.format(
        role_id_prec * 100.0, role_id_rec * 100.0, role_id_f * 100.0))
    print('Role: P: {:.2f}, R: {:.2f}, F: {:.2f}'.format(
        role_prec * 100.0, role_rec * 100.0, role_f * 100.0))
    print('Entity Clusters: P: {:.2f}, R: {:.2f}, F: {:.2f}'.format(
        cluster_prec * 100.0, cluster_rec * 100.0, cluster_f * 100.0))

    scores = {
        'entity': {'prec': entity_prec, 'rec': entity_rec, 'f': entity_f},
        'entity_overlap': {'prec': entity_overlap_prec, 'rec': entity_overlap_rec, 'f': entity_overlap_f},
        # 'mention': {'prec': mention_prec, 'rec': mention_rec, 'f': mention_f},
        'trigger': {'prec': trigger_prec, 'rec': trigger_rec, 'f': trigger_f},
        'trigger_id': {'prec': trigger_id_prec, 'rec': trigger_id_rec,
                       'f': trigger_id_f},
        'role': {'prec': role_prec, 'rec': role_rec, 'f': role_f},
        'role_id': {'prec': role_id_prec, 'rec': role_id_rec, 'f': role_id_f},
        'relation': {'prec': relation_prec, 'rec': relation_rec,
                     'f': relation_f},
        'entity_clusters': {'prec': cluster_prec, 'rec': cluster_rec, 'f': cluster_f}
    }
    return scores


def score_coref(gold_graphs, pred_graphs):
    pass
