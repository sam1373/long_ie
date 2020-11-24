import os
import json
import time
import logging

from tqdm import tqdm

import torch
import numpy as np
import random
from torch.utils.data import DataLoader
from transformers import (RobertaConfig,
                          RobertaTokenizer,
                          RobertaModel,
                          BertTokenizer,
                          BertModel,
                          AdamW,
                          get_linear_schedule_with_warmup)

from data import IEDataset
from model import OneIEpp, Linears, PairLinears
from gnn2 import GraphConv
from util import generate_vocabs, save_result
from config import config
from util import build_information_graph, pack_code, build_local_information_graph
from metric import score_graphs, best_performance
from ontology import Ontology

from argparse import ArgumentParser

# Set random seed
random.seed(61802)
np.random.seed(61802)
torch.manual_seed(61802)
torch.cuda.manual_seed(61802)
torch.cuda.manual_seed_all(61802)

parser = ArgumentParser()
parser.add_argument('-d', '--device', type=int, default=0)
parser.add_argument('--local', action='store_true')
args = parser.parse_args()

torch.cuda.set_device(args.device)

use_gnn = False#not args.local

timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())

logging.basicConfig(level=logging.INFO)
logging.getLogger("transformers").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

output_path = '/shared/nas/data/m1/yinglin8/projects/oneiepp/model/{}'.format(
    timestamp)
os.mkdir(output_path)
log_file = os.path.join(output_path, 'log.txt')
log_writer = open(log_file, 'w')
print('Output path', output_path)

train_file = '/shared/nas/data/m1/yinglin8/projects/oneiepp/data/ere/train.json'
dev_file = '/shared/nas/data/m1/yinglin8/projects/oneiepp/data/ere/dev.json'
test_file = '/shared/nas/data/m1/yinglin8/projects/oneiepp/data/ere/test.json'
print('Loading data')
train_set = IEDataset(train_file, config)
dev_set = IEDataset(dev_file, config)
test_set = IEDataset(test_file, config)

# bert_model = 'bert-large-cased'
bert_model = 'roberta-large'
bert_dir = '/shared/nas/data/m1/yinglin8/embedding/bert'
if bert_model.startswith('bert'):
    tokenizer = BertTokenizer.from_pretrained(bert_model,
                                              cache_dir=bert_dir)
else:
    tokenizer = RobertaTokenizer.from_pretrained(bert_model,
                                                 cache_dir=bert_dir)

print('Processing data')
train_set.process(tokenizer)
dev_set.process(tokenizer)
test_set.process(tokenizer)
print('Generating vocabs')
vocabs = generate_vocabs(train_set, dev_set, test_set,
                         add_rev_relation_type=False)

# load ontology
ontology = Ontology.from_file('ere.ontology.json', vocabs, ignore_arg_order=True)


print('Tensorizing')
train_set.tensorize(vocabs, config, ontology)
dev_set.tensorize(vocabs, config, ontology)
test_set.tensorize(vocabs, config, ontology)

# print('Train set')
# for tensor in train_set.tensors:
#     ontology.check_graph(tensor)
# print('Dev set')
# for tensor in dev_set.tensors:
#     ontology.check_graph(tensor)
# print('Test set')
# for tensor in test_set.tensors:
#     ontology.check_graph(tensor)

# vocabs
entity_label_size = len(vocabs['entity'])
event_label_size = trigger_label_size = len(vocabs['event'])
mention_label_size = len(vocabs['mention'])
relation_label_size = len(vocabs['relation'])
role_label_size = len(vocabs['role'])
print(vocabs)

print('Initialize model')
if bert_model.startswith('bert'):
    bert = BertModel.from_pretrained(bert_model,
                                     cache_dir=bert_dir)
else:
    bert = RobertaModel.from_pretrained(bert_model,
                                        cache_dir=bert_dir)
bert_dim = bert.config.hidden_size
node_dim = bert_dim * 3 + 1
edge_dim = node_dim * 2
dimensions = {
    'entity_repr': node_dim,
    'trigger_repr': node_dim,
    'entity_label': entity_label_size,
    'mention_label': mention_label_size,
    'trigger_label': trigger_label_size,
    'relation_label': relation_label_size,
    'role_label': role_label_size
}
entity_classifier = Linears([node_dim, 150, entity_label_size],
                            dropout_prob=.4)
trigger_classifier = Linears([node_dim, 600, event_label_size],
                             dropout_prob=.4)
mention_classifier = Linears([node_dim, 100, mention_label_size],
                             dropout_prob=.4)
relation_classifier = PairLinears(node_dim, node_dim, 200, relation_label_size,
                                  dropout_prob=.4)
role_classifier = PairLinears(node_dim, node_dim, 600, role_label_size,
                              dropout_prob=.4)

# entity_classifier_gnn = Linears([node_dim, 150, entity_label_size],
#                             dropout_prob=.4)
# trigger_classifier_gnn = Linears([node_dim, 600, event_label_size],
#                              dropout_prob=.4)
relation_classifier_gnn = PairLinears(node_dim + entity_label_size,
                                      node_dim + entity_label_size,
                                      200,
                                      relation_label_size,
                                      dropout_prob=.4)
role_classifier_gnn = PairLinears(node_dim + trigger_label_size,
                                  node_dim + entity_label_size,
                                  600,
                                  role_label_size,
                                  dropout_prob=.4)

# conv_list = [
#     ('relation', node_dim + entity_label_size, node_dim +
#      entity_label_size, relation_label_size, node_dim),
#     ('role', node_dim + event_label_size, node_dim +
#      entity_label_size, role_label_size, node_dim),
#     ('role_rev', node_dim + entity_label_size, node_dim +
#      event_label_size, role_label_size, node_dim),
#     ('entity_self', node_dim + entity_label_size, node_dim + entity_label_size,
#      1, node_dim),
#     ('trigger_self', node_dim + event_label_size, node_dim + event_label_size,
#      1, node_dim)
# ]
# feat_dims = {
#     'entity': (node_dim, entity_label_size),
#     'trigger': (node_dim, event_label_size),
# }
if use_gnn:
    gnn = GraphConv(2,
                    dimensions,
                    relation_classifier=relation_classifier_gnn,
                    role_classifier=role_classifier_gnn,
                    feat_drop=.2,
                    attn_drop=.1,
                    residual=False)
else:
    gnn = None
model = OneIEpp(config,
                vocabs,
                bert,
                entity_classifier=entity_classifier,
                mention_classifier=mention_classifier,
                event_classifier=trigger_classifier,
                relation_classifier=relation_classifier,
                role_classifier=role_classifier,
                gnn=gnn)
model.cuda()

# Training parameters
batch_size = config.get('batch_size', 10)
batch_num = len(train_set) // batch_size
epoch_num = config['epoch_num']

# Save code to the output dir
pack_code(os.path.dirname(os.path.abspath(__file__)),
          os.path.join(output_path, 'code.tar.gz'))

param_groups = [
    {
        'params': [p for n, p in model.named_parameters() if n.startswith('encoder')],
        'lr': 1e-5, 'weight_decay': 1e-5
    },
    {
        'params': [p for n, p in model.named_parameters() if not n.startswith('encoder')],
        'lr': 1e-4, 'weight_decay': 1e-3
    },
]
optimizer = AdamW(params=param_groups)
schedule = get_linear_schedule_with_warmup(optimizer,
                                           num_warmup_steps=batch_num * 5,
                                           num_training_steps=batch_num * epoch_num)

losses = []
best_dev_score = best_test_score = 0
for epoch in range(epoch_num):
    print('******* Epoch {} *******'.format(epoch))
    dataloader = DataLoader(train_set,
                            batch_size,
                            shuffle=True,
                            drop_last=True,
                            collate_fn=train_set.collate_fn)

    for batch_idx, batch in enumerate(tqdm(dataloader, ncols=75)):
        loss = model(batch)
        if loss is not None:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            schedule.step()
            optimizer.zero_grad()
            # losses.append(loss.item())
            # if len(losses) == 100:
            #     print('Loss {:.4f}'.format(sum(losses) / len(losses)))
            #     losses = []

    # Dev
    dev_loader = DataLoader(dev_set,
                            batch_size,
                            shuffle=False,
                            collate_fn=dev_set.collate_fn)
    gold_dev_graphs, pred_dev_graphs = [], []
    dev_sent_ids, dev_tokens = [], []
    for batch in tqdm(dev_loader, ncols=75):
        result = model.predict(batch)
        if use_gnn:
            pred_graphs = build_information_graph(batch, *result, vocabs, ontology)
        else:
            pred_graphs = build_local_information_graph(batch, *result, vocabs, ontology)
        pred_dev_graphs.extend(pred_graphs)
        gold_dev_graphs.extend(batch.graphs)
        dev_sent_ids.extend(batch.sent_ids)
        dev_tokens.extend(batch.tokens)
    # gold_dev_graphs = [g.clean(False)
    #                    for g in gold_dev_graphs]
    # pred_dev_graphs = [g.clean(False)
    #                    for g in pred_dev_graphs]

    print('Dev')
    dev_scores = score_graphs(gold_dev_graphs, pred_dev_graphs, False)
    save_result(os.path.join(output_path, 'dev.result.{}.json'.format(epoch)),
                gold_dev_graphs,
                pred_dev_graphs,
                dev_sent_ids,
                tokens=dev_tokens)

    # Test
    test_loader = DataLoader(test_set,
                             batch_size,
                             shuffle=False,
                             collate_fn=test_set.collate_fn)
    gold_test_graphs, pred_test_graphs = [], []
    test_sent_ids, test_tokens = [], []
    for batch in tqdm(test_loader, ncols=75):
        result = model.predict(batch)
        if use_gnn:
            pred_graphs = build_information_graph(batch, *result, vocabs, ontology)
        else:
            pred_graphs = build_local_information_graph(batch, *result, vocabs, ontology)
        pred_test_graphs.extend(pred_graphs)
        gold_test_graphs.extend(batch.graphs)
        test_sent_ids.extend(batch.sent_ids)
        test_tokens.extend(batch.tokens)
    # gold_test_graphs = [g.clean(False)
    #                     for g in gold_test_graphs]
    # pred_test_graphs = [g.clean(False)
    #                     for g in pred_test_graphs]

    print('Test')
    test_scores = score_graphs(gold_test_graphs, pred_test_graphs, False)
    save_result(os.path.join(output_path, 'test.result.{}.json'.format(epoch)),
                gold_test_graphs,
                pred_test_graphs,
                test_sent_ids,
                tokens=test_tokens)

    log_writer.write(json.dumps({'epoch': epoch,
                                 'dev': dev_scores,
                                 'test': test_scores}) + '\n')
    print('Output path:', output_path)

log_writer.close()
best_performance(os.path.join(output_path, 'log.txt'))