import os
import json
import time
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader
from transformers import (LongformerModel, LongformerTokenizer,
                          RobertaTokenizer, BertConfig, AdamW,
                          ElectraTokenizer, ElectraForMaskedLM,
                          XLNetModel, XLNetTokenizer,
                          get_linear_schedule_with_warmup)
from model import OneIEpp, Linears, PairLinears
from graph import Graph
from config import Config
# from data import IEDataset
from data2 import IEDataset
from scorer import score_graphs
from util import generate_vocabs, load_valid_patterns, save_result, best_score_by_task, \
    build_information_graph, build_local_information_graph, label2onehot, load_model, \
    summary_graph, load_word_embed

from torch.utils.tensorboard import SummaryWriter

#from roberta import RobertaModelPlus

from tqdm import tqdm

#import torch_optimizer as optim

import gc





skip_train = False

# configuration
parser = ArgumentParser()
parser.add_argument('-c', '--config', default='config/example.json')
parser.add_argument('-d', '--debug', default=False)
args = parser.parse_args()
config = Config.from_json_file(args.config)
# print(config.to_dict())

max_sent_len = config.get('max_sent_len', 2000)

use_gnn = False
batch_size = config.get('batch_size')
output_path = "output"
epoch_num = config.get('max_epoch')
# bert_model = "/home/samuel/Projects/exampleStudy/longTrans/chp/roberta-base-long"
bert_dir = config.get("bert_cache_dir")

extra_bert = config.get("extra_bert")
if not config.get("use_extra_bert"):
    extra_bert = 0

use_extra_word_embed = config.get("use_extra_word_embed")

# set GPU device
use_gpu = config.use_gpu
if use_gpu and config.gpu_device >= 0:
    torch.cuda.set_device(config.gpu_device)

# output
timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
output_dir = os.path.join(config.log_path, timestamp)
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
log_file = os.path.join(output_dir, 'log.txt')
with open(log_file, 'w', encoding='utf-8') as w:
    w.write(json.dumps(config.to_dict()) + '\n')
    print('Log file: {}'.format(log_file))
best_role_model = os.path.join(output_dir, 'best.role.mdl')
dev_result_file = os.path.join(output_dir, 'result.dev.json')
test_result_file = os.path.join(output_dir, 'result.test.json')



word_embed_dim = 0
word_vocab = None
word_embed = None
if use_extra_word_embed:

    embed_file = '../lstm_fet/enwiki.skip.size200.win10.neg15.sample1e-5.min15.txt'
    word_embed_dim = 200

    print('Loading word embeddings from %s' % embed_file)
    word_embed, word_vocab = load_word_embed(embed_file,
                                             word_embed_dim,
                                             skip_first=True)

# datasets
model_name = config.bert_model_name

tokenizer = RobertaTokenizer.from_pretrained("roberta-base",
                                             cache_dir=config.bert_cache_dir,
                                             do_lower_case=False)


#tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")

wordswap_tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-generator')
wordswap_model = ElectraForMaskedLM.from_pretrained('google/electra-small-generator', return_dict=True)


if args.debug:
    train_set = IEDataset(config.file_dir + config.dev_file, config, word_vocab, wordswap_tokenizer, wordswap_model)
    dev_set = IEDataset(config.file_dir + config.dev_file, config, word_vocab)
    test_set = IEDataset(config.file_dir + config.dev_file, config, word_vocab)
else:
    train_set = IEDataset(config.file_dir + config.train_file, config, word_vocab, wordswap_tokenizer, wordswap_model)
    dev_set = IEDataset(config.file_dir + config.dev_file, config, word_vocab, wordswap_tokenizer, wordswap_model)
    test_set = IEDataset(config.file_dir + config.test_file, config, word_vocab, wordswap_tokenizer, wordswap_model)

cur_swap_prob = 0

print('Processing data')
train_set.process(tokenizer, max_sent_len, cur_swap_prob)
dev_set.process(tokenizer, max_sent_len)
test_set.process(tokenizer, max_sent_len)

"""print(train_set.data[5].sentences)

for i in range(10):
    train_set.process(tokenizer, max_sent_len, cur_swap_prob)
    print(train_set.data[5].sentences)

input()"""

vocabs = generate_vocabs([train_set, dev_set, test_set])

if skip_train == False:
    train_set.tensorize(vocabs, config)
dev_set.tensorize(vocabs, config)
test_set.tensorize(vocabs, config)
# valid_patterns = load_valid_patterns(config.valid_pattern_path, vocabs)

if skip_train == False:
    batch_num = len(train_set) // config.batch_size
dev_batch_num = len(dev_set) // config.eval_batch_size + \
                (len(dev_set) % config.eval_batch_size != 0)
test_batch_num = len(test_set) // config.eval_batch_size + \
                 (len(test_set) % config.eval_batch_size != 0)

# vocabs
entity_label_size = len(vocabs['entity'])
event_label_size = trigger_label_size = len(vocabs['event'])
mention_label_size = len(vocabs['mention'])
relation_label_size = len(vocabs['relation'])
role_label_size = len(vocabs['role'])
print(vocabs)

# initialize the model
print('Initialize model')
"""if bert_model.startswith('bert'):
    bert = BertModel.from_pretrained(bert_model,
                                     cache_dir=bert_dir)
else:
    bert = RobertaModel.from_pretrained(bert_model,
                                        cache_dir=bert_dir)"""

"""bert = RobertaModelPlus.from_pretrained(model_name,
                                        cache_dir=bert_dir,
                                        output_hidden_states=True,
                                        fast=True)"""

bert = LongformerModel.from_pretrained("allenai/longformer-base-4096")

#bert = XLNetModel.from_pretrained("xlnet-base-cased")

bert_dim = bert.config.hidden_size
if config.get('use_extra_bert', False):
    bert_dim *= 2
span_len_embed_dim = 32
node_dim = (bert_dim + word_embed_dim) * (1 + config.use_avg_repr + config.use_end_boundary) + span_len_embed_dim
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
"""entity_classifier = Linears([node_dim, 200, 200, entity_label_size],
                            dropout_prob=.2)
trigger_classifier = Linears([node_dim, 200, event_label_size],
                             dropout_prob=.2)
mention_classifier = Linears([node_dim, 100, mention_label_size],
                             dropout_prob=.2)
relation_classifier = PairLinears(node_dim, node_dim, 120, relation_label_size,
                                  dropout_prob=.2)
role_classifier = PairLinears(node_dim, node_dim, 200, role_label_size,
                              dropout_prob=.2)

# entity_classifier_gnn = Linears([node_dim, 150, entity_label_size],
#                             dropout_prob=.4)
# trigger_classifier_gnn = Linears([node_dim, 600, event_label_size],
#                              dropout_prob=.4)
relation_classifier_gnn = PairLinears(node_dim + entity_label_size,
                                      node_dim + entity_label_size,
                                      200,
                                      relation_label_size,
                                      dropout_prob=.2)
role_classifier_gnn = PairLinears(node_dim + trigger_label_size,
                                  node_dim + entity_label_size,
                                  600,
                                  role_label_size,
                                  dropout_prob=.2)

span_len_embed = torch.nn.Embedding(config.max_entity_len, 32)"""

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
"""if use_gnn:
    gnn = GraphConv(2,
                    dimensions,
                    relation_classifier=relation_classifier_gnn,
                    role_classifier=role_classifier_gnn,
                    feat_drop=.2,
                    attn_drop=.1,
                    residual=False)
else:
    gnn = None"""
model = OneIEpp(config,
                vocabs,
                bert,
                node_dim=node_dim,
                word_embed=word_embed,
                gnn=None,
                extra_bert=extra_bert,
                span_repr_dim=node_dim,
                span_comp_dim=512,
                word_embed_dim=word_embed_dim)

# model.load_bert(model_name, cache_dir=config.bert_cache_dir)
if use_gpu:
    model.cuda(device=config.gpu_device)

optimizer = None

if skip_train == False:
    # optimizer

    param_groups = [
        {
            'params': [p for n, p in model.named_parameters() if n.startswith('encoder')],
            'lr': 1e-6, 'weight_decay': 1e-6
        },
        {
            'params': [p for n, p in model.named_parameters() if not n.startswith('encoder')],
            'lr': 1e-4, 'weight_decay': 1e-4
        },
    ]

    optimizer = AdamW(params=param_groups)

    #optimizer = optim.NovoGrad(params=param_groups)

    #optimizer = optim.AdaBound(params=param_groups,
    #                           betas=(0.9, 0.999),
    #                           amsbound=False)

    schedule = get_linear_schedule_with_warmup(optimizer,
                                               num_warmup_steps=batch_num * 5,
                                               num_training_steps=batch_num * epoch_num)

# model state
state = dict(model=model.state_dict(),
             vocabs=vocabs,
             optimizer=optimizer)

#model, vocabs, optimizer = load_model("model.pt", model)

losses = []
best_dev_score = best_test_score = 0.1

global_step = 0

writer = SummaryWriter()

do_test = True

for epoch in range(epoch_num):
    print('******* Epoch {} *******'.format(epoch))

    if epoch > 0:
        if epoch % 5 == 0 and cur_swap_prob < 0.4:
            cur_swap_prob += 0.05
            print("swap prob increased to", cur_swap_prob)

        if cur_swap_prob > 0:
            print("reprocessing train dataset")
            train_set.process(tokenizer, max_sent_len, cur_swap_prob)

    if skip_train == False:
        dataloader = DataLoader(train_set,
                                batch_size,
                                shuffle=True,
                                drop_last=True,
                                collate_fn=train_set.collate_fn)
        for batch_idx, batch in enumerate(tqdm(dataloader, ncols=75)):

            loss, train_loss_names = model(batch, epoch=epoch)
            # print(loss)
            # print(batch.pieces.shape, batch.entity_labels.shape, batch.relation_labels.shape)
            # print(batch.trigger_labels.shape, batch.role_labels.shape)
            if loss is not None:
                loss_sum = sum(loss)
                loss_sum.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                schedule.step()
                optimizer.zero_grad()
                # losses.append(loss.item())
                # if len(losses) == 100:
                #     print('Loss {:.4f}'.format(sum(losses) / len(losses)))
                #     losses = []

                writer.add_scalar('epoch', epoch, global_step)

                lrs = schedule.get_lr()
                writer.add_scalar('lr_bert', lrs[0], global_step)
                #writer.add_scalar('lr_par', lrs[1], global_step)

                #train_loss_names = ["entity", "trigger", "relation", "role"]

                for i, nm in enumerate(train_loss_names):
                    writer.add_scalar('train_' + nm, loss[i], global_step)

                writer.add_scalar('train_loss', loss_sum, global_step)

                # train_entity_span_table = torch.cat((local_scores['entity'].softmax(1),
                #                                     label2onehot(batch.entity_labels,
                #                                                  entity_label_size).float()), dim=1).unsqueeze(0).cpu()

                # writer.add_image("train_entity_span_table", train_entity_span_table, global_step)

                # writer.add_scalar("train_pred_entities", sum(local_scores['entity'].argmax(dim=-1) > 0), global_step)

                # writer.add_image("train_entity_span_scores", local_scores['entity'].softmax(1).unsqueeze(0).cpu(), global_step)
                # writer.add_image("train_entity_span_true", label2onehot(batch.entity_labels, entity_label_size).unsqueeze(0).cpu(),global_step)

                global_step += 1

                # del loss
                # del true_scores
                # del local_scores
                # gc.collect()

        if epoch % 5 == 0:

            gold_train_graphs, pred_train_graphs = [], []

            for batch_idx, batch in enumerate(tqdm(dataloader, ncols=75)):
                if batch_idx % 150 == 0 or batch_idx < 30:
                    result = model.predict(batch, epoch=epoch)

                    pred_graphs = build_information_graph(batch, *result, vocabs)

                    coref_embeds = result[-1]

                    pred_train_graphs.extend(pred_graphs)
                    gold_train_graphs.extend(batch.graphs)

                    if batch_idx % 150 == 0:
                        summary_graph(pred_graphs[0], batch.graphs[0], batch,
                                  writer, global_step, "train_", vocabs, coref_embeds)

            print('Train')
            train_scores = score_graphs(gold_train_graphs, pred_train_graphs, False)

            # writer.add_scalar("dev_entity_num", max_entity_pred, global_step)

            for k, v in train_scores.items():
                writer.add_scalar('train_' + k + '_f', v['f'], global_step)

    # Dev
    dev_loader = DataLoader(dev_set,
                            batch_size,
                            shuffle=False,
                            collate_fn=dev_set.collate_fn)
    gold_dev_graphs, pred_dev_graphs = [], []
    dev_sent_ids, dev_tokens = [], []

    if epoch % 5 == 0:

        for batch_idx, batch in enumerate(tqdm(dev_loader, ncols=75)):
            result = model.predict(batch, epoch=epoch)

            coref_embeds = result[-1]

            #max_entity_pred = max(max_entity_pred, result[3][0])

            # writer.add_image("dev_entity_span_scores", result[1]['entity'].softmax(1).unsqueeze(0).cpu(), global_step)

            """dev_entity_span_table = torch.cat((result[1]['entity'].softmax(1),
                                               label2onehot(batch.entity_labels,
                                                            entity_label_size).float()), dim=1).unsqueeze(0).cpu()
    
            writer.add_image("dev_entity_span_table", dev_entity_span_table, global_step)"""

            pred_graphs = build_information_graph(batch, *result, vocabs)

            if batch_idx % 8 == 0:
                summary_graph(pred_graphs[0], batch.graphs[0], batch,
                          writer, global_step, "dev_", vocabs, coref_embeds)

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
        """save_result(os.path.join(output_path, 'dev.result.{}.json'.format(epoch)),
                    gold_dev_graphs,
                    pred_dev_graphs,
                    dev_sent_ids,
                    tokens=dev_tokens)"""

        #writer.add_scalar("dev_entity_num", max_entity_pred, global_step)

        for k, v in dev_scores.items():
            writer.add_scalar('dev_' + k + '_f', v['f'], global_step)

        score_to_use = "entity"

        if dev_scores[score_to_use]['f'] > best_dev_score:
            print('Saving best dev model by ' + score_to_use)
            torch.save(state, "model.pt")
            best_dev_score = dev_scores[score_to_use]['f']

    if epoch % 5 == 0 and do_test:
        # Test
        test_loader = DataLoader(test_set,
                                 batch_size,
                                 shuffle=False,
                                 collate_fn=test_set.collate_fn)
        gold_test_graphs, pred_test_graphs = [], []
        test_sent_ids, test_tokens = [], []
        for batch in tqdm(test_loader, ncols=75):
            result = model.predict(batch)

            pred_graphs = build_information_graph(batch, *result, vocabs)

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
        """save_result(os.path.join(output_path, 'test.result.{}.json'.format(epoch)),
                    gold_test_graphs,
                    pred_test_graphs,
                    test_sent_ids,
                    tokens=test_tokens)"""

    # torch.save(model, "model.pt")

    #for k, v in test_scores.items():
    #    writer.add_scalar('test_' + k + '_f', v['f'], global_step)

    # log_writer.write(json.dumps({'epoch': epoch,
    #                             'dev': dev_scores,
    #                             'test': test_scores}) + '\n')
    # print('Output path:', output_path)

# log_writer.close()
# best_performance(os.path.join(output_path, 'log.txt'))
