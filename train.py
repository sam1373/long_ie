import os
import json
import time
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader
from transformers import (LongformerModel, LongformerTokenizer, RobertaModel,
                          RobertaTokenizer, BertConfig, BertTokenizer,
                          AutoTokenizer,
                          AutoModel, BertModel,
                          AdamW,
                          ElectraTokenizer, ElectraForMaskedLM,
                          XLNetModel, XLNetTokenizer,
                          get_linear_schedule_with_warmup)
from model import LongIE, Linears, PairLinears
from graph import Graph
from config import Config
# from data import IEDataset
from data2 import IEDataset
from scorer import score_graphs
from util import generate_vocabs, load_valid_patterns, save_result, best_score_by_task, \
    build_information_graph, label2onehot, load_model, \
    summary_graph, load_word_embed, get_facts, get_rev_dict, adjust_thresholds

from torch.utils.tensorboard import SummaryWriter

#from roberta import RobertaModelPlus

from tqdm import tqdm

from optim import GradualWarmupScheduler


#import torch_optimizer as optim

import gc





skip_train = False


# configuration
parser = ArgumentParser()
parser.add_argument('-c', '--config', default='config/example.json')
parser.add_argument('-d', '--debug', default=False)
parser.add_argument('-r', '--result_name', default='result')
args = parser.parse_args()
config = Config.from_json_file(args.config)
# print(config.to_dict())

produce_outputs = config.get("produce_outputs")

if produce_outputs:
    rev_dict = get_rev_dict("output/docred/rel_info.json")

max_sent_len = config.get('max_sent_len', 2000)
symmetric_rel = config.get('symmetric_relations')
multitype = config.get("relation_type_level") == "multitype"

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

    embed_file = config.word_embed_file
    word_embed_dim = 200

    print('Loading word embeddings from %s' % embed_file)
    word_embed, word_vocab = load_word_embed(embed_file,
                                             word_embed_dim,
                                             skip_first=True)

# datasets
model_name = config.bert_model_name

tokenizer = RobertaTokenizer.from_pretrained(config.bert_model_name)


#tokenizer = BertTokenizer.from_pretrained("allenai/scibert_scivocab_cased")

#tokenizer = BertTokenizer.from_pretrained("SpanBERT/spanbert-base-cased")

#tokenizer = AutoTokenizer.from_pretrained("SpanBERT/spanbert-base-cased")

#tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")

wordswap_tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-generator')
wordswap_model = ElectraForMaskedLM.from_pretrained('google/electra-small-generator', return_dict=True).cuda()

#ace+
sent_lens = [0, 300, 500, 1000, 3000]
#scierc
#sent_lens = [0, 100, 200, 300, 500]

"""if args.debug:
    train_set = IEDataset(config.file_dir + config.dev_file, config, word_vocab, wordswap_tokenizer, wordswap_model)
    dev_set = IEDataset(config.file_dir + config.dev_file, config, word_vocab)
    test_sets = [IEDataset(config.file_dir + config.dev_file, config, word_vocab)]
    if config.get("use_sent_set"):
        test_sent_set = IEDataset(config.file_dir + config.sent_set_file, config, word_vocab)
else:"""
train_set = IEDataset(config.file_dir + config.train_file, config, word_vocab, wordswap_tokenizer, wordswap_model)
dev_set = IEDataset(config.file_dir + config.dev_file, config, word_vocab)
if config.get("split_by_doc_lens"):
    test_sets = []
    for i in range(1, len(sent_lens)):
        max_len = sent_lens[i]
        min_len = sent_lens[i - 1]
        test_sets.append(IEDataset(config.file_dir + config.test_file, config, word_vocab,
                                   min_sent_len=min_len, max_sent_len=max_len))
else:
    test_sets = [IEDataset(config.file_dir + config.test_file, config, word_vocab)]
if config.get("use_sent_set"):
    test_sent_set = IEDataset(config.file_dir + config.sent_set_file, config, word_vocab)

cur_swap_prob = 0.

print('Processing data')
train_set.process(tokenizer, cur_swap_prob)
dev_set.process(tokenizer)
for test_set in test_sets:
    test_set.process(tokenizer)
if config.get("use_sent_set"):
    test_sent_set.process(tokenizer)

"""print(train_set.data[5].sentences)

for i in range(10):
    train_set.process(tokenizer, max_sent_len, cur_swap_prob)
    print(train_set.data[5].sentences)

input()"""

all_sets = [train_set, dev_set, *test_sets]

if config.get("use_sent_set"):
    all_sets.append(test_sent_set)

vocabs = generate_vocabs(all_sets)

if skip_train == False:
    train_set.tensorize(vocabs, config)
dev_set.tensorize(vocabs, config)
for test_set in test_sets:
    test_set.tensorize(vocabs, config)
if config.get("use_sent_set"):
    test_sent_set.tensorize(vocabs, config)
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
#print(vocabs)

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

bert = RobertaModel.from_pretrained(config.bert_model_name)
#bert = LongformerModel.from_pretrained(config.bert_model_name)

#bert = BertModel.from_pretrained("SpanBERT/spanbert-base-cased")
#bert = XLNetModel.from_pretrained("xlnet-base-cased")

#bert = AutoModel.from_pretrained("allenai/scibert_scivocab_cased")

bert_dim = bert.config.hidden_size
if config.get('use_extra_bert', False):
    bert_dim *= 2

model = LongIE(config,
               vocabs,
               bert,
               word_embed=word_embed,
               extra_bert=extra_bert,
               word_embed_dim=word_embed_dim,
               coref_embed_dim=config.coref_embed_dim,
               hidden_dim=config.hidden_dim, )

# model.load_bert(model_name, cache_dir=config.bert_cache_dir)
if use_gpu:
    model.cuda(device=config.gpu_device)

optimizer = None

if skip_train == False:
    # optimizer

    fine_tune_param_names = [n for n, p in model.named_parameters() if (n.startswith('encoder') or
                                                                        n.startswith('word_embed'))]
    #print(fine_tune_param_names)
    fine_tune_param_names = set(fine_tune_param_names)

    param_groups = [
        {
            'params': [p for n, p in model.named_parameters() if n in fine_tune_param_names],
            'lr': config.bert_learning_rate, 'weight_decay': config.weight_decay
        },
        {
            'params': [p for n, p in model.named_parameters() if not n in fine_tune_param_names],
            'lr': config.learning_rate, 'weight_decay': config.weight_decay
        },
    ]

    optimizer = AdamW(params=param_groups)

    #optimizer = optim.NovoGrad(params=param_groups)

    #optimizer = optim.AdaBound(params=param_groups,
    #                           betas=(0.9, 0.999),
    #                           amsbound=False)

    #schedule = get_linear_schedule_with_warmup(optimizer,
    #                                           num_warmup_steps=batch_num * config.warmup_epoch,
    #                                           num_training_steps=batch_num * epoch_num)

    schedule_reduce_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                    mode='max',
                                                                    verbose=True,
                                                                    patience=20
                                                                    )

    schedule = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=config.get("warmup_epoch"),
                                      after_scheduler=schedule_reduce_lr)

    schedule.step(epoch=0, metrics=0)

"""for i in range(1, 120):
    print(i)
    print(schedule.get_lr()[0])

    cur_score = float(input())

    schedule.step(epoch=i, metrics=cur_score)"""

# model state
state = dict(model=model.state_dict(),
             vocabs=vocabs,
             optimizer=optimizer)

#model, vocabs, optimizer = load_model("model.pt", model)

losses = []
best_dev_score = best_test_score = 0.3

global_step = 0

writer = SummaryWriter()

do_test = config.get("do_test", True)

cur_dev_score = 0

# additional values to be passed into build_information_graph

# evidence threshold
extra_values_0 = [0.5]
# non-zero relation cand threshold
extra_values_1 = [0.1]
# relation type prediction threshold
# extra_values_2 = [0.2, 0.3, 0.4]

extra_values = []

for i in extra_values_0:
    for j in extra_values_1:
        #for k in extra_values_2:
        extra_values.append([i, j])

extra_value_num = len(extra_values)

rel_type_thr = [0.3 for i in range(len(vocabs['relation']))]

for epoch in range(epoch_num):
    print('******* Epoch {} *******'.format(epoch))

    if epoch > 0 and not args.debug:
        if epoch % 5 == 0 and cur_swap_prob < 0.:
            cur_swap_prob += 0.05
            print("swap prob increased to", cur_swap_prob)

        if cur_swap_prob > 0:
            print("reprocessing train dataset")
            train_set.process(tokenizer, cur_swap_prob)

    if skip_train == False:
        dataloader = DataLoader(train_set,
                                batch_size,
                                shuffle=True,
                                drop_last=True,
                                collate_fn=train_set.collate_fn)
        for batch_idx, batch in enumerate(tqdm(dataloader, ncols=75)):

            if args.debug and batch_idx == 200:
                break

            loss, train_loss_names = model(batch, epoch=epoch)
            # print(loss)
            # print(batch.pieces.shape, batch.entity_labels.shape, batch.relation_labels.shape)
            # print(batch.trigger_labels.shape, batch.role_labels.shape)
            if loss is not None:
                loss_sum = sum(loss)
                loss_sum.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                #schedule.step()
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



    # Dev
    is_best = False

    dev_loader = DataLoader(dev_set,
                            batch_size,
                            shuffle=False,
                            collate_fn=dev_set.collate_fn)
    gold_dev_graphs = []
    pred_dev_graphs, pred_dev_gold_input_graphs = [], []
    dev_sent_ids, dev_tokens = [], []





    for j in range(extra_value_num):
        pred_dev_graphs.append([])
        pred_dev_gold_input_graphs.append([])

    judge_value = "evidence"


    if epoch % 5 == 0:

        for batch_idx, batch in enumerate(tqdm(dev_loader, ncols=75)):

            if args.debug and batch_idx == 200:
                break

            if not config.get("only_test_g_i"):
                result = model.predict(batch, epoch=epoch)
                for j, ex_val in enumerate(extra_values):
                    pred_graphs = build_information_graph(batch, *result, vocabs,
                                                          rel_type_thr=rel_type_thr, extra=ex_val, config=config)
                    pred_dev_graphs[j].extend(pred_graphs)

                if len(batch.tokens[0]) < 400 and batch_idx < 20:
                    summary_graph(pred_graphs[0], batch.graphs[0], batch,
                          writer, global_step, "dev_", vocabs, None, id=batch_idx)

            result_gold_inputs = model.predict(batch, epoch=epoch, gold_inputs=True)

            for j, ex_val in enumerate(extra_values):
                pred_gold_input_graphs = build_information_graph(batch, *result_gold_inputs, vocabs,
                                                             gold_inputs=True, rel_type_thr=rel_type_thr,
                                                                 extra=ex_val, config=config)

                pred_dev_gold_input_graphs[j].extend(pred_gold_input_graphs)

            if len(batch.tokens[0]) < 400 and batch_idx < 20:
                summary_graph(pred_gold_input_graphs[0], batch.graphs[0], batch,
                          writer, global_step, "dev_gi_", vocabs, None, id=batch_idx)


            gold_dev_graphs.extend(batch.graphs)
            dev_sent_ids.extend(batch.sent_ids)
            dev_tokens.extend(batch.tokens)
        # gold_dev_graphs = [g.clean(False)
        #                    for g in gold_dev_graphs]
        # pred_dev_graphs = [g.clean(False)
        #                    for g in pred_dev_graphs]

        #dev_scores = []
        #dev_g_i_scores = []
        best_dev_scores = None
        best_dev_g_i_scores = None
        best_rel_class_stats = None

        best_ex_val = -1
        best_ex_val_res = 0

        for j, ex_val in enumerate(extra_values):
            print('Extra Value:', ex_val)
            dev_scores = None
            if not config.get("only_test_g_i"):
                print('Dev')
                dev_scores = score_graphs(gold_dev_graphs, pred_dev_graphs[j], not symmetric_rel, multitype=multitype)

            print('Dev Gold Inputs')
            dev_g_i_scores, rel_class_stats = score_graphs(gold_dev_graphs, pred_dev_gold_input_graphs[j],
                                                           not symmetric_rel, return_class_scores=True, multitype=multitype)

            cur_judge_value = dev_g_i_scores[judge_value]['f']
            if not config.get("only_test_g_i"):
                cur_judge_value = dev_scores[judge_value]['f']

            if cur_judge_value >= best_ex_val_res:
                best_ex_val = ex_val
                best_ex_val_res = cur_judge_value
                best_dev_scores = dev_scores
                best_dev_g_i_scores = dev_g_i_scores
                best_rel_class_stats = rel_class_stats

        #writer.add_scalar("dev_entity_num", max_entity_pred, global_step)
        for k in range(len(best_ex_val)):
            writer.add_scalar('best_ex_val_'+str(k), best_ex_val[k], global_step)
        print('Best extra value:', best_ex_val)

        if not config.get("only_test_g_i"):
            for k, v in best_dev_scores.items():
                writer.add_scalar('dev_' + k + '_f', v['f'], global_step)

        for k, v in best_dev_g_i_scores.items():
            writer.add_scalar('dev_gi_' + k + '_f', v['f'], global_step)

        print("Relation Class Metrics:")
        for (rel_type, metrics) in rel_class_stats.items():
            print(rel_type, "~ thr:", round(rel_type_thr[vocabs["relation"][rel_type]], 2),
                  "~ prec:", round(metrics['prec'], 2),
                  "rec:", round(metrics['rec'], 2),
                  "f:", round(metrics['f'], 2))


        adjust_thresholds(rel_type_thr, rel_class_stats, vocabs["relation"], epoch)

        if config.get("only_test_g_i"):
            cur_dev_score = best_dev_g_i_scores[judge_value]
        else:
            cur_dev_score = best_dev_scores[judge_value]

        if cur_dev_score['prec'] > cur_dev_score['rec'] + 0.03:
            extra_values[0][0] -= 0.05
        elif cur_dev_score['prec'] < cur_dev_score['rec'] - 0.03:
            extra_values[0][0] += 0.05

        cur_dev_score = cur_dev_score['f']

        if cur_dev_score > best_dev_score:
            print('Saving res for best dev model by ' + judge_value)
            #torch.save(state, "model.pt")
            best_dev_score = cur_dev_score
            is_best = True

    schedule.step(epoch=epoch + 1, metrics=cur_dev_score)

    if epoch % 5 == 0 and not args.debug:

        gold_train_graphs, pred_train_graphs = [], []
        pred_train_gold_input_graphs = []

        for batch_idx, batch in enumerate(tqdm(dataloader, ncols=75)):
            if batch_idx < 300:

                if not config.get("only_test_g_i"):
                    result = model.predict(batch, epoch=epoch)
                    pred_graphs = build_information_graph(batch, *result, vocabs, rel_type_thr=rel_type_thr,
                                                          extra=best_ex_val, config=config)
                    # TODO: change

                result_gold_inputs = model.predict(batch, epoch=epoch, gold_inputs=True)

                pred_gold_input_graphs = build_information_graph(batch, *result_gold_inputs, vocabs,
                                                                 gold_inputs=True, rel_type_thr=rel_type_thr,
                                                                 extra=best_ex_val, config=config)

                pred_train_gold_input_graphs.extend(pred_gold_input_graphs)

                if not config.get("only_test_g_i"):
                    pred_train_graphs.extend(pred_graphs)
                    # summary_graph(pred_graphs[0], batch.graphs[0], batch,
                    #    writer, global_step, "train_", vocabs, None, id=batch_idx)

                gold_train_graphs.extend(batch.graphs)

        if not config.get("only_test_g_i"):
            print('Train')
            train_scores = score_graphs(gold_train_graphs, pred_train_graphs, not symmetric_rel, multitype=multitype)
        print('Train Gold Inputs')
        train_g_i_scores = score_graphs(gold_train_graphs, pred_train_gold_input_graphs, not symmetric_rel, multitype=multitype)

        # writer.add_scalar("dev_entity_num", max_entity_pred, global_step)

        if not config.get("only_test_g_i"):
            for k, v in train_scores.items():
                writer.add_scalar('train_' + k + '_f', v['f'], global_step)

        for k, v in train_g_i_scores.items():
            writer.add_scalar('train_gi_' + k + '_f', v['f'], global_step)

    if epoch % 5 == 0 and do_test and not(produce_outputs and not is_best) and not args.debug:

        fact_dict_list = []

        for ts_idx, test_set in enumerate(test_sets):
            if len(test_set) == 0:
                continue
            # Test
            test_loader = DataLoader(test_set,
                                     batch_size,
                                     shuffle=False,
                                     collate_fn=test_set.collate_fn)
            gold_test_graphs, pred_test_graphs, pred_test_gold_input_graphs = [], [], []
            test_sent_ids, test_tokens = [], []
            for batch in tqdm(test_loader, ncols=75):

                if not config.get("only_test_g_i"):
                    result = model.predict(batch)
                    pred_graphs = build_information_graph(batch, *result, vocabs, rel_type_thr=rel_type_thr,
                                                          extra=best_ex_val, config=config)

                result_gold_inputs = model.predict(batch, epoch=epoch, gold_inputs=True)

                pred_gold_input_graphs = build_information_graph(batch, *result_gold_inputs, vocabs,
                                                                 gold_inputs=True, rel_type_thr=rel_type_thr,
                                                                 extra=best_ex_val, config=config)

                pred_test_gold_input_graphs.extend(pred_gold_input_graphs)

                if not config.get("only_test_g_i"):
                    pred_test_graphs.extend(pred_graphs)

                gold_test_graphs.extend(batch.graphs)
                test_sent_ids.extend(batch.sent_ids)
                test_tokens.extend(batch.tokens)

                if produce_outputs:
                    fact_dict_list.extend(get_facts(pred_gold_input_graphs, batch.sent_ids, rev_dict))

            if is_best and produce_outputs:
                print("saving test results in", "output/" + args.result_name + ".json")
                json.dump(fact_dict_list, open("output/" + args.result_name + ".json", "w"))

            if not config.get("only_test_g_i"):
                print('Test Set', ts_idx)
                test_scores = score_graphs(gold_test_graphs, pred_test_graphs, not symmetric_rel, multitype=multitype)

                for k, v in test_scores.items():
                    writer.add_scalar('test_' + str(ts_idx) + '_' + k + '_f', v['f'], global_step)

            if not produce_outputs:
                print('Test', ts_idx, 'Gold Inputs')
                test_g_i_scores = score_graphs(gold_test_graphs, pred_test_gold_input_graphs, not symmetric_rel, gold_inputs=True, multitype=multitype)

                for k, v in test_g_i_scores.items():
                    writer.add_scalar('test_gi_' + str(ts_idx) + '_' + k + '_f', v['f'], global_step)




    if epoch % 5 == 0 and config.get("use_sent_set"):

        # Test Sent
        test_loader = DataLoader(test_sent_set,
                                 batch_size,
                                 shuffle=False,
                                 collate_fn=test_sent_set.collate_fn)
        gold_test_graphs, pred_test_graphs, pred_test_gold_input_graphs = [], [], []
        test_sent_ids, test_tokens = [], []
        for batch_idx, batch in enumerate(tqdm(test_loader, ncols=75)):
            result = model.predict(batch)

            pred_graphs = build_information_graph(batch, *result, vocabs, config=config, rel_type_thr=rel_type_thr)

            if batch_idx % 500 == 0:
                summary_graph(pred_graphs[0], batch.graphs[0], batch,
                          writer, global_step, "sent_", vocabs, None)

            result_gold_inputs = model.predict(batch, epoch=epoch, gold_inputs=True)

            pred_gold_input_graphs = build_information_graph(batch, *result_gold_inputs, vocabs,
                                                             gold_inputs=True, rel_type_thr=rel_type_thr, config=config)

            pred_test_gold_input_graphs.extend(pred_gold_input_graphs)

            pred_test_graphs.extend(pred_graphs)
            gold_test_graphs.extend(batch.graphs)
            test_sent_ids.extend(batch.sent_ids)
            test_tokens.extend(batch.tokens)

        print('Test Sent')
        test_scores = score_graphs(gold_test_graphs, pred_test_graphs, not symmetric_rel, multitype=multitype)
        print('Test Sent Gold Inputs')
        test_g_i_scores = score_graphs(gold_test_graphs, pred_test_gold_input_graphs, not symmetric_rel, gold_inputs=True, multitype=multitype)

        for k, v in test_scores.items():
            writer.add_scalar('test_sent_' + k + '_f', v['f'], global_step)

        for k, v in test_g_i_scores.items():
            writer.add_scalar('test_sent_gi_' + k + '_f', v['f'], global_step)

    # log_writer.write(json.dumps({'epoch': epoch,
    #                             'dev': dev_scores,
    #                             'test': test_scores}) + '\n')
    # print('Output path:', output_path)

# log_writer.close()
# best_performance(os.path.join(output_path, 'log.txt'))