import os
import json
import time
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader
from transformers import (RobertaModel,
                          RobertaTokenizer,
                          AdamW,
                          ElectraTokenizer, ElectraForMaskedLM,
                          get_linear_schedule_with_warmup)
from model import LongIE
from config import Config
from data import IEDataset
from scorer import score_graphs
from util import generate_vocabs, \
    build_information_graph, \
    summary_graph, load_word_embed, get_facts, get_rev_dict

from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm



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

cur_swap_prob = 0.
max_swap_prob = 0.

if max_swap_prob == 0:
    wordswap_tokenizer = wordswap_model = None
else:
    wordswap_tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-generator')
    wordswap_model = ElectraForMaskedLM.from_pretrained('google/electra-small-generator', return_dict=True).cuda()

if config.get("split_by_doc_lens"):
    sent_lens = config.get("sent_lens")

train_set = IEDataset(config.file_dir + config.train_file, config, word_vocab, wordswap_tokenizer, wordswap_model)
dev_set = IEDataset(config.file_dir + config.dev_file, config, word_vocab, wordswap_tokenizer, wordswap_model)
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


print('Processing data')
train_set.process(tokenizer, cur_swap_prob)
dev_set.process(tokenizer, cur_swap_prob)
for test_set in test_sets:
    test_set.process(tokenizer)
if config.get("use_sent_set"):
    test_sent_set.process(tokenizer)


all_sets = [train_set, dev_set, *test_sets]

if config.get("use_sent_set"):
    all_sets.append(test_sent_set)

vocabs = generate_vocabs(all_sets)

if not skip_train:
    train_set.tensorize(vocabs, config)
dev_set.tensorize(vocabs, config)
for test_set in test_sets:
    test_set.tensorize(vocabs, config)
if config.get("use_sent_set"):
    test_sent_set.tensorize(vocabs, config)


if not skip_train:
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


# initialize the model
print('Initialize model')

bert = RobertaModel.from_pretrained(config.bert_model_name)

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
               hidden_dim=config.hidden_dim)

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

    schedule = get_linear_schedule_with_warmup(optimizer,
                                               num_warmup_steps=batch_num * config.warmup_epoch,
                                               num_training_steps=batch_num * epoch_num)



# model state
state = dict(model=model.state_dict(),
             vocabs=vocabs,
             optimizer=optimizer)


losses = []
best_dev_score = best_test_score = 0.3
reset_value = 0.05

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

rel_type_thr = [0.1 for i in range(len(vocabs['relation']))]
evid_type_thr = [0.1 for i in range(len(vocabs['relation']))]


epoch = 0

while epoch < epoch_num:
    print('******* Epoch {} *******'.format(epoch))

    if epoch > 0:
        if epoch % 5 == 0 and cur_swap_prob < max_swap_prob:
            cur_swap_prob += 0.05
            print("swap prob increased to", cur_swap_prob)

        if cur_swap_prob > 0:
            print("reprocessing train dataset")
            train_set.process(tokenizer, cur_swap_prob)
            dev_set.process(tokenizer, cur_swap_prob)
            #??

    if skip_train == False:
        dataloader = DataLoader(train_set,
                                batch_size,
                                shuffle=True,
                                drop_last=True,
                                collate_fn=train_set.collate_fn)
        for batch_idx, batch in enumerate(tqdm(dataloader, ncols=75)):

            if args.debug and batch_idx == 300:
                break

            if epoch == 0 and batch_idx == 1000:
                break

            loss, train_loss_names = model(batch, epoch=epoch)
            if loss is not None:
                loss_sum = sum(loss)
                loss_sum.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                schedule.step()
                optimizer.zero_grad()

                writer.add_scalar('epoch', epoch, global_step)

                lrs = schedule.get_lr()
                writer.add_scalar('lr_bert', lrs[0], global_step)

                for i, nm in enumerate(train_loss_names):
                    writer.add_scalar('train_' + nm, loss[i], global_step)

                writer.add_scalar('train_loss', loss_sum, global_step)


                global_step += 1




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


    if epoch % 2 == 0:

        for batch_idx, batch in enumerate(tqdm(dev_loader, ncols=75)):

            if args.debug and batch_idx == 200:
                break

            if not config.get("only_test_g_i"):
                result = model.predict(batch, epoch=epoch)
                for j, ex_val in enumerate(extra_values):
                    pred_graphs = build_information_graph(batch, *result, vocabs,
                                                          rel_type_thr=rel_type_thr, evid_type_thr=evid_type_thr, extra=ex_val, config=config)
                    pred_dev_graphs[j].extend(pred_graphs)

                if len(batch.tokens[0]) < 400 and batch_idx < 50:
                    summary_graph(pred_graphs[0], batch.graphs[0], batch,
                          writer, global_step + batch_idx, "dev_", vocabs, None, id=batch_idx)

            result_gold_inputs = model.predict(batch, epoch=epoch, gold_inputs=True)

            for j, ex_val in enumerate(extra_values):
                pred_gold_input_graphs = build_information_graph(batch, *result_gold_inputs, vocabs,
                                                             gold_inputs=True, rel_type_thr=rel_type_thr, evid_type_thr=evid_type_thr,
                                                                 extra=ex_val, config=config)

                pred_dev_gold_input_graphs[j].extend(pred_gold_input_graphs)

            if len(batch.tokens[0]) < 400 and batch_idx < 20:
                summary_graph(pred_gold_input_graphs[0], batch.graphs[0], batch,
                          writer, global_step + batch_idx, "dev_gi_", vocabs, None, id=batch_idx)


            gold_dev_graphs.extend(batch.graphs)
            dev_sent_ids.extend(batch.sent_ids)
            dev_tokens.extend(batch.tokens)

        best_dev_scores = None
        best_dev_g_i_scores = None
        best_rel_class_stats = None

        best_ex_val = extra_values[0]


        if not config.get("only_test_g_i"):
            dev_scores = score_graphs(gold_dev_graphs, pred_dev_graphs[0], not symmetric_rel, multitype=multitype)
            for k, v in dev_scores.items():
                writer.add_scalar('dev_' + k + '_f', v['f'], global_step)

        dev_g_i_scores, rel_class_stats, evid_class_stats = score_graphs(gold_dev_graphs, pred_dev_gold_input_graphs[0],
                                                       not symmetric_rel, return_class_scores=True, multitype=multitype)

        for k, v in dev_g_i_scores.items():
            writer.add_scalar('dev_gi_' + k + '_f', v['f'], global_step)


        if config.get("only_test_g_i"):
            cur_dev_score = dev_g_i_scores[judge_value]
        else:
            cur_dev_score = dev_scores[judge_value]



        cur_dev_score = cur_dev_score['f']


        if cur_dev_score > best_dev_score and epoch % 6 == 0:
            print('Saving res for best dev model by ' + judge_value)
            #torch.save(state, "model.pt")
            best_dev_score = cur_dev_score
            is_best = True

            writer.add_scalar("best_dev_score", best_dev_score, global_step)


    if epoch % 6 == 0 and not args.debug:

        gold_train_graphs, pred_train_graphs = [], []
        pred_train_gold_input_graphs = []

        for batch_idx, batch in enumerate(tqdm(dataloader, ncols=75)):
            if batch_idx < 300:

                if not config.get("only_test_g_i"):
                    result = model.predict(batch, epoch=epoch)
                    pred_graphs = build_information_graph(batch, *result, vocabs, rel_type_thr=rel_type_thr, evid_type_thr=evid_type_thr,
                                                          extra=extra_values[0], config=config)
                    # TODO: change

                result_gold_inputs = model.predict(batch, epoch=epoch, gold_inputs=True)

                pred_gold_input_graphs = build_information_graph(batch, *result_gold_inputs, vocabs,
                                                                 gold_inputs=True, rel_type_thr=rel_type_thr, evid_type_thr=evid_type_thr,
                                                                 extra=extra_values[0], config=config)

                pred_train_gold_input_graphs.extend(pred_gold_input_graphs)

                if not config.get("only_test_g_i"):
                    pred_train_graphs.extend(pred_graphs)

                gold_train_graphs.extend(batch.graphs)

        if not config.get("only_test_g_i"):
            print('Train')
            train_scores = score_graphs(gold_train_graphs, pred_train_graphs, not symmetric_rel, multitype=multitype)
        print('Train Gold Inputs')
        train_g_i_scores = score_graphs(gold_train_graphs, pred_train_gold_input_graphs, not symmetric_rel, multitype=multitype)

        if not config.get("only_test_g_i"):
            for k, v in train_scores.items():
                writer.add_scalar('train_' + k + '_f', v['f'], global_step)

        for k, v in train_g_i_scores.items():
            writer.add_scalar('train_gi_' + k + '_f', v['f'], global_step)

    if epoch % 6 == 0 and do_test and not(produce_outputs and not is_best) and not args.debug:

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
                    pred_graphs = build_information_graph(batch, *result, vocabs, rel_type_thr=rel_type_thr, evid_type_thr=evid_type_thr,
                                                          extra=best_ex_val, config=config)

                result_gold_inputs = model.predict(batch, epoch=epoch, gold_inputs=True)

                pred_gold_input_graphs = build_information_graph(batch, *result_gold_inputs, vocabs,
                                                                 gold_inputs=True, rel_type_thr=rel_type_thr, evid_type_thr=evid_type_thr,
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
                          writer, global_step + batch_idx, "sent_", vocabs, None)

            result_gold_inputs = model.predict(batch, epoch=epoch, gold_inputs=True)

            pred_gold_input_graphs = build_information_graph(batch, *result_gold_inputs, vocabs,
                                                             gold_inputs=True, rel_type_thr=rel_type_thr, evid_type_thr=evid_type_thr, config=config)

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


    epoch += 1