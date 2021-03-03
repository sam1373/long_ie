import copy
import json
import os

from typing import Dict, Any

from transformers import (BertConfig, RobertaConfig, XLMRobertaConfig, ElectraConfig,
                          PretrainedConfig)

class Config():
    def __init__(self, **kwargs):
        self.coref = kwargs.pop('coref', True)
        # bert
        self.bert_model_name = kwargs.pop('bert_model_name', 'bert-large-cased')
        self.bert_cache_dir = kwargs.pop('bert_cache_dir', None)
        self.extra_bert = kwargs.pop('extra_bert', 0)
        self.use_extra_bert = kwargs.pop('use_extra_bert', False)
        # model
        self.multi_piece_strategy = kwargs.pop('multi_piece_strategy', 'first')
        self.bert_dropout = kwargs.pop('bert_dropout', .2)
        self.hidden_dim = kwargs.pop('hidden_dim', 500)
        self.hidden_dim = kwargs.pop('span_transformer_layers', 10)
        self.hidden_dim = kwargs.pop('coref_embed_dim', 64)
        # files
        self.file_dir = kwargs.pop('file_dir', None)
        self.train_file = kwargs.pop('train_file', None)
        self.dev_file = kwargs.pop('dev_file', None)
        self.test_file = kwargs.pop('test_file', None)
        self.log_path = kwargs.pop('log_path', None)
        self.split_by_doc_lens = kwargs.pop('split_by_doc_lens', False)
        self.do_test = kwargs.pop('do_test', True)

        self.use_sent_set = kwargs.pop('use_sent_set', False)
        self.sent_set_file = kwargs.pop('sent_set_file', None)
        # training
        self.batch_size = kwargs.pop('batch_size', 10)
        self.eval_batch_size = kwargs.pop('eval_batch_size', 1)
        self.max_epoch = kwargs.pop('max_epoch', 50)
        self.learning_rate = kwargs.pop('learning_rate', 1e-3)
        self.bert_learning_rate = kwargs.pop('bert_learning_rate', 1e-5)
        self.weight_decay = kwargs.pop('weight_decay', 0.001)
        self.bert_weight_decay = kwargs.pop('bert_weight_decay', 0.00001)
        self.warmup_epoch = kwargs.pop('warmup_epoch', 5)
        self.grad_clipping = kwargs.pop('grad_clipping', 5.0)
        # others
        self.symmetric_relations = kwargs.pop('symmetric_relations', True)
        self.only_in_sent_rels = kwargs.pop('only_in_sent_rels', False)

        self.remove_pro = kwargs.pop('remove_pro', False)

        self.use_gpu = kwargs.pop('use_gpu', True)
        self.gpu_device = kwargs.pop('gpu_device', -1)

        self.max_entity_len = kwargs.pop('max_entity_len', 8)
        self.max_trigger_len = kwargs.pop('max_trigger_len', 3)

        self.use_avg_repr = kwargs.pop('use_avg_repr', False)
        self.use_end_boundary = kwargs.pop('use_avg_repr', True)

        self.use_first_wp = kwargs.pop('use_first_wp', False)

        self.use_extra_word_embed = kwargs.pop('use_extra_word_embed', False)

        self.use_sent_num_embed = kwargs.pop('use_sent_num_embed', False)
        self.sent_num_embed_dim = kwargs.pop('sent_num_embed_dim', 128)
        self.use_sent_context = kwargs.pop('use_sent_context', False)

        self.max_sent_len = kwargs.pop("max_sent_len", 2000)

        self.classify_entities = kwargs.pop("classify_entities", True)
        self.classify_relations = kwargs.pop("classify_relations", True)
        self.classify_triggers = kwargs.pop("classify_triggers", True)
        self.classify_roles = kwargs.pop("classify_roles", True)
        self.do_coref = kwargs.pop("do_coref", True)
        self.classify_evidence = kwargs.pop("classify_evidence", False)

        self.only_test_g_i = kwargs.pop("only_test_g_i", False)
        self.only_train_g_i = kwargs.pop("only_train_g_i", False)

        self.produce_outputs = kwargs.pop("produce_outputs", False)

        self.truncate_long_docs = kwargs.pop("truncate_long_docs", True)


    def get(self, attr, default=None):
        return getattr(self, attr, default)

    @classmethod
    def from_dict(cls, dict_obj):
        """Creates a Config object from a dictionary.
        Args:
            dict_obj (Dict[str, Any]): a dict where keys are
        """
        config = cls()
        for k, v in dict_obj.items():
            setattr(config, k, v)
        return config

    @classmethod
    def from_json_file(cls, path):
        with open(path, 'r', encoding='utf-8') as r:
            return cls.from_dict(json.load(r))

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        return output

    def save_config(self, path):
        """Save a configuration object to a file.
        :param path (str): path to the output file or its parent directory.
        """
        if os.path.isdir(path):
            path = os.path.join(path, 'config.json')
        print('Save config to {}'.format(path))
        with open(path, 'w', encoding='utf-8') as w:
            w.write(json.dumps(self.to_dict(), indent=2,
                               sort_keys=True))
    @property
    def bert_config(self):
        if self.bert_model_name.startswith('bert-'):
            return BertConfig.from_pretrained(self.bert_model_name,
                                              cache_dir=self.bert_cache_dir)
        elif 'roberta' in self.bert_model_name:
            return RobertaConfig.from_pretrained(self.bert_model_name,
                                                 cache_dir=self.bert_cache_dir)
        elif self.bert_model_name.startswith('xlm-roberta-'):
            return XLMRobertaConfig.from_pretrained(self.bert_model_name,
                                                    cache_dir=self.bert_cache_dir)
        elif 'electra' in self.bert_model_name:
            return ElectraConfig.from_pretrained(self.bert_model_name,
                                                    cache_dir=self.bert_cache_dir)
        else:
            raise ValueError('Unknown model: {}'.format(self.bert_model_name))