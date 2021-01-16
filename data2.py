#from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any, Union, Set
from random import choice, randrange, random


import torch
from transformers import (PreTrainedTokenizer,
                          BertTokenizer,
                          RobertaTokenizer,
                          XLMRobertaTokenizer)
from torch.utils.data import Dataset

from graph import Graph
from util import enumerate_spans, augment

from tqdm import tqdm

import copy

import string

logger = logging.getLogger(__name__)


@dataclass
class Batch:
    tokens: List[List[str]]
    pieces: torch.LongTensor
    token_lens: List[List[int]]
    attention_mask: torch.FloatTensor
    entity_idxs: torch.LongTensor
    entity_mask: torch.FloatTensor
    entity_boundaries: torch.LongTensor
    trigger_idxs: torch.LongTensor
    trigger_mask: torch.FloatTensor
    trigger_boundaries: torch.LongTensor
    entity_labels: torch.LongTensor
    mention_labels: torch.LongTensor
    trigger_labels: torch.LongTensor
    id_entity_labels: torch.LongTensor
    id_mention_labels: torch.LongTensor
    id_trigger_labels: torch.LongTensor
    relation_labels: torch.LongTensor
    coref_labels: torch.LongTensor
    role_labels: torch.LongTensor
    max_entity_len: int
    max_trigger_len: int
    pos_entity_idxs: torch.LongTensor
    pos_trigger_idxs: torch.LongTensor
    pos_entity_offsets: List[List[Tuple[int, int]]]
    pos_trigger_offsets: List[List[Tuple[int, int]]]
    entity_span_mask: torch.LongTensor
    trigger_span_mask: torch.LongTensor
    entity_offsets: List[Tuple[int, int]]
    trigger_offsets: List[Tuple[int, int]]
    entity_lens: torch.FloatTensor
    trigger_lens: torch.FloatTensor
    entity_labels_sep: List[torch.LongTensor]
    trigger_labels_sep: List[torch.LongTensor]
    relation_labels_sep: List[torch.LongTensor]
    role_labels_sep: List[torch.LongTensor]
    graphs: List[Graph]
    sent_ids: List[str]
    pieces_text: List[List[str]]
    token_embed_ids: List[List[int]]
    entities_coref: List[Dict]
    mention_to_ent_coref: List[List[int]]
    is_start: torch.LongTensor
    len_from_here: torch.LongTensor
    type_from_here: torch.LongTensor

    @property
    def batch_size(self):
        return len(self.tokens)


@dataclass
class Span:
    text: str
    start: int
    end: int

    @staticmethod
    def from_dict(dict_obj: Dict[str, Any], **kwargs):
        return Span(text=dict_obj.get('text', ''),
                    start=dict_obj['start'],
                    end=dict_obj['end'])


@dataclass
class Entity(Span):
    entity_id: str
    mention_id: str
    entity_type: str
    mention_type: str = 'UNK'
    entity_subtype: str = None
    entity_subsubtype: str = None
    data: Dict[str, Any] = None

    def __post_init__(self):
        self.data = {}

    @staticmethod
    def from_dict(dict_obj: Dict[str, Any], **kwargs):
        return Entity(text=dict_obj.get('text', ''),
                      start=dict_obj['start'],
                      end=dict_obj['end'],
                      entity_id=dict_obj['entity_id'],
                      mention_id=dict_obj['mention_id'],
                      entity_type=dict_obj['entity_type'],
                      entity_subtype=dict_obj.get('entity_subtype', None),
                      mention_type=dict_obj.get('mention_type', 'UNK'),
                      )

    @property
    def is_filler(self):
        return self.mention_type == 'FILLER'

    @property
    def uid(self):
        return '{}_{}'.format(self.entity_id, self.mention_id)

    def get_type(self, level: str = 'type'):
        if level == 'type':
            return self.entity_type
        elif level == 'subtype':
            return '{}.{}'.format(self.entity_type, self.entity_subtype)
        elif level == 'subsubtype':
            return '{}.{}.{}'.format(self.entity_type,
                                     self.entity_subtype,
                                     self.entity_subsubtype)
        else:
            raise ValueError('Unknown type level: {}'.format(level))


@dataclass
class Trigger(Span):
    data: Dict[str, Any] = None

    def __post_init__(self):
        self.data = {}

    @staticmethod
    def from_dict(dict_obj: Dict[str, Any], **kwargs):
        return Trigger(text=dict_obj.get('text', ''),
                       start=dict_obj['start'],
                       end=dict_obj['end'])


@dataclass
class EventArgument:
    entity_id: str
    mention_id: str
    role: str
    text: str
    realis: Union[str, bool] = True
    data: Dict[str, Any] = None
    role_idx: int = 0

    def __post_init__(self):
        self.data = {}

    @staticmethod
    def from_dict(dict_obj: Dict[str, Any], **kwargs):
        return EventArgument(entity_id=dict_obj['entity_id'][:dict_obj['entity_id'].rfind("-")],
                             mention_id=dict_obj['entity_id'],
                             role=dict_obj['role'],
                             text=dict_obj['text'],
                             realis=dict_obj.get('realis', True))

    @property
    def uid(self):
        return '{}_{}'.format(self.entity_id, self.mention_id)


@dataclass
class Event:
    event_id: str
    mention_id: str
    trigger: Trigger
    arguments: List[EventArgument]
    event_type: str
    event_subtype: str = None
    realis: str = None
    event_subsubtype: str = None
    data: Dict[str, Any] = None
    event_type_idx: int = 0

    def __post_init__(self):
        self.data = {}

    @staticmethod
    def from_dict(dict_obj: Dict[str, Any], **kwargs):
        return Event(event_id=dict_obj['event_id'],
                     mention_id=dict_obj['mention_id'],
                     trigger=Trigger.from_dict(dict_obj['trigger']),
                     arguments=[EventArgument.from_dict(arg, **kwargs)
                                for arg in dict_obj['arguments']],
                     event_type=dict_obj['event_type'],
                     event_subtype=dict_obj.get('event_subtype', None),
                     event_subsubtype=dict_obj.get('event_subsubtype', None),
                     realis=dict_obj.get('realis', 'actual')
                     )

    @property
    def uid(self):
        return '{}_{}'.format(self.event_id, self.mention_id)

    @property
    def start(self) -> int:
        return self.trigger.start

    @property
    def end(self) -> int:
        return self.trigger.end

    def get_type(self, level: str = 'type'):
        if level == 'type':
            return self.event_type
        elif level =='subtype':
            return '{}.{}'.format(self.event_type, self.event_subtype)
        elif level == 'subsubtype':
            return '{}.{}.{}'.format(self.event_type,
                                     self.event_subtype,
                                     self.event_subsubtype)
        else:
            return ValueError('Unknown type level: {}'.format(level))


@dataclass
class RelationArgument:
    entity_id: str
    mention_id: str
    role: str = None
    text: str = ''
    data: Dict[str, Any] = None

    def __post_init__(self):
        self.data = {}

    @staticmethod
    def from_dict(dict_obj: Dict[str, Any], **kwargs):
        return RelationArgument(entity_id=dict_obj['entity_id'][:dict_obj['entity_id'].rfind("-")],
                                mention_id=dict_obj['entity_id'],
                                role=dict_obj.get('role_id', None),
                                text=dict_obj.get('text', ''))

    @property
    def uid(self):
        return '{}_{}'.format(self.entity_id, self.mention_id)


@dataclass
class Relation:
    relation_id: str
    mention_id: str
    relation_type: str
    arg1: RelationArgument
    arg2: RelationArgument
    relation_type_idx: int = 0
    relation_type_idx_rev: int = 0
    relation_subtype: str = None
    data: Dict[str, Any] = None
    is_symmetric: bool = False

    def __post_init__(self):
        self.data = {}

    @staticmethod
    def from_dict(dict_obj: Dict[str, Any], **kwargs):
        # symmetric_relations = kwargs.get('symmetric_relations', [])
        return Relation(relation_id=dict_obj['relation_id'],
                        mention_id=dict_obj['mention_id'],
                        relation_type=dict_obj['relation_type'],
                        relation_subtype=dict_obj['relation_subtype'],
                        arg1=RelationArgument.from_dict(dict_obj['arguments'][0], **kwargs),
                        arg2=RelationArgument.from_dict(dict_obj['arguments'][1], **kwargs),
                        # is_symmetric=dict_obj['relation_type'] in symmetric_relations
                        )

    @property
    def uid(self):
        return '{}.{}'.format(self.relation_id, self.mention_id)

    def get_type(self, level: str = 'type'):
        if level == 'type':
            return self.relation_type
        elif level == 'subtype':
            return '{}.{}'.format(self.relation_type, self.relation_subtype)
        else:
            return 'Unknonw type level: {}'.format(level)


@dataclass
class Sentence:
    sent_id: str
    tokens: List[str]
    entities: List[Entity]
    relations: List[Relation]
    events: List[Event]
    pieces: List[str] = None
    token_lens: List[int] = None
    data: Dict[str, Any] = None
    piece_idxs: List[int] = None
    attention_mask: List[int] = None
    token_embed_ids: List[int] = None

    def __post_init__(self):
        self.data = {}
        self.entity_map = {entity.uid: entity for entity in self.entities}
        self.entity_offset_map = {(entity.start, entity.end): entity
                                  for entity in self.entities}
        self.relation_map = {relation.uid: relation
                             for relation in self.relations}
        self.event_map = {event.uid: event for event in self.events}
        self.event_offset_map = {(event.trigger.start, event.trigger.end): event
                                 for event in self.events}

    @staticmethod
    def from_dict(dict_obj: Dict[str, Any], **kwargs):
        return Sentence(sent_id=dict_obj['sent_id'],
                        tokens=dict_obj['tokens'],
                        entities=[Entity.from_dict(x, **kwargs)
                                  for x in dict_obj['entity_mentions']],
                        relations=[Relation.from_dict(x, **kwargs)
                                   for x in dict_obj['relation_mentions']],
                        events=[Event.from_dict(x, **kwargs)
                                for x in dict_obj['event_mentions']],
                        pieces=dict_obj.get('pieces', None),
                        token_lens=dict_obj.get('token_lens', None),
                        token_embed_ids=dict_obj.get("token_embed_ids", None)
                        )

    @property
    def token_num(self):
        return len(self.tokens)

    def update_entities(self, entities: List[Entity]):
        self.entities = entities
        self.entity_map = {entity.uid: entity for entity in self.entities}
        self.entity_offset_map = {(entity.start, entity.end): entity
                                  for entity in self.entities}

    def update_relations(self, relations: List[Relation]):
        self.relations = relations
        self.relation_map = {relation.uid: relation
                             for relation in self.relations}

    def update_events(self, events: List[Event]):
        self.events = events
        self.event_map = {event.uid: event for event in self.events}
        self.event_offset_map = {(event.trigger.start, event.trigger.end): event
                                 for event in self.events}

    def has_entity(self, entity_id: str, mention_id: str):
        uid = '{}_{}'.format(entity_id, mention_id)
        return uid in self.entity_map

    def get_entity(self, entity_id: str, mention_id: str):
        uid = '{}_{}'.format(entity_id, mention_id)
        return self.entity_map.get(uid, None)

    def get_entity_by_offsets(self, start: int, end: int):
        return self.entity_offset_map.get((start, end), None)

    def get_event_by_offsets(self, start: int, end: int):
        return self.event_offset_map.get((start, end), None)

    def get_entity_type(self,
                        key: Union[int, str, Tuple[int, int]],
                        level: str = 'type') -> str:
        if type(key) is int:
            if key >= len(self.entities):
                return ''
            else:
                return self.entities[key].get_type(level)
        elif type(key) is str:
            entity = self.entity_map.get(key, None)
            if entity:
                return entity.get_type(level)
            else:
                return ''
        elif type(key) is tuple:
            entity = self.entity_offset_map.get(key, None)
            if entity:
                return entity.get_type(level)
            else:
                return ''
        else:
            raise ValueError('Entity key must be an int, str, or tuple')


@dataclass
class Document:
    doc_id: str
    sentences: List[Sentence]
    data: Dict[str, Any] = None

    def __post_init__(self):
        self.data = {}

    @staticmethod
    def from_dict(dict_obj: Dict[str, Any],
                  **kwargs):
        return Document(doc_id=dict_obj['doc_id'],
                        sentences=[Sentence.from_dict(dict_obj, **kwargs)],
                        )


class IEDataset(Dataset):
    def __init__(self,
                 path: str,
                 config: Dict[str, Any],
                 word_vocab=None,
                 ws_tokenizer=None,
                 ws_model=None):
        self.path = path
        self.config = config
        self.orig_data = []
        self.data = []
        self.tensors = []

        #self.word_embed = word_embed
        self.word_vocab = word_vocab

        self.ws_tokenizer = ws_tokenizer
        self.ws_model = ws_model

        self.load(path)

    def __len__(self) -> int:
        return len(self.tensors)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        return self.tensors[item]

    @property
    def entity_num(self) -> int:
        entity_num = 0
        for doc in self.data:
            for sent in doc.sentences:
                entity_num += len(sent.entities)
        return entity_num

    @property
    def relation_num(self) -> int:
        relation_num = 0
        for doc in self.data:
            for sent in doc.sentences:
                relation_num += len(sent.relations)
        return relation_num

    @property
    def event_num(self) -> int:
        event_num = 0
        for doc in self.data:
            for sent in doc.sentences:
                event_num += len(sent.events)
        return event_num

    @property
    def sentence_num(self) -> int:
        sent_num = sum(len(doc.sentences) for doc in self.data)
        return sent_num

    def load(self, path: str):
        with open(path) as r:
            for line in r:
                doc = Document.from_dict(json.loads(line),
                                        #  symmetric_relations={
                                            #  'Personal-Social',
                                        #  }
                                         )
                self.data.append(doc)
                doc2 = Document.from_dict(json.loads(line))
                self.orig_data.append(doc2)

        logger.info('File: {}'.format(path))
        logger.info('#Documents: {}'.format(len(self.data)))
        logger.info('#Sentences: {}'.format(self.sentence_num))
        logger.info('#Entity: {}'.format(self.entity_num))
        logger.info('#Relation: {}'.format(self.relation_num))
        logger.info('#Event: {}'.format(self.event_num))
        print('-' * 80)

    @property
    def entity_type_set(self) -> Set[str]:
        entity_type_level = self.config.get('entity_type_level', 'type')

        entity_type_set = set()
        for doc in self.data:
            for sent in doc.sentences:
                for entity in sent.entities:
                    entity_type = entity.get_type(entity_type_level)
                    entity_type_set.add(entity_type)
        return entity_type_set

    @property
    def relation_type_set(self) -> Set[str]:
        relation_type_level = self.config.get('relation_type_level', 'subtype')

        relation_type_set = set()
        for doc in self.data:
            for sent in doc.sentences:
                for relation in sent.relations:
                    if relation_type_level == 'type':
                        relation_type = relation.relation_type
                    elif relation_type_level == 'subtype':
                        relation_type = '{}.{}'.format(
                            relation.relation_type,
                            relation.relation_subtype)
                    elif relation_type_level == 'subsubtype':
                        relation_type = '{}.{}.{}'.format(
                            relation.relation_type,
                            relation.relation_subtype,
                            relation.relation_subsubtype)
                    else:
                        raise ValueError('Unknown type level: {}'.format(
                            relation_type_level))

                    relation_type_set.add(relation_type)
        return relation_type_set

    @property
    def event_type_set(self) -> Set[str]:
        event_type_level = self.config.get('event_type_level', 'subtype')

        event_type_set = set()
        for doc in self.data:
            for sent in doc.sentences:
                for event in sent.events:
                    if event_type_level == 'type':
                        event_type = event.event_type
                    elif event_type_level == 'subtype':
                        event_type = '{}.{}'.format(
                            event.event_type,
                            event.event_subtype)
                    elif event_type_level == 'subsubtype':
                        event_type = '{}.{}.{}'.foramt(
                            event.event_type,
                            event.event_subtype,
                            event.evnet_subsubtype)
                    else:
                        raise ValueError('Unknown type level: {}'.format(
                            event_type_level))
                    event_type_set.add(event_type)
        return event_type_set

    @property
    def role_set(self) -> Set[str]:
        arg_role_set = set()
        for doc in self.data:
            for sent in doc.sentences:
                for event in sent.events:
                    for arg in event.arguments:
                        arg_role_set.add(arg.role)
        return arg_role_set

    @property
    def mention_type_set(self) -> Set[str]:
        mention_type_set = set()
        for doc in self.data:
            for sent in doc.sentences:
                for entity in sent.entities:
                    mention_type_set.add(entity.mention_type)
        if self.config.get('remove_fillers', True):
            mention_type_set.discard('FILLER')
        return mention_type_set

    def process_entities(self,
                         sentence: Sentence,
                         remove_fillers: bool = True):
        entities = sentence.entities
        processed_entities = []
        for entity in entities:
            if remove_fillers and entity.mention_type == 'FILLER':
                # Skip fillers
                continue
            processed_entities.append(entity)

        sentence.update_entities(processed_entities)

    def process_relations(self,
                          sentence: Sentence):
        relations = sentence.relations
        processed_relations = []
        for relation in relations:
            if (sentence.has_entity(relation.arg1.entity_id,
                                    relation.arg1.mention_id) and
                    sentence.has_entity(relation.arg2.entity_id,
                                        relation.arg2.mention_id)):
                processed_relations.append(relation)

        sentence.update_relations(processed_relations)

    def process_events(self,
                       sentence: Sentence):
        events = sentence.events
        processed_events = []
        for event in events:
            arguments = [arg for arg in event.arguments
                         if sentence.has_entity(arg.entity_id, arg.mention_id)]
            event.arguments = arguments
            processed_events.append(event)
        sentence.update_events(processed_events)

    def process_tokens(self,
                       sentence,
                       tokenizer: PreTrainedTokenizer,
                       max_sent_len: int = 128,
                       swap_prob=0.05):
        tokens = sentence.tokens
        if self.ws_model is not None and swap_prob > 0:
            if len(tokens) > 50 or random() < 0.2:
                #st = randrange(0, max(1, len(tokens) - 500))
                #run on each 500-token segment
                for st in range(0, len(tokens), 500):
                    end = st + 500
                    end = min(len(tokens), end)

                    tokens_orig = tokens[st:end]

                    #print(tokens_orig)

                    tokens_aug = augment(tokens_orig, swap_prob, self.ws_tokenizer, self.ws_model)

                    #print(tokens_aug)

                    tokens[st:end] = tokens_aug
        #print(tokens)
        pieces = []
        for i, word in enumerate(tokens):
            if word == '\n':
                pieces.append(tokenizer.tokenize("a\n")[-1])
                continue
            if i > 0 and not word in string.punctuation and not tokens[i - 1] in ['-', '\'', '(']:
                word = " " + word
            wp = tokenizer.tokenize(word)
            pieces.append(wp)
        #pieces = [tokenizer.tokenize(" " * (i > 0 and not (token in [".", ","])) + token) for i, token in enumerate(tokens)]
        token_lens = [len(p) for p in pieces]
        # Remove overlength sentences
        if sum(token_lens) > max_sent_len:
            print("skipped due to length:", sum(token_lens))
            return False
        # Todo: automatically remove 0-len tokens
        assert all(l > 0 for l in token_lens)
        # Flatten word pieces
        pieces = [p for ps in pieces for p in ps]
        sentence.pieces = pieces
        sentence.token_lens = token_lens
        sentence.piece_idxs = tokenizer.encode(pieces,
                                               add_special_tokens=True,
                                               max_length=max_sent_len,
                                               truncation=True)
        sentence.attention_mask = [1.0] * len(sentence.piece_idxs)
        #token_lens[0] += 1  # for <s>
        #token_lens[-1] += 1  # for </s>
        #accounted for later
        if self.word_vocab:
            sentence.token_embed_ids = [self.word_vocab.get(tok.lower(), 0) for tok in tokens]
        return True

    def process_doc(self,
                    doc: Document,
                    tokenizer: PreTrainedTokenizer,
                    max_sent_len: int = 128,
                    swap_prob = 0.05):
        # Convert tokens to wordpieces
        sentences = []
        for sentence in doc.sentences:
            keep_sent = self.process_tokens(sentence, tokenizer, max_sent_len, swap_prob)
            if not keep_sent:
                continue
            # print('Before', len(sentence.entities), len(sentence.relations), len(sentence.events))
            self.process_entities(sentence)
            self.process_relations(sentence)
            self.process_events(sentence)
            # print('After', len(sentence.entities), len(sentence.relations), len(sentence.events))
            # Add the processed sentence
            sentences.append(sentence)
        doc.sentences = sentences

    def process(self,
                tokenizer: PreTrainedTokenizer,
                max_sent_len: int = 128,
                swap_prob = 0):

        for doc_id, _ in enumerate(tqdm(self.data)):
            #print("processing doc", doc_id + 1, "out of", len(self.data))
            self.data[doc_id] = copy.deepcopy(self.orig_data[doc_id])
            self.process_doc(self.data[doc_id], tokenizer, max_sent_len, swap_prob)

    def tensorize_pieces(self, doc: Document, gpu: bool = True):
        all_pieces = [sent.piece_idxs for sent in doc.sentences]
        max_piece_num = max(len(x) for x in all_pieces)
        assert max_piece_num > 0
        # Padding wordpiece indices
        all_pieces = [x + [0] * (max_piece_num - len(x)) for x in all_pieces]
        # Convert to tensor
        if gpu:
            all_pieces = torch.cuda.LongTensor(all_pieces)
        else:
            all_pieces = torch.LongTensor(all_pieces)
        return all_pieces

    def tensorize_sentence(self,
                           sentence: Sentence,
                           vocabs,
                           ontology,
                           max_entity_len: int,
                           max_trigger_len: int
                           ):
        # Vocabularies
        entity_type_level = self.config.get('entity_type_level', 'type')
        event_type_level = self.config.get('event_type_level', 'subtype')
        relation_type_level = self.config.get('relation_type_level', 'subtype')
        entity_type_stoi = vocabs['entity']
        mention_type_stoi = vocabs['mention']
        event_type_stoi = vocabs['event']
        relation_type_stoi = vocabs['relation']
        role_stoi = vocabs['role']

        instance = {'sentence': sentence}
        instance['graph'] = self.inst2graph(sentence.entities,
                                            sentence.events,
                                            sentence.relations,
                                            self.config)
        token_num = sentence.token_num
        instance['sent_id'] = sentence.sent_id
        instance['tokens'] = sentence.tokens
        instance['token_num'] = token_num
        instance['token_lens'] = sentence.token_lens
        instance['token_embed_ids'] = sentence.token_embed_ids
        # tensorize tokens
        instance['pieces'] = sentence.piece_idxs
        instance['pieces_text'] = sentence.pieces
        instance['piece_num'] = len(sentence.piece_idxs)
        # Filter entities
        entities = [entity for entity in sentence.entities
                    if (entity.end - entity.start) <= max_entity_len]
        sentence.update_entities(entities)
        # Filter relations
        relations = []
        for relation in sentence.relations:
            if (sentence.has_entity(relation.arg1.entity_id,
                                    relation.arg1.mention_id) and
                sentence.has_entity(relation.arg2.entity_id,
                                    relation.arg2.mention_id)):
                relations.append(relation)
        sentence.update_relations(relations)
        # Filter events
        events = []
        for event in sentence.events:
            if event.trigger.end - event.trigger.start > max_trigger_len:
                continue
            event.arguments = [arg for arg in event.arguments
                               if sentence.has_entity(arg.entity_id,
                                                      arg.mention_id)]
            events.append(event)
        sentence.update_events(events)

        # labels
        instance['entity_labels'] = {
            (x.start, x.end): entity_type_stoi[x.get_type(entity_type_level)]
            for x in sentence.entities
        }
        instance['entity_uids'] = {(x.start, x.end): x.uid
                                   for x in sentence.entities}
        """instance['mention_labels'] = {
            (x.start, x.end): mention_type_stoi[x.mention_type]
            for x in sentence.entities
        }"""
        instance['event_labels'] = {
            (x.start, x.end): event_type_stoi[x.get_type(event_type_level)]
            for x in sentence.events
        }
        instance['event_uids'] = {(x.start, x.end): x.uid
                                  for x in sentence.events}

        instance['entity_num'] = len(sentence.entities)
        instance['event_num'] = len(sentence.events)
        instance['relation_num'] = len(sentence.relations)
        for relation in sentence.relations:
            relation_type = relation.get_type(relation_type_level)
            relation.relation_type_idx = relation_type_stoi[relation_type]
            #relation.is_symmetric = ontology.is_symmetric(relation_type)

            relation_type_rev = '{}_rev'.format(relation_type)
            if relation_type_rev in relation_type_stoi:
                relation.relation_type_idx_rev = relation_type_stoi[relation_type_rev]
        for event in sentence.events:
            for arg in event.arguments:
                arg.role_idx = role_stoi[arg.role]
        instance['relations'] = sentence.relations
        instance['entities'] = sentence.entities
        instance['events'] = sentence.events

        return instance

    def tensorize(self, vocabs: Dict[str, Dict[str, int]], config, ontology=None):
        self.vocabs = vocabs
        self.tensors = []
        for doc in self.data:
            self.tensors.extend(self.tensorize_sentence(sent,
                                                        vocabs,
                                                        ontology,
                                                        config.get('max_entity_len', 10),
                                                        config.get('max_trigger_len', 2))
                                for sent in doc.sentences)

    def get_relation_labels_for_clusters(self, relations, mention_to_ent, entity_uids):
        relation_type_level = self.config.get('relation_type_level', 'subtype')

        cluster_num = max(mention_to_ent) + 1

        labels = [[0] * cluster_num for i in range(cluster_num)]

        relations_cl_set = set()

        for rel_id, relation in enumerate(relations):
            relation_type = relation.relation_type_idx
            relation_type_rev = relation.relation_type_idx_rev
            arg1 = relation.arg1.uid
            arg2 = relation.arg2.uid
            if arg1 == arg2:
                continue
            i = entity_uids.index(arg1)
            j = entity_uids.index(arg2)
            c_i = mention_to_ent[i]
            c_j = mention_to_ent[j]

            labels[c_i][c_j] = labels[c_j][c_i] = relation_type

            if c_i > c_j:
                c_i, c_j = c_j, c_i

            relations_cl_set.add((c_i, c_j, relation.get_type(relation_type_level)))

        relations_cl = list(relations_cl_set)

        return labels, relations_cl

    def get_relation_labels(self, relations, entities, graph, entity_uids):
        #also adds missing relations to rels list
        entity_num = len(entities)
        labels = [[0] * entity_num for i in range(entity_num)]
        rel_set = set([(i[0], i[1]) for i in graph.relations])
        for rel_id, relation in enumerate(relations):
            relation_type = relation.relation_type_idx
            relation_type_rev = relation.relation_type_idx_rev
            arg1 = relation.arg1.uid
            arg2 = relation.arg2.uid
            if arg1 == arg2:
                continue
            i = entity_uids.index(arg1)
            j = entity_uids.index(arg2)
            labels[i][j] = labels[j][i] = relation_type
            """arg1 = relation.arg1.entity_id
            arg2 = relation.arg2.entity_id
            if arg1 == arg2:
                continue
            #arg1 = entity_uids.index(arg1)
            #arg2 = entity_uids.index(arg2)
            for i in range(entity_num):
                for j in range(entity_num):
                    if entities[i].entity_id == arg1 and entities[j].entity_id == arg2:
                        labels[i][j] = relation_type
                        labels[j][i] = relation_type
                        if (i, j) not in rel_set and (j, i) not in rel_set:
                            rel_set.add((i, j))
                            graph.relations.append((i, j, graph.relations[rel_id][2]))
                    if entities[j].entity_id == arg1 and entities[i].entity_id == arg2:
                        labels[i][j] = relation_type
                        labels[j][i] = relation_type
                        if (i, j) not in rel_set and (j, i) not in rel_set:
                            rel_set.add((i, j))
                            graph.relations.append((i, j, graph.relations[rel_id][2]))"""
            """if arg1 > arg2:
                labels[arg1 * (entity_num - 1) + arg2] = relation_type
                # Reverse link
                labels[arg2 * (entity_num - 1) + arg1 - 1] = relation_type
            else:
                labels[arg1 * (entity_num - 1) + arg2 - 1] = relation_type
                # Reverse link
                labels[arg2 * (entity_num - 1) + arg1] = relation_type"""
        return labels

    def get_role_labels(self,
                        events,
                        event_uids,
                        entity_uids):
        event_num, entity_num = len(event_uids), len(entity_uids)
        labels = [0] * event_num * entity_num
        for event in events:
            event_idx = event_uids.index(event.uid)
            for arg in event.arguments:
                role = arg.role_idx
                arg_idx = entity_uids.index(arg.uid)
                labels[event_idx * entity_num + arg_idx] = role
        return labels

    def inst2graph(self, entities, events, relations, config):
        entity_type_level = config.get('entity_type_level', 'type')
        event_type_level = config.get('event_type_level', 'subtype')
        relation_type_level = config.get('relation_type_level', 'subtype')

        entities_ = [(entity.start, entity.end, entity.get_type(entity_type_level))
                    for entity in entities]
        entity_uids = {entity.uid: idx
                       for idx, entity in enumerate(entities)}
        triggers = [(event.trigger.start, event.trigger.end,
                     event.get_type(event_type_level))
                    for event in events]
        event_uids = {event.uid: idx
                      for idx, event in enumerate(events)}

        relations_ = []
        for relation in relations:
            arg1 = relation.arg1.uid
            arg2 = relation.arg2.uid
            relations_.append((entity_uids[arg1],
                               entity_uids[arg2],
                               relation.get_type(relation_type_level)))

        roles = []
        for event in events:
            trigger = event_uids[event.uid]
            for argument in event.arguments:
                entity = entity_uids[argument.uid]
                roles.append((trigger, entity, argument.role))
        return Graph(entities_, triggers, relations_, roles)

    def collate_fn(self,
                   batch: List[Dict[str, Any]],) -> Batch:
        config = self.config
        max_entity_len = config.get('max_entity_len')
        max_trigger_len = config.get('max_trigger_len')

        # Pad wordpiece indices
        max_piece_num = max_token_num = 0
        max_entity_num = max_event_num = 1
        for inst in batch:
            if inst['piece_num'] > max_piece_num:
                max_piece_num = inst['piece_num']
            if inst['token_num'] > max_token_num:
                max_token_num = inst['token_num']
            if inst['entity_num'] > max_entity_num:
                max_entity_num = inst['entity_num']
            if inst['event_num'] > max_event_num:
                max_event_num = inst['event_num']

        pieces, attention_mask, token_lens = [], [], []

        # Index and mask tensors to generate span representations
        (entity_idxs,
         entity_mask,
         entity_offsets,
         entity_lens,
         entity_boundaries
         ) = enumerate_spans(max_token_num, max_entity_len)
        (trigger_idxs,
         trigger_mask,
         trigger_offsets,
         trigger_lens,
         trigger_boundaries
         ) = enumerate_spans(max_token_num, max_trigger_len)
        entity_labels, mention_labels, trigger_labels = [], [], []
        id_entity_labels, id_mention_labels, id_trigger_labels = [], [], []

        # Entity spans
        pos_entity_idxs, pos_trigger_idxs = [], []
        pos_entity_offsets, pos_trigger_offsets = [], []

        # Relation and arg role labels
        relation_labels, role_labels = [], []

        entity_labels_sep, trigger_labels_sep = [], []
        relation_labels_sep, role_labels_sep = [], []

        entity_span_mask, trigger_span_mask = [], []

        tokens = []
        graphs = []

        sent_ids = []

        pieces_text = []

        token_embed_ids = []

        coref_labels = []
        entities_coref = []
        mention_to_ent_coref = []

        is_start = []
        len_from_here = []
        type_from_here = []

        for inst in batch:
            # graphs.append(self.inst2graph(inst))
            graphs.append(inst['graph'])
            sent_ids.append(inst['sent_id'])
            tokens.append(inst['tokens'])

            if inst['token_embed_ids']:
                token_embed_ids.append(inst['token_embed_ids'] + [0] * (max_token_num - inst['token_num']))

            # Inputs
            piece_pad_num = max_piece_num - inst['piece_num']
            pieces.append(inst['pieces'] + [0] * piece_pad_num)
            attention_mask.append([1] * inst['piece_num'] + [0] * piece_pad_num)
            token_lens.append(inst['token_lens'])
            pieces_text.append(inst['pieces_text'])

            # TODO: check offsets and labels
            # Entity labels
            inst_pos_entity_idxs = []
            inst_pos_entity_offsets = []
            inst_pos_entity_uids = []
            inst_entity_span_mask = []
            inst_neg_entity_idxs = []
            inst_neg_entity_offsets = []
            inst_entity_labels_sep, inst_trigger_labels_sep = [], []
            int_relation_labels_sep, inst_role_labels_sep = [], []

            inst_is_start = [0 for i in range(inst['token_num'])]
            inst_len_from_here = [[0 for j in range(max_entity_len)] for i in range(inst['token_num'])]
            inst_type_from_here = [0 for i in range(inst['token_num'])]

            for offset_idx, (start, end) in enumerate(entity_offsets):
                if end > inst['token_num']:
                    entity_labels.append(-100)
                    mention_labels.append(-100)
                    inst_entity_span_mask.append(0)
                else:
                    entity_labels.append(
                        inst['entity_labels'].get((start, end), 0))
                    mention_labels.append(-100)
                        #inst['mention_labels'].get((start, end), -100))
                    if (start, end) in inst['entity_labels']:
                        inst_pos_entity_idxs.append(offset_idx)
                        inst_pos_entity_offsets.append((start, end))
                        inst_pos_entity_uids.append(
                            inst['entity_uids'][(start, end)])
                        id_entity_labels.append(inst['entity_labels'][(start, end)])
                        id_mention_labels.append(-100)#inst['mention_labels'][(start, end)])
                        # Gold labels to GNN
                        inst_entity_labels_sep.append(inst['entity_labels'][(start, end)])

                        """overlaps_longer = False
                        for j in range(start + 1, end):
                            if inst_is_start[j]:
                                if inst_len_from_here[j] > end - start:
                                    overlaps_longer = True
                                else:
                                    inst_is_start[j] = 0
                                    inst_len_from_here[j] = 0
                                    inst_type_from_here[j] = 0
                        if not overlaps_longer:"""
                        inst_is_start[start] = 1
                        inst_len_from_here[start][end - start] = 1
                        inst_type_from_here[start] = inst['entity_labels'][(start, end)]

                    else:
                        inst_neg_entity_idxs.append(offset_idx)
                        inst_neg_entity_offsets.append((start, end))
                    inst_entity_span_mask.append(1)
            # TODO: overlength entity causes the following issue
            # if len(inst_pos_entity_offsets) != inst['entity_num']:
            #     print(inst)

            is_start.append(inst_is_start)
            len_from_here.append(inst_len_from_here)
            type_from_here.append(inst_type_from_here)

            inst_pos_entity_idxs += [0] * (max_entity_num - inst['entity_num'])
            pos_entity_idxs.append(inst_pos_entity_idxs)
            pos_entity_offsets.append(inst_pos_entity_offsets)
            entity_span_mask.append(inst_entity_span_mask)
            entity_labels_sep.append(inst_entity_labels_sep)

            # Event labels
            inst_pos_trigger_idxs = []
            inst_pos_trigger_offsets = []
            inst_pos_event_uids = []
            inst_trigger_span_mask = []
            for offset_idx, (start, end) in enumerate(trigger_offsets):
                if end > inst['token_num']:
                    trigger_labels.append(-100)
                    inst_trigger_span_mask.append(0)
                else:
                    trigger_labels.append(
                        inst['event_labels'].get((start, end), 0))
                    if (start, end) in inst['event_labels']:
                        inst_pos_trigger_idxs.append(offset_idx)
                        inst_pos_trigger_offsets.append((start, end))
                        inst_pos_event_uids.append(
                            inst['event_uids'][(start, end)])
                        id_trigger_labels.append(inst['event_labels'][(start, end)])
                        # Gold labels to GNN
                        inst_trigger_labels_sep.append(inst['event_labels'][(start, end)])
                    inst_trigger_span_mask.append(1)
            inst_pos_trigger_idxs += [0] * (max_event_num - inst['event_num'])
            pos_trigger_idxs.append(inst_pos_trigger_idxs)
            pos_trigger_offsets.append(inst_pos_trigger_offsets)
            trigger_span_mask.append(inst_trigger_span_mask)
            trigger_labels_sep.append(inst_trigger_labels_sep)

            # Relation labels


            entity_num = len(inst['entities'])

            inst_entities_coref = dict()
            inst_mention_to_ent_coref = []

            inst_coref_labels = [[0] * (entity_num) for i in range(entity_num)]
            for a in range(entity_num):
                cur_ent = inst['entities'][a].entity_id
                if cur_ent not in inst_entities_coref:
                    inst_entities_coref[cur_ent] = len(inst_entities_coref)
                inst_mention_to_ent_coref.append(inst_entities_coref[cur_ent])

                inst_coref_labels[a][a] = 1

                for b in range(a + 1, entity_num):
                    if cur_ent == inst['entities'][b].entity_id:
                        inst_coref_labels[b][a] = 1
                        # Reverse link
                        inst_coref_labels[a][b] = 1

            coref_labels.append(inst_coref_labels)
            entities_coref.append(inst_entities_coref)
            mention_to_ent_coref.append(inst_mention_to_ent_coref)

            inst['graph'].coref_matrix = inst_coref_labels
            inst['graph'].cluster_labels = inst_mention_to_ent_coref

            inst_relation_labels_cl, relations_cl = self.get_relation_labels_for_clusters(inst['relations'],
                                                                          inst_mention_to_ent_coref,
                                                                          inst_pos_entity_uids)

            inst['graph'].relations = relations_cl



            """inst_relation_labels = self.get_relation_labels(inst['relations'],
                                                                        inst['entities'],
                                                                        inst['graph'],
                                                                        inst_pos_entity_uids)"""

            # max_entity_num)
            relation_labels_sep.append(inst_relation_labels_cl)

            relation_labels.append(inst_relation_labels_cl)


            # Role labels
            inst_role_labels = self.get_role_labels(inst['events'],
                                                    inst_pos_event_uids,
                                                    inst_pos_entity_uids)
                                                    # max_event_num,
                                                    # max_entity_num)
            role_labels_sep.append(inst_role_labels)
            # for x in inst_role_labels:
            #     role_labels.extend(x)
            role_labels.extend(inst_role_labels)

        if self.config.get('gpu', True):
            pieces = torch.cuda.LongTensor(pieces)
            attention_mask = torch.cuda.FloatTensor(attention_mask)
            token_embed_ids = torch.cuda.LongTensor(token_embed_ids)

            entity_idxs = torch.cuda.LongTensor([entity_idxs])
            entity_mask = torch.cuda.FloatTensor([entity_mask])
            entity_boundaries = torch.cuda.LongTensor([entity_boundaries])
            trigger_idxs = torch.cuda.LongTensor([trigger_idxs])
            trigger_mask = torch.cuda.FloatTensor([trigger_mask])
            trigger_boundaries = torch.cuda.LongTensor([trigger_boundaries])

            entity_labels = torch.cuda.LongTensor(entity_labels)
            mention_labels = torch.cuda.LongTensor(mention_labels)
            trigger_labels = torch.cuda.LongTensor(trigger_labels)

            id_entity_labels = torch.cuda.LongTensor(id_entity_labels)
            id_mention_labels = torch.cuda.LongTensor(id_mention_labels)
            id_trigger_labels = torch.cuda.LongTensor(id_trigger_labels)

            pos_entity_idxs = torch.cuda.LongTensor(pos_entity_idxs)
            pos_trigger_idxs = torch.cuda.LongTensor(pos_trigger_idxs)

            relation_labels = torch.cuda.LongTensor(relation_labels)
            role_labels = torch.cuda.LongTensor(role_labels)

            coref_labels = torch.cuda.LongTensor(coref_labels)

            entity_span_mask = torch.cuda.LongTensor(entity_span_mask)
            trigger_span_mask = torch.cuda.LongTensor(trigger_span_mask)

            entity_lens = torch.cuda.FloatTensor(entity_lens)
            trigger_lens = torch.cuda.FloatTensor(trigger_lens)

            entity_labels_sep = [torch.cuda.LongTensor(x)
                                 for x in entity_labels_sep]
            trigger_labels_sep = [torch.cuda.LongTensor(x)
                                  for x in trigger_labels_sep]
            relation_labels_sep = [torch.cuda.LongTensor(x)
                                   for x in relation_labels_sep]
            role_labels_sep = [torch.cuda.LongTensor(x)
                               for x in role_labels_sep]

            mention_to_ent_coref = torch.cuda.LongTensor(mention_to_ent_coref)

            is_start = torch.cuda.LongTensor(is_start)
            len_from_here = torch.cuda.LongTensor(len_from_here)
            type_from_here = torch.cuda.LongTensor(type_from_here)
        else:
            pieces = torch.LongTensor(pieces)
            attention_mask = torch.FloatTensor(attention_mask)
            token_embed_ids = torch.LongTensor(token_embed_ids)

            entity_idxs = torch.LongTensor([entity_idxs])
            entity_mask = torch.FloatTensor([entity_mask])
            entity_boundaries = torch.LongTensor([entity_boundaries])
            trigger_idxs = torch.LongTensor([trigger_idxs])
            trigger_mask = torch.FloatTensor([trigger_mask])
            trigger_boundaries = torch.LongTensor([trigger_boundaries])

            entity_labels = torch.LongTensor(entity_labels)
            mention_labels = torch.LongTensor(mention_labels)
            trigger_labels = torch.LongTensor(trigger_labels)

            id_entity_labels = torch.LongTensor(id_entity_labels)
            id_mention_labels = torch.LongTensor(id_mention_labels)
            id_trigger_labels = torch.LongTensor(id_trigger_labels)

            pos_entity_idxs = torch.LongTensor(pos_entity_idxs)
            pos_trigger_idxs = torch.LongTensor(pos_trigger_idxs)

            relation_labels = torch.LongTensor(relation_labels)
            role_labels = torch.LongTensor(relation_labels)

            entity_span_mask = torch.LongTensor(entity_span_mask)
            trigger_span_mask = torch.LongTensor(trigger_span_mask)

            entity_lens = torch.FloatTensor(entity_lens)
            trigger_lens = torch.FloatTensor(trigger_lens)

            entity_labels_sep = [torch.LongTensor(x)
                                 for x in entity_labels_sep]
            trigger_labels_sep = [torch.LongTensor(x)
                                  for x in trigger_labels_sep]
            relation_labels_sep = [torch.LongTensor(x)
                                   for x in relation_labels_sep]
            role_labels_sep = [torch.LongTensor(x)
                               for x in role_labels_sep]
        return Batch(
            pieces=pieces,
            token_lens=token_lens,
            attention_mask=attention_mask,
            entity_idxs=entity_idxs,
            entity_mask=entity_mask,
            entity_boundaries=entity_boundaries,
            trigger_idxs=trigger_idxs,
            trigger_mask=trigger_mask,
            trigger_boundaries=trigger_boundaries,
            entity_labels=entity_labels,
            mention_labels=mention_labels,
            trigger_labels=trigger_labels,
            id_entity_labels=id_entity_labels,
            id_mention_labels=id_mention_labels,
            id_trigger_labels=id_trigger_labels,
            max_entity_len=max_entity_len,
            max_trigger_len=max_trigger_len,
            pos_entity_idxs=pos_entity_idxs,
            pos_trigger_idxs=pos_trigger_idxs,
            pos_entity_offsets=pos_entity_offsets,
            pos_trigger_offsets=pos_trigger_offsets,
            relation_labels=relation_labels,
            role_labels=role_labels,
            entity_span_mask=entity_span_mask,
            trigger_span_mask=trigger_span_mask,
            entity_offsets=entity_offsets,
            trigger_offsets=trigger_offsets,
            tokens=tokens,
            entity_lens=entity_lens,
            trigger_lens=trigger_lens,
            entity_labels_sep=entity_labels_sep,
            trigger_labels_sep=trigger_labels_sep,
            relation_labels_sep=relation_labels_sep,
            role_labels_sep=role_labels_sep,
            graphs=graphs,
            sent_ids=sent_ids,
            pieces_text=pieces_text,
            token_embed_ids=token_embed_ids,
            coref_labels=coref_labels,
            entities_coref=entities_coref,
            mention_to_ent_coref=mention_to_ent_coref,
            is_start=is_start,
            len_from_here=len_from_here,
            type_from_here=type_from_here
        )
