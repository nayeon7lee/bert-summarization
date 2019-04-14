from __future__ import unicode_literals

import pickle
import torch
import torch.utils.data as data
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

import numpy as np
import argparse
import collections
import logging
import json
import re

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel

import sys
sys.path.insert(0, '/home/nayeon/naver2')

# DailyMail 
import glob
import random
import struct
import csv
from tensorflow.core.example import example_pb2
from tqdm import tqdm
from utils import config

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', 
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

'BERT'
class InputExample(object):

    def __init__(self, unique_id, text_a, text_b):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def abstract2sents(abstract):
  cur = 0
  sents = []
  while True:
    try:
      start_p = abstract.index(config.SENTENCE_START, cur)
      end_p = abstract.index(config.SENTENCE_END, start_p + 1)
      cur = end_p + len(config.SENTENCE_END)
      sents.append(abstract[start_p+len(config.SENTENCE_START):end_p])
    except ValueError as e: # no more sentences
      return sents

def load_examples(is_small):
    articles = {'train':[],'val':[],'test':[]}
    summaries = {'train':[],'val':[],'test':[]}
    if not is_small:
        article_path = '{}/articles_{}.pickle'
        summary_path = '{}/summaries_{}.pickle'
    else:
        article_path = '{}/articles_small_{}.pickle'
        summary_path = '{}/summaries_small_{}.pickle'

    for k in articles:
        with open(article_path.format(config.train_data_path, k), 'rb') as handle:
            articles[k] = pickle.load(handle)
        with open(summary_path.format(config.train_data_path, k), 'rb') as handle:
            summaries[k] = pickle.load(handle)

    return (articles['train'], summaries['train']), (articles['val'], summaries['val']), (articles['test'], summaries['test'])

def create_examples(data_path):
    articles, summaries = {'train':[],'val':[],'test':[]},{'train':[],'val':[],'test':[]}
    unique_id = 0

    while True:
        filelist = glob.glob(data_path) # get the list of datafiles
        assert filelist, ('Error: Empty filelist at %s' % data_path) # check filelist isn't empty
        filelist = sorted(filelist)
        for f in tqdm(filelist):
            split = 'train' if 'train' in f else 'val' if 'val' in f else 'test'

            reader = open(f, 'rb')
            while True:
                len_bytes = reader.read(8)
                if not len_bytes: break # finished reading this file
                str_len = struct.unpack('q', len_bytes)[0]
                example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
                e = example_pb2.Example.FromString(example_str) 

                # the article text was saved under the key 'article' in the data files
                article_text = e.features.feature['article'].bytes_list.value[0]
                # the abstract text was saved under the key 'abstract' in the data files
                abstract_text = e.features.feature['abstract'].bytes_list.value[0]
                # Use the <s> and </s> tags in abstract to get a list of sentences.
                abstract_text = ' '.join([sent.decode('unicode_escape').strip() for sent in abstract2sents(abstract_text)])

                articles[split].append(
                  InputExample(unique_id=unique_id, text_a=article_text.decode('unicode_escape').strip(), text_b=None))
                summaries[split].append(
                  InputExample(unique_id=unique_id, text_a=abstract_text, text_b=None))
                unique_id += 1
        
        for k in articles:
            print("Saving full datafiles")
            with open('{}/articles_{}.pickle'.format(config.train_data_path, k), 'wb') as handle:
                pickle.dump(articles[k], handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open('{}/summaries_{}.pickle'.format(config.train_data_path, k), 'wb') as handle:
                pickle.dump(summaries[k], handle, protocol=pickle.HIGHEST_PROTOCOL)

            print("Saving smaller version for debugging")
            size = int(len(articles[k])*0.1)
            with open('{}/articles_small_{}.pickle'.format(config.train_data_path, k), 'wb') as handle:
                pickle.dump(articles[k][:size], handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open('{}/summaries_small_{}.pickle'.format(config.train_data_path, k), 'wb') as handle:
                pickle.dump(summaries[k][:size], handle, protocol=pickle.HIGHEST_PROTOCOL)

        print("Completed reading all datafiles. No more data.")
        break

'Dataset'
class Dataset(data.Dataset):
    'Custom data.Dataset compatible with data.DataLoader.'
    def __init__(self, tokenizer, articles, summaries):
        """Reads news data from CNN/Daily Mail data files."""
        self.articles = articles
        self.summaries = summaries
        self.total_num = len(self.summaries)

        self.bert_tokenizer = tokenizer

    def __getitem__(self, idx):
        item = {}

        # bert_feature = self.preprocess(self.articles[idx], is_bert=True) # BERT input features
        # input_ids = torch.tensor(bert_feature.input_ids, dtype=torch.long)
        # input_mask = torch.tensor(bert_feature.input_mask, dtype=torch.long)
        # example_index = torch.arange(input_ids.size(0), dtype=torch.long)
        # item['input_feature']={"input_ids":input_ids, "input_mask":input_mask, "example_index":example_index}
        
        item['input_feature']=self.preprocess(self.articles[idx], is_bert=True) # BERT input features  
        item['target_feature']=self.preprocess(self.summaries[idx], is_bert=True) # BERT target feature
        # item['target_feature']=self.preprocess(self.summaries[idx], is_bert=False) # WordPiece features only
        item['target_txt']=self.summaries[idx].text_a

        if config.pointer_gen:
            # TODO: NEED TO UPDATE 
            item["input_ext_vocab_batch"], item["article_oovs"] = self.process_input(item["input_txt"])
            item["target_ext_vocab_batch"] = self.process_target(item["target_txt"], item["article_oovs"])
            item['target_ptr'], item['target_gate'] = self.create_ptr_and_gate(item["input_batch"],item["target_batch"],item["input_txt"],item["target_txt"])
        return item

    def preprocess(self, example, seq_length=config.max_seq_length, is_bert=False):
        tokens_a = self.bert_tokenizer.tokenize(example.text_a)
        tokens_b = None
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > seq_length - 2:
            tokens_a = tokens_a[0:(seq_length - 2)]

        tokens = [] # equals raw text tokens 
        input_type_ids = [] # equals segments_ids
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)

        input_ids = self.bert_tokenizer.convert_tokens_to_ids(tokens) # WordPiece embedding rep

        if is_bert:
            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < seq_length:
                input_ids.append(0)
                input_mask.append(0)
                input_type_ids.append(0)

            return InputFeatures(
                    unique_id=example.unique_id,
                    tokens=tokens, # raw text tokens
                    input_ids=input_ids, # WordPiece tokens
                    input_mask=input_mask, # mask tokens for later
                    input_type_ids=input_type_ids) # segments_ids
        else:
            while len(input_ids) < seq_length:
                input_ids.append(0)
            return input_ids # WordPiece represetnation

    def __len__(self):
        return self.total_num

def collate_fn(data):
    def merge(sequences):
        lengths = [len(seq) for seq in sequences]
        padded_seqs = torch.ones(len(sequences), max(lengths)).long()
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs, lengths 

    def merge_gate(sequences):
        lengths = [len(seq) for seq in sequences]
        padded_seqs = torch.zeros(len(sequences), max(lengths), dtype=torch.float)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs, lengths 

    def merge_ptr(sequences):
        lengths_row = [seq.size(0) for seq in sequences]
        lengths_clm = [seq.size(1) for seq in sequences]
        # a = np.empty([len(sequences), max(lengths_row) ,max(lengths_clm)])
        # a.fill(2)
        # padded_seqs = torch.LongTensor(a)
        padded_seqs = torch.zeros(len(sequences), max(lengths_row), max(lengths_clm), dtype=torch.float)
        for i, seq in enumerate(sequences):
            ## seq is a matrix
            for j, row in enumerate(seq):
                end = lengths_clm[i]        
                padded_seqs[i,j, :end] = row[:end]
        return padded_seqs, None 

    # data.sort(key=lambda x: len(x["input_feature"]), reverse=True) ## sort by source seq
    item_info = {}
    for key in data[0].keys():
        item_info[key] = [d[key] for d in data]

    # input_batch, input_lengths = merge(item_info['input_feature'])
    input_features = item_info['input_feature']
    input_ids_batch = torch.tensor([f.input_ids for f in input_features], dtype=torch.long)
    input_mask_batch = torch.tensor([f.input_mask for f in input_features], dtype=torch.long)
    example_index_batch = torch.zeros(input_ids_batch.size(), dtype=torch.long)
    # example_index_batch = torch.arange(input_ids_batch.size(0), dtype=torch.long)

    target_feature = item_info['target_feature']
    target_batch = torch.tensor([f.input_ids for f in target_feature], dtype=torch.long)
    # input_ids_batch == target_batch
    target_mask_batch = torch.tensor([f.input_mask for f in target_feature], dtype=torch.long)
    target_index_batch = torch.zeros(target_batch.size(), dtype=torch.long)

    target_batch = target_batch.transpose(0, 1)
    
    if config.USE_CUDA:
        input_ids_batch = input_ids_batch.cuda()
        input_mask_batch = input_mask_batch.cuda()
        example_index_batch = example_index_batch.cuda()
        target_batch = target_batch.cuda()
        target_mask_batch = target_mask_batch.cuda()
        target_index_batch = target_index_batch.cuda()

    d = {}
    d["input_ids_batch"]=input_ids_batch
    d["input_mask_batch"]=input_mask_batch
    d["example_index_batch"]=example_index_batch

    d["target_batch"] = target_batch
    d["target_mask_batch"] = target_mask_batch
    d["target_index_batch"] = target_index_batch
    d["target_txt"] = item_info["target_txt"]

    # pointer network
    if config.pointer_gen and 'input_ext_vocab_batch' in item_info:
        target_gate, target_gate_lenghts = merge_gate(item_info['target_gate'])
        target_ptr, target_ptr_lenghts   = merge_ptr(item_info['target_ptr'])

        input_ext_vocab_batch, _ = merge(item_info['input_ext_vocab_batch'])
        target_ext_vocab_batch, _ = merge(item_info['target_ext_vocab_batch'])
        input_ext_vocab_batch = input_ext_vocab_batch.transpose(0, 1)
        target_ext_vocab_batch = target_ext_vocab_batch.transpose(0, 1)
        if config.USE_CUDA:
            target_gate = target_gate.cuda()
            target_ptr = target_ptr.cuda()
            input_ext_vocab_batch = input_ext_vocab_batch.cuda()
            target_ext_vocab_batch = target_ext_vocab_batch.cuda()

        d["target_gate"] = target_gate
        d["target_ptr"] = target_ptr
        d["input_ext_vocab_batch"] = input_ext_vocab_batch
        d["target_ext_vocab_batch"] = target_ext_vocab_batch
        if "article_oovs" in item_info:
            d["article_oovs"] = item_info["article_oovs"]
            d["max_art_oovs"] = max(len(art_oovs) for art_oovs in item_info["article_oovs"])
    return d 

def get_dataloaders(is_small=False):
    train, val, test = load_examples(is_small)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    train_dataset = Dataset(tokenizer, *train)
    train_dl = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=config.batch_size, 
                                            collate_fn=collate_fn,
                                            shuffle=True)
    val_dataset = Dataset(tokenizer, *val)
    val_dl = torch.utils.data.DataLoader(dataset=val_dataset,
                                            batch_size=config.batch_size, 
                                            collate_fn=collate_fn,
                                            shuffle=True)
    test_dataset = Dataset(tokenizer, *test) 
    test_dl = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=config.batch_size, 
                                            collate_fn=collate_fn,
                                            shuffle=False)
    return train_dl, val_dl, test_dl, tokenizer


def text_input2bert_input(example, bert_tokenizer, seq_length=config.max_seq_length):
    tokens_a = bert_tokenizer.tokenize(example.text_a)
    tokens_b = None
    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens_a) > seq_length - 2:
        tokens_a = tokens_a[0:(seq_length - 2)]

    tokens = [] # equals raw text tokens 
    input_type_ids = [] # equals segments_ids
    tokens.append("[CLS]")
    input_type_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        input_type_ids.append(0)
    tokens.append("[SEP]")
    input_type_ids.append(0)

    input_ids = bert_tokenizer.convert_tokens_to_ids(tokens) # WordPiece embedding rep

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < seq_length:
        input_ids.append(0)
        input_mask.append(0)
        input_type_ids.append(0)

    input_ids_batch = torch.tensor(input_ids, dtype=torch.long)
    input_mask_batch = torch.tensor(input_mask, dtype=torch.long)
    example_index_batch = torch.zeros(input_ids_batch.size(), dtype=torch.long)

    return input_ids_batch, input_mask_batch, example_index_batch

if __name__ == "__main__":
    # # 1. for generating the pickle files from chunked.bin data files
    create_examples(config.train_data_path+'/chunked/*')

    # 2. obtain dataloaders
    # train_dl, val_dl, test_dl = get_dataloaders()

    # print(len(train_dl))
    # print(len(val_dl))
    # print(len(test_dl))