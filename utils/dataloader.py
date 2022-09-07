# -*- coding: utf-8 -*-

# author:Administrator
# contact: test@test.com
# datetime:2022/6/11 21:39
# software: PyCharm

"""
文件说明：
    
"""

import json
import numpy as np
import torch
from torch.utils.data import Dataset


def load_name(filename):
    """{"ID": "AT0010", "text": "故障现象：车速达到45Km/h时中央门锁不能落锁。",
    "spo_list": [{"h": {"name": "中央门锁", "pos": [16, 20]}, "t": {"name": "不能落锁", "pos": [20, 24]}, "relation": "部件故障"}]}
    """
    D = []
    data = json.load(open(filename, 'r', encoding='utf-8'))
    for line in data:
        D.append({
            'text': line['text'],
            'spo_list': [(spo['h']['name'], tuple(spo['h']['pos']), spo['relation'], spo['t']['name'], tuple(spo['t']['pos']))
                         for spo in line['spo_list']]
        })
    return D

    # with open(filename, 'r', encoding='utf-8') as f:
    #     for line in f:
    #         line = json.loads(line)
    #         D.append({
    #             'text': line['sentence'],
    #             'spo_list': [(spo['s'], spo['p'], spo['o']) for spo in line['spos']]
    #         })
    #     return D


def sequence_padding(inputs, length=None, value=0, seq_dims=1, mode='post'):
    """numpy函数，将序列padding到同一长度"""
    if length is None:
        length = np.max([np.shape(x)[:seq_dims] for x in inputs], axis=0)
    elif not hasattr(length, '__getitem__'):
        length = [length]

    slices = [np.s_[:length[i]] for i in range(seq_dims)]
    slices = tuple(slices) if len(slices) > 1 else slices[0]
    pad_width = [(0, 0) for _ in np.shape(inputs[0])]

    outputs = []
    for x in inputs:
        x = x[slices]
        for i in range(seq_dims):
            if mode == 'post':
                pad_width[i] = (0, length[i] - np.shape(x)[i])
            elif mode == 'pre':
                pad_width[i] = (length[i] - np.shape(x)[i], 0)
            else:
                raise ValueError('"mode" argument must be "post" or "pre".')
        x = np.pad(x, pad_width, 'constant', constant_values=value)
        outputs.append(x)
    return np.array(outputs)


def sequence_padding_entity(inputs, length=None, value=0, seq_dims=1, mode='post'):
    """numpy函数，将序列padding到同一长度"""
    if length is None:
        length = np.max([np.shape(x[0])[:seq_dims] for x in inputs], axis=0)
    elif not hasattr(length, '__getitem__'):
        length = [length]

    slices = [np.s_[:length[i]] for i in range(seq_dims)]
    slices = tuple(slices) if len(slices) > 1 else slices[0]
    pad_width = [(0, 0) for _ in np.shape(inputs[0][0])]

    outputs = []
    for x in inputs:
        x = x[0][slices]
        for i in range(seq_dims):
            if mode == 'post':
                pad_width[i] = (0, length[i] - np.shape(x)[i])
            elif mode == 'pre':
                pad_width[i] = (length[i] - np.shape(x)[i], 0)
            else:
                raise ValueError('"mode" argument must be "post" or "pre".')
        x = np.pad(x, pad_width, 'constant', constant_values=value)
        outputs.append(x)
    return np.array(outputs)


def search(pattern, sequence):
    """从sequence中寻找子串pattern
    如果找到，返回第一个下标；否则返回-1。
    """
    n = len(pattern)
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            return i
    return -1


class data_generator(Dataset):
    def __init__(self, data, tokenizer, max_len, schema):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.schema = schema  # spo

    def __len__(self):
        return len(self.data)

    def encoder(self, item):

        text = item['text']
        encoder_text = self.tokenizer(text, return_offsets_mapping=True, max_length=self.max_len, truncation=True)

        input_ids = encoder_text['input_ids']
        token_type_ids = encoder_text['token_type_ids']
        attention_mask = encoder_text['attention_mask']

        spoes = set()
        for s, s_pos, p, o, o_pos in item['spo_list']:
            s = self.tokenizer.encode(s, add_special_tokens=False)
            p = self.schema[p]
            o = self.tokenizer.encode(o, add_special_tokens=False)

            sh = search(s, input_ids)
            oh = search(o, input_ids)

            if sh != -1 and oh != -1:
                spoes.add((sh, sh + len(s) - 1, p, oh, oh + len(o) - 1))

        entity_labels = [set() for i in range(2)]
        head_labels = [set() for i in range(len(self.schema))]
        tail_labels = [set() for i in range(len(self.schema))]

        for sh, st, p, oh, ot in spoes:
            entity_labels[0].add((sh, st))  # 实体提取：2个类型，头实体or尾实体
            entity_labels[1].add((oh, ot))
            head_labels[p].add((sh, oh))  # 类似于TP-Linker
            tail_labels[p].add((st, ot))

        for label in entity_labels + head_labels + tail_labels:
            if not label:
                label.add((0, 0))

        # 例如entity = [{(1, 3)}, {(4, 5), (7, 9)}]
        # entity[0]即{(1,3)}代表头实体首尾， entity[1]即{(4,5),{7,9}}代表尾实体首尾
        # 需要标签对齐为 [[[1,3][0,0]] , [[4,5][7,9]]]
        entity_labels = sequence_padding([list(l) for l in entity_labels])
        head_labels = sequence_padding([list(l) for l in head_labels])
        tail_labels = sequence_padding([list(l) for l in tail_labels])

        return text, entity_labels, head_labels, tail_labels, \
               input_ids, attention_mask, token_type_ids

    def __getitem__(self, idx):
        item = self.data[idx]
        return self.encoder(item)

    @staticmethod
    def collate(examples):

        batch_token_ids, batch_mask_ids, batch_token_type_ids = [], [], []
        batch_entity_labels, batch_head_labels, batch_tail_labels = [], [], []
        text_list = []

        for item in examples:

            text, entity_labels, head_labels, tail_labels, \
            input_ids, attention_mask, token_type_ids = item

            batch_entity_labels.append(entity_labels)
            batch_head_labels.append(head_labels)
            batch_tail_labels.append(tail_labels)
            batch_token_ids.append(input_ids)
            batch_mask_ids.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)
            text_list.append(text)

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).float()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()  # RoBERTa 不需要NSP
        batch_entity_labels = torch.tensor(sequence_padding(batch_entity_labels, seq_dims=2)).long()
        batch_head_labels = torch.tensor(sequence_padding(batch_head_labels, seq_dims=2)).long()
        batch_tail_labels = torch.tensor(sequence_padding(batch_tail_labels, seq_dims=2)).long()

        return text_list, batch_token_ids, batch_mask_ids, batch_token_type_ids,\
               batch_entity_labels, batch_head_labels, batch_tail_labels
