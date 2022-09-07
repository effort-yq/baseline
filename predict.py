# -*- coding: utf-8 -*-

# author:Administrator
# contact: test@test.com
# datetime:2022/6/12 10:46
# software: PyCharm

"""
文件说明：
    
"""
import torch
import json
import sys
import numpy as np
import torch.nn as nn
import configparser
from nets.gpNet import RawGlobalPointer, sparse_multilabel_categorical_crossentropy
from utils.dataloader import data_generator, load_name
from utils.bert_optimization import BertAdam
from transformers import BertModel, BertTokenizerFast

con = configparser.ConfigParser()
con.read('./config.ini', encoding='utf-8')
args_path = dict(dict(con.items('paths')), **dict(con.items('para')))

tokenizer = BertTokenizerFast.from_pretrained(args_path['model_path'], do_lower_case=True)
encoder = BertModel.from_pretrained(args_path['model_path'])

# subject_type表明subject是什么类型，object_type同理
schema = {'部件故障': 0, '性能故障': 1, '检测工具': 2, '组成': 3}
id2schema = {}
for k, v in schema.items():
    id2schema[v] = k
device = torch.device('cuda')

mention_detect = RawGlobalPointer(hiddensize=768, ent_type_size=2, inner_dim=64).to(device)  # 实体关系抽取任务默认不提取实体类型
s_o_head = RawGlobalPointer(hiddensize=768, ent_type_size=len(schema), inner_dim=64, RoPE=False, tril_mask=False).to(device)
s_o_tail = RawGlobalPointer(hiddensize=768, ent_type_size=len(schema), inner_dim=64, RoPE=False, tril_mask=False).to(device)

class ERENet(nn.Module):
    def __init__(self, encoder, a, b, c):
        super(ERENet, self).__init__()
        self.mention_detect = a
        self.s_o_head = b
        self.s_o_tail = c
        self.encoder = encoder

    def forward(self, batch_token_ids, batch_mask_ids, batch_token_type_ids):
        outputs = self.encoder(batch_token_ids, batch_mask_ids, batch_token_type_ids)
        mention_outputs = self.mention_detect(outputs, batch_mask_ids)
        so_head_outputs = self.s_o_head(outputs, batch_mask_ids)
        so_tail_outputs = self.s_o_tail(outputs, batch_mask_ids)
        return mention_outputs, so_head_outputs, so_tail_outputs

net = ERENet(encoder, mention_detect, s_o_head, s_o_tail).to(device)
net.load_state_dict(torch.load('./erenet.pth'))
net.eval()
data = []
with open(args_path['test_file'], 'r', encoding='utf-8') as f:
    text_list = [(json.loads(text.rstrip())['text'], json.loads(text.rstrip())['ID']) for text in f.readlines()]

    for text, id_ in text_list:
        result = {}
        token2char_span_mapping = tokenizer(text, return_offsets_mapping=True, max_length=512)['offset_mapping']
        new_span, entities = [], []
        for i in token2char_span_mapping:
            if i[0] == i[1]:
                new_span.append([])
            else:
                if i[0] + 1 == i[1]:
                    new_span.append([i[0]])
                else:
                    new_span.append([i[0], i[-1] - 1])
        threshold = 0.0
        encoder_txt = tokenizer.encode_plus(text, max_length=512)
        input_ids = torch.tensor(encoder_txt['input_ids']).long().unsqueeze(0).to(device)
        token_type_ids = torch.tensor(encoder_txt["token_type_ids"]).unsqueeze(0).to(device)
        attention_mask = torch.tensor(encoder_txt["attention_mask"]).unsqueeze(0).to(device)
        scores = net(input_ids, attention_mask, token_type_ids)
        outputs = [o[0].data.cpu().numpy() for o in scores]
        subjects, objects = set(), set()
        outputs[0][:, [0, -1]] -= np.inf  # 首尾取负无穷
        outputs[0][:, :, [0, -1]] -= np.inf
        for l, h, t in zip(*np.where(outputs[0] > 0)):
            if l == 0:
                subjects.add((h, t))
            else:
                objects.add((h, t))
        spoes = set()
        for sh, st in subjects:
            for oh, ot in objects:
                p1s = np.where(outputs[1][:, sh, oh] > threshold)[0]
                p2s = np.where(outputs[2][:, st, ot] > threshold)[0]
                ps = set(p1s) & set(p2s)
                for p in ps:
                    spoes.add((
                        text[new_span[sh][0]:new_span[st][-1] + 1], (new_span[sh][0], new_span[st][-1] + 1), id2schema[p],
                        text[new_span[oh][0]:new_span[ot][-1] + 1], (new_span[oh][0], new_span[ot][-1] + 1)
                    ))
        spo_list = []
        result['ID'] = id_
        result['text'] = text
        for spo in list(spoes):
            spo_list.append({'h': {'name': spo[0], 'pos': list(spo[1])}, 't': {'name': spo[3], 'pos': list(spo[4])}, 'relation': spo[2]})
        result["spo_list"] = spo_list
        data.append(json.dumps(result, ensure_ascii=False))

    with open('./evalResult.json', 'w', encoding='utf-8') as w:
        for line in data:
            w.write(line)
            w.write('\n')

    print('Finish!!!!')