# -*- coding: utf-8 -*-

# author:Administrator
# contact: test@test.com
# datetime:2022/6/12 9:08
# software: PyCharm

"""
文件说明：
    
"""


import json
import torch
import sys
import numpy as np
import torch.nn as nn
from transformers import BertTokenizerFast, BertModel
from nets.gpNet import RawGlobalPointer, sparse_multilabel_categorical_crossentropy
from utils.dataloader import data_generator, load_name
from utils.bert_optimization import BertAdam
from torch.utils.data import DataLoader
import configparser
from tqdm import tqdm
import os
import random



con = configparser.ConfigParser()
con.read('./config.ini', encoding='utf8')
args_path = dict(dict(con.items('paths')), **dict(con.items("para")))

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

setup_seed(2022)

tokenizer = BertTokenizerFast.from_pretrained(args_path["model_path"], do_lower_case=True)
encoder = BertModel.from_pretrained(args_path["model_path"])

# subject_type表明subject是什么类型，object_type同理
schema = {'部件故障': 0, '性能故障': 1, '检测工具': 2, '组成': 3}
id2schema = {}
for k, v in schema.items():
    id2schema[v] = k

train_data = data_generator(load_name(args_path['train_file']), tokenizer, max_len=con.getint('para', 'maxlen'), schema=schema)

dev_data = data_generator(load_name(args_path['valid_file']), tokenizer, max_len=con.getint('para', 'maxlen'), schema=schema)
train_loader = DataLoader(train_data, batch_size=con.getint('para', 'batch_size'), shuffle=True, collate_fn=train_data.collate)
dev_loader = DataLoader(dev_data, batch_size=con.getint('para', 'batch_size'), shuffle=True, collate_fn=dev_data.collate)

valid_data = load_name(args_path['valid_file'])

device = torch.device('cuda')

mention_detect = RawGlobalPointer(hiddensize=768, ent_type_size=2, inner_dim=64).to(device)  # 实体关系抽取任务默认不提取实体类型
s_o_head = RawGlobalPointer(hiddensize=768, ent_type_size=len(schema), inner_dim=64, RoPE=False, tril_mask=False).to(device)
s_o_tail = RawGlobalPointer(hiddensize=768, ent_type_size=len(schema), inner_dim=64, RoPE=False, tril_mask=False).to(device)

class ERENet(nn.Module):
    def __init__(self, encoder, a, b, c):
        super(ERENet, self).__init__()
        self.mention_detect = a   # mention_detect
        self.s_o_head = b  # 检测s、o的头部
        self.s_o_tail = c  # 检测s、o的尾部
        self.encoder = encoder  # bert编码器

    def forward(self, batch_token_ids, batch_mask_ids, batch_token_type_ids):
        outputs = self.encoder(batch_token_ids, batch_mask_ids, batch_token_type_ids)
        mention_outputs = self.mention_detect(outputs, batch_mask_ids)  # [batch_size, ent_type_size, seq_len, seq_len]
        so_head_outputs = self.s_o_head(outputs, batch_mask_ids)
        so_tail_outputs = self.s_o_tail(outputs, batch_mask_ids)
        return mention_outputs, so_head_outputs, so_tail_outputs

net = ERENet(encoder, mention_detect, s_o_head, s_o_tail).to(device)

def set_optimizer(model, train_steps=None):
    param_optimizer = list(model.named_parameters())
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=4e-5,
                         warmup=0.1,
                         t_total=train_steps)
    return optimizer

def search(pattern, sequence):
    """从sequence中寻找子串pattern
    如果找到，返回第一个下标；否则返回-1。
    """
    n = len(pattern)
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            return i
    return -1

def extract_spoes(text, threshold=0, model=None):
    """抽取输入text中所包含的三元组"""
    max_seq_len = 512
    token2char_span_mapping = tokenizer(text, return_offsets_mapping=True, max_length=max_seq_len)['offset_mapping']
    new_span, entities = [], []
    for i in token2char_span_mapping:
        if i[0] == i[1]:
            new_span.append([])
        else:
            if i[0] + 1 == i[1]:  # 单个字
                new_span.append([i[0]])
            else:
                new_span.append([i[0], i[-1] - 1])  # 闭区间
    encoder_txt = tokenizer.encode_plus(text, max_length=max_seq_len)
    input_ids = torch.tensor(encoder_txt['input_ids']).long().unsqueeze(0).to(device)
    token_type_ids = torch.tensor(encoder_txt["token_type_ids"]).unsqueeze(0).to(device)
    attention_mask = torch.tensor(encoder_txt["attention_mask"]).unsqueeze(0).to(device)
    scores = model(input_ids, attention_mask, token_type_ids)
    outputs = [o[0].data.cpu().numpy() for o in scores]  # list类型，每个位置形状[ent_type_size, seq_len, seq_len]
    # 抽取subject和object
    subjects, objects = set(), set()
    outputs[0][:, [0, -1]] -= np.inf  # 在seq_len维度首尾取负无穷
    outputs[0][:, :, [0, -1]] -= np.inf
    for l, h, t in zip(*np.where(outputs[0] > 0)):
        if l == 0:
            subjects.add((h, t))
        else:
            objects.add((h, t))
    # 识别对应的predicate
    spoes = set()
    for sh, st in subjects:
        for oh, ot in objects:
            p1s = np.where(outputs[1][:, sh, oh] > threshold)[0]
            p2s = np.where(outputs[2][:, st, ot] > threshold)[0]
            ps = set(p1s) & set(p2s)  # 取交集
            for p in ps:
                spoes.add((
                    text[new_span[sh][0]:new_span[st][-1] + 1], (new_span[sh][0], new_span[st][-1] + 1), id2schema[p],
                    text[new_span[oh][0]:new_span[ot][-1] + 1], (new_span[oh][0], new_span[ot][-1] + 1)
                ))
    return list(spoes)

class SPO(tuple):
    """用来存三元组的类，表现跟tuple基本一致，重写了两个特殊方法，使得在判断两个三元组是否等价时容错性更好"""
    def __init__(self, spo):
        self.spox = (
            tuple(tokenizer.tokenize(spo[0], add_special_tokens=False)),
            tuple(spo[1]),
            spo[2],
            tuple(tokenizer.tokenize(spo[3], add_special_tokens=False)),
            tuple(spo[4])
        )

    def __hash__(self):
        return self.spox.__hash__()
    def __eq__(self, spo):
        return self.spox == spo.spox


def evaluate(data, model):
    """评估函数，计算f1、Precision、Recall"""
    
    model.eval()
    
    X, Y, Z = 1e-10, 1e-10, 1e-10
    correct_bujian, predict_bujian, gold_bujian = 1e-10, 1e-10, 1e-10
    correct_xingneng, predict_xingneng, gold_xingneng = 1e-10, 1e-10, 1e-10
    correct_jiance, predict_jiance, gold_jiance = 1e-10, 1e-10, 1e-10
    correct_zucheng, predict_zucheng, gold_zucheng = 1e-10, 1e-10, 1e-10

    f = open('./dev_pred.json', 'w', encoding='utf-8')
    bujian = 0
    xingneng = 0
    jiance = 0
    zucheng = 0
    
    for d in tqdm(data, desc='Evaluation', total=len(data)):
        R = set([SPO(spo) for spo in extract_spoes(d['text'], threshold=0, model=model)])
        T = set([SPO(spo) for spo in d['spo_list']])
        X += len(R & T)  # 抽取三元组和标注三元组匹配的个数，包括h.name,t.name,h.pos,t.pos以及relation都相同
        Y += len(R)  # 抽取三元组个数
        Z += len(T)  # 标注三元组个数

        bujian_pred, bujian_gold = [], []
        xingneng_pred, xingneng_gold = [], []
        jiance_pred, jiance_gold = [], []
        zucheng_pred, zucheng_gold = [], []
        for item in list(R):
            if item[2] == '部件故障':
                bujian_pred.append((item[0], item[1], item[-2], item[-1]))
            elif item[2] == '性能故障':
                xingneng_pred.append((item[0], item[1], item[-2], item[-1]))
            elif item[2] == "检测工具":
                jiance_pred.append((item[0], item[1], item[-2], item[-1]))
            else:
                zucheng_pred.append((item[0], item[1], item[-2], item[-1]))

        for dom in list(T):
            if dom[2] == '部件故障':
                bujian += 1
                bujian_gold.append((dom[0], dom[1], dom[-2], dom[-1]))
            if dom[2] == '性能故障':
                xingneng += 1
                xingneng_gold.append((dom[0], dom[1], dom[-2], dom[-1]))
            if dom[2] == '检测工具':
                jiance += 1
                jiance_gold.append((dom[0], dom[1], dom[-2], dom[-1]))
            if dom[2] == '组成':
                zucheng += 1
                zucheng_gold.append((dom[0], dom[1], dom[-2], dom[-1]))

        correct_bujian += len([t for t in bujian_pred if t in bujian_gold])
        predict_bujian += len(bujian_pred)
        gold_bujian += len(bujian_gold)

        correct_xingneng += len([t for t in xingneng_pred if t in xingneng_gold])
        predict_xingneng += len(xingneng_pred)
        gold_xingneng += len(xingneng_gold)

        correct_jiance += len([t for t in jiance_pred if t in jiance_gold])
        predict_jiance += len(jiance_pred)
        gold_jiance += len(jiance_gold)

        correct_zucheng += len([t for t in zucheng_pred if t in zucheng_gold])
        predict_zucheng += len(zucheng_pred)
        gold_zucheng += len(zucheng_gold)


        s = json.dumps({
            'text': d['text'],
            'spo_list': list(T),
            'spo_list_pred': list(R),
            'new': list(R - T),
            'lack': list(T - R),
        },
                       ensure_ascii=False,
                       indent=4)
        f.write(s + '\n')

    bujian_p = correct_bujian / predict_bujian
    bujian_r = correct_bujian / gold_bujian
    bujian_f = 2 * bujian_p * bujian_r / (bujian_p + bujian_r)

    xingneng_p = correct_xingneng / predict_xingneng
    xingneng_r = correct_xingneng / gold_xingneng
    xingneng_f = 2 * xingneng_p * xingneng_r / (xingneng_p + xingneng_r)

    jiance_p = correct_jiance / predict_jiance
    jiance_r = correct_jiance / gold_jiance
    jiance_f = 2 * jiance_p * jiance_r / (jiance_p + jiance_r)

    zucheng_p = correct_zucheng / predict_zucheng
    zucheng_r = correct_zucheng / gold_zucheng
    zucheng_f = 2 * zucheng_p * zucheng_r / (zucheng_p + zucheng_r)
    model.train()
    print('部件的f1是: ', bujian_f, '性能的f1是: ', xingneng_f, '检测f1是: ', jiance_f, '组成的f1是：', zucheng_f)
    micro_f1 = (bujian_f * bujian + xingneng_f * xingneng + jiance_f * jiance + zucheng_f * zucheng) / (bujian + xingneng + jiance + zucheng)
    f.close()
    return micro_f1


optimizer = set_optimizer(net, train_steps=(int(len(train_data) / con.getint("para", "batch_size")) + 1) * con.getint("para", "epochs"))
total_loss, total_f1 = 0., 0.
total_step = 0
best_f1 = 0
for eo in range(con.getint('para', 'epochs')):
    for idx, batch in enumerate(train_loader):
        total_step += 1
        text, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels, batch_head_labels, batch_tail_labels = batch
        batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels, batch_head_labels, batch_tail_labels = \
            batch_token_ids.to(device), batch_mask_ids.to(device), batch_token_type_ids.to(
                device), batch_entity_labels.to(device), batch_head_labels.to(device), batch_tail_labels.to(device)
        logits1, logits2, logits3 = net(batch_token_ids, batch_mask_ids, batch_token_type_ids)
        loss1 = sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits1, mask_zero=True)
        loss2 = sparse_multilabel_categorical_crossentropy(y_true=batch_head_labels, y_pred=logits2, mask_zero=True)
        loss3 = sparse_multilabel_categorical_crossentropy(y_true=batch_tail_labels, y_pred=logits3, mask_zero=True)
        loss = (1.5 * loss1 + loss2 + loss3) / 3
        # loss = sum([loss1, loss2, loss3]) / 3
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        sys.stdout.write("\r [EPOCH %d/%d] [Loss:%f]" % (eo, con.getint("para", "epochs"), loss.item()))
        if total_step % 200 == 0:
            f = evaluate(valid_data, net)
            if f > best_f1:
                best_f1 = f
                torch.save(net.state_dict(), './erenet.pth')
                print('The Best f1 score is: ', best_f1)


    torch.save(net.state_dict(), './erenet.pth')