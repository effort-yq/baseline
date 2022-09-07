# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import numpy as np

def sparse_multilabel_categorical_crossentropy(y_true=None, y_pred=None, mask_zero=False):
    '''
    稀疏多标签交叉熵损失的torch实现
    '''
    shape = y_pred.shape  # [batch_size, ent_type_size, seq_len, seq_len]
    y_true = y_true[..., 0] * shape[2] + y_true[..., 1]
    y_pred = y_pred.reshape(shape[0], -1, np.prod(shape[2:]))
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred = torch.cat([y_pred, zeros], dim=-1)
    if mask_zero:
        infs = zeros + 1e12
        y_pred = torch.cat([infs, y_pred[..., 1:]], dim=-1)
    y_pos_2 = torch.gather(y_pred, index=y_true, dim=-1)
    y_pos_1 = torch.cat([y_pos_2, zeros], dim=-1)
    if mask_zero:
        y_pred = torch.cat([-infs, y_pred[..., 1:]], dim=-1)
        y_pos_2 = torch.gather(y_pred, index=y_true, dim=-1)
    pos_loss = torch.logsumexp(-y_pos_1, dim=-1)
    all_loss = torch.logsumexp(y_pred, dim=-1)
    aux_loss = torch.logsumexp(y_pos_2, dim=-1) - all_loss
    aux_loss = torch.clip(1 - torch.exp(aux_loss), 1e-10, 1)
    neg_loss = all_loss + torch.log(aux_loss)
    loss = torch.mean(torch.sum(pos_loss + neg_loss))
    return loss



class RawGlobalPointer(nn.Module):
    def __init__(self, hiddensize, ent_type_size, inner_dim=64, RoPE=True, tril_mask=True):
        '''
        :param encoder: BERT
        :param ent_type_size: 实体数目
        :param inner_dim: 64
        '''
        super(RawGlobalPointer, self).__init__()
        self.ent_type_size = ent_type_size
        self.inner_dim = inner_dim
        self.hidden_size = hiddensize
        self.dense = nn.Linear(self.hidden_size, self.ent_type_size * self.inner_dim * 2)
        self.RoPE = RoPE
        self.trail_mask = tril_mask

    def sinusoidal_position_embedding(self, batch_size, seq_len, output_dim):
        position_ids = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(-1)  # [seq_len, 1]

        indices = torch.arange(0, output_dim // 2, dtype=torch.float)  # [output_dim//2]
        indices = torch.pow(10000, -2 * indices / output_dim)  # 做幂相乘 [output_dim/2]
        embeddings = position_ids * indices  # [seq_len, out_dim/2]，每一行内容是position_ids值乘indices的值
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)  # [seq_len, out_dim/2, 2]
        embeddings = embeddings.repeat((batch_size, *([1] * len(embeddings.shape))))  # [batch_size, seq_len, out_dim/2, 2]
        embeddings = torch.reshape(embeddings, (batch_size, seq_len, output_dim)) # [batch_size, seq_len, out_dim]
        embeddings = embeddings.to(self.device)
        return embeddings

    def forward(self, context_outputs, attention_mask):
        self.device = attention_mask.device
        last_hidden_state = context_outputs[0]  # [batch, seq_len, hidden_size]
        batch_size = last_hidden_state.size()[0]
        seq_len = last_hidden_state.size()[1]
        outputs = self.dense(last_hidden_state)  # [batch_size, seq_len, ent_type_size * inner_dim * 2]
        outputs = torch.split(outputs, self.inner_dim * 2, dim=-1)  # 将outputs在最后一个维度上切分，每一份分别是[batch_size, seq_len, inner_dim*2]，即有ent_type_size份
        outputs = torch.stack(outputs, dim=-2)  # [batch_size, seq_len, ent_type_size, inner_dim*2]
        qw, kw = outputs[..., :self.inner_dim], outputs[..., self.inner_dim:]  # qw -->> kw -->> [batch_size, seq_len, ent_type_size, inner_dim]
        if self.RoPE:
            # pos_emb:(batch_size, seq_len, inner_dim)
            pos_emb = self.sinusoidal_position_embedding(batch_size, seq_len, self.inner_dim)
            cos_pos = pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)  # 按dim的维度重复2次
            sin_pos = pos_emb[..., None, ::2].repeat_interleave(2, dim=-1)  # [batch_size, seq_len, inner_dim]
            qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], -1)  # [batch_size, seq_len, ent_type_size, inner_dim//2, 2]
            qw2 = qw2.reshape(qw.shape)  # [batch_size, seq_len, ent_type_size, inner_dim]
            qw = qw * cos_pos + qw2 * sin_pos  # [batch_size, seq_len, ent_type_size, inner_dim]
            kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], -1)
            kw2 = kw2.reshape(kw.shape)
            kw = kw * cos_pos + kw2 * sin_pos  # [batch_size, seq_len, ent_type_size, inner_dim]
        # logits:(batch_size, ent_type_size, seq_len, seq_len)
        logits = torch.einsum('bmhd,bnhd->bhmn', qw, kw)
        # padding mask  # [batch_size, ent_type_size, seq_len, seq_len]
        pad_mask = attention_mask.unsqueeze(1).unsqueeze(1).expand(batch_size, self.ent_type_size, seq_len, seq_len)
        logits = logits * pad_mask - (1 - pad_mask) * 1e12   # padding部分置为负无穷
        # 排除下三角
        if self.trail_mask:
            mask = torch.tril(torch.ones_like(logits), -1)  # 下三角（不包括斜对角）
            logits = logits - mask * 1e12  # 下三角部分置为负无穷

        return logits / self.inner_dim ** 0.5