# -*- coding: utf-8 -*-

# author:Administrator
# contact: test@test.com
# datetime:2022/9/4 10:54
# software: PyCharm

"""
文件说明：
    mean: 172, 75% 235
    总共1491条数据
"""
import json
import pandas as pd
import random
t = set()
text = []
D = []
with open('./train.json', 'r', encoding='utf-8') as f:
    for line in f:
        line = json.loads(line)
        D.append(line)

print(len(D))
random.shuffle(D)

train = D[:1000]
eval = D[1000:]
with open('train_data.json', 'w', encoding='utf-8') as w:
    json.dump(train, w, ensure_ascii=False, indent=4)

with open('eval_data.json', 'w', encoding='utf-8') as w:
    json.dump(eval, w, ensure_ascii=False, indent=4)

#         spos = line['spo_list']
#         text.append(line['text'])
#         for spo in spos:
#             t.add(spo['relation'])
# print(t)
# df = pd.DataFrame({'text': text})
# print(df.text.apply(len).describe())
