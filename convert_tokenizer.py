#!/usr/bin/python

import json

I_TO_TOKEN = {}
lines = open("rwkv_vocab_v20230424.txt", "r", encoding="utf-8").readlines()
for l in lines:
    idx = int(l[:l.index(' ')])
    x = eval(l[l.index(' '):l.rindex(' ')])
    if not isinstance(x, str):
        x = list(x)
    I_TO_TOKEN[idx] = x

out = open("rwkv_vocab_v20230424.json", "w")
out.write(json.dumps(I_TO_TOKEN, indent=4))