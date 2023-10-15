import os
import json
import argparse
import random
import numpy as np
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

import pypinyin
from pypinyin import lazy_pinyin, Style

def get_pinyin_table(tokenizer):
    def handle_error(chars):
        return ['bad', 'bad']

    tokens = tokenizer.convert_ids_to_tokens(np.arange(0, len(tokenizer), 1).astype(int))
    # print (tokens)
    token_pinyin = []
    pinyin_reverse = {}
    for i in range(len(tokens)):
        try:
            cur_pinyin = lazy_pinyin(tokens[i], style=Style.NORMAL, errors=handle_error)
        except:
            cur_pinyin = ['bad', 'bad']
        if len(cur_pinyin) == 1:
            token_pinyin.append(cur_pinyin[0])
            if cur_pinyin[0] not in pinyin_reverse.keys():
                pinyin_reverse[cur_pinyin[0]] = [i,]
            else:
                pinyin_reverse[cur_pinyin[0]].append(i)
        else:
            token_pinyin.append('bad')

    return token_pinyin, pinyin_reverse


if __name__ == "__main__":
    hf_tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')

    token_pinyin, pinyin_reverse = get_pinyin_table(hf_tokenizer)

    pinyin_lookup_table = {}
    for i in range(len(token_pinyin)):
        if not token_pinyin[i] in pinyin_lookup_table:
            pinyin_lookup_table[token_pinyin[i]] = len(pinyin_lookup_table) + 1

    with open(f"bert_base_chinese_pronunce_table.json", 'w') as f:
        json.dump([token_pinyin, pinyin_reverse, pinyin_lookup_table], f, indent=2)