import os
import json
import argparse
import random
import numpy as np
from typing import Iterator, Tuple, Optional
from tqdm import tqdm
import copy
from pathlib import Path
from pypinyin import lazy_pinyin, Style

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import whisper
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from module.align_model import AlignModel
from dataset import get_alignment_dataloader, get_transcript_dataloader
from utils.alignment import perform_viterbi, perform_viterbi_sil, get_mae

os.environ["TOKENIZERS_PARALLELISM"]="false"

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-f', '--test-data',
        type=str,
        required=True
    )
    parser.add_argument(
        '--model-dir',
        type=str,
        default=None
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16
    )
    parser.add_argument(
        '--predict-sil',
        action='store_true'
    )
    parser.add_argument(
        '--use-pypinyin',
        action='store_true'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=114514
    )

    args = parser.parse_args()
    return args

whisper_dim = {'tiny': 384,
               'base': 512,
               'small': 768,
               'medium': 1024,
               'large': 1280}

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def load_align_model(
    model_path: Optional[str],
    whisper_model_name: str,
    text_output_dim: int,
    device: str='cuda'
) -> AlignModel:
    whisper_model = whisper.load_model(whisper_model_name, device=device)
    model = AlignModel(whisper_model=whisper_model,
                       embed_dim=whisper_dim[whisper_model_name],
                       output_dim=text_output_dim,
                       device=device)
    
    if model_path is not None:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict=state_dict)
    
    return model

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

def pypinyin_reweight(
    logits: torch.Tensor,
    labels,
    token_pinyin,
    pinyin_reverse,
):
    pinyin_reverse_keys = list(pinyin_reverse.keys())

    cur_same_pronun_token = []
    for k in range(len(pinyin_reverse_keys)):
        cur_same_pronun_token.append(torch.tensor(pinyin_reverse[pinyin_reverse_keys[k]]))

    for i in range(len(logits)):

        effective_pronun = []
        for k in range(len(labels[i])):
            # print(labels[i][k])
            if labels[i][k] == -100:
                continue

            # print (labels[i][k], token_pinyin[labels[i][k]])
            cur_key = token_pinyin[labels[i][k]]
            cur_key_index = pinyin_reverse_keys.index(cur_key)
            if cur_key_index not in effective_pronun:
                effective_pronun.append(cur_key_index)

        for j in range(len(logits[i])):
            cur_frame_best = torch.max(logits[i][j])
            # for k in range(len(pinyin_reverse_keys)):
            for k in effective_pronun:
                # selected = torch.index_select(logits[i][j], dim=0, index=cur_same_pronun_token[k])
                cur_value_list = cur_same_pronun_token[k]
                selected = logits[i][j][cur_value_list]
                # print (selected.shape)
                cur_max = torch.max(selected)

                logits[i][j][cur_value_list] = (cur_max + logits[i][j][cur_value_list]) / 2.0

    return logits

@torch.no_grad()
def align_and_evaluate(
    model: AlignModel,
    tokenizer,
    test_dataloader: DataLoader, 
    predict_sil: bool=False,
    use_pypinyin: bool=False,
    device: str='cuda'
):
    total_mae = 0
    model.eval()
    model.to(device)
    pbar = tqdm(test_dataloader)

    if use_pypinyin:
        print('Use Pypinyin to reweight, building pinyin table...')
        token_pinyin, pinyin_reverse = get_pinyin_table(tokenizer)
        print('Done.')

    for batch in pbar:
        mel, tokens, _, lyric_word_onset_offset = batch
        mel = mel.to(device)

        align_logits, _ = model(mel)

        align_logits = align_logits.cpu()
        if use_pypinyin:
            align_logits = pypinyin_reweight(align_logits, tokens, token_pinyin, pinyin_reverse)

        if predict_sil:
            align_results = perform_viterbi_sil(align_logits, tokens)
        else:
            align_results = perform_viterbi(align_logits, tokens)
        mae = get_mae(lyric_word_onset_offset, align_results)
        pbar.set_postfix({"current MAE": mae})

        total_mae += mae

    avg_mae = total_mae / len(test_dataloader)
    print("Average MAE:", avg_mae)
    return avg_mae

def main():
    args = parse_args()

    set_seed(args.seed)

    device = args.device
    if 'cuda' in device and torch.cuda.is_available() == False:
        device = 'cpu'

    # Load Tokenizer, Model
    if args.model_dir is not None:
        assert os.path.exists(args.model_dir)
        with open(os.path.join(args.model_dir, 'args.json'), 'r') as f:
            model_args = json.load(f)
        tokenizer_name = model_args['tokenizer']
        whisper_model_name = model_args['whisper_model']
        model_path = os.path.join(args.model_dir, 'best_model.pt')
    else:
        tokenizer_name = 'bert-base-chinese'
        whisper_model_name = 'base'
        model_path = None
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model = load_align_model(model_path=model_path,
                             whisper_model_name=whisper_model_name,
                             text_output_dim=len(tokenizer) + args.predict_sil,
                             device=device)
    
    assert os.path.exists(args.test_data)
    test_dataloader = get_alignment_dataloader(data_path=args.test_data,
                                                tokenizer=tokenizer,
                                                batch_size=args.batch_size,
                                                shuffle=False)
    
    align_and_evaluate(model=model,
                       tokenizer=tokenizer,
                       test_dataloader=test_dataloader,
                       predict_sil=args.predict_sil,
                       use_pypinyin=args.use_pypinyin,
                       device=device)


if __name__ == "__main__":
    main()
