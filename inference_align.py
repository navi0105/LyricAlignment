import os
import json
import argparse
import random
import itertools
import numpy as np
from typing import Iterator, Tuple, Optional
from tqdm import tqdm
from pathlib import Path
from pypinyin import lazy_pinyin, Style

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import whisper
from whisper.audio import log_mel_spectrogram
from whisper.tokenizer import get_tokenizer
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from module.align_model import AlignModel
from dataset import get_alignment_dataloader, get_multitask_dataloader
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
        '--model-name',
        choices=['best', 'best_align', 'best_trans', 'last'],
        default='best'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16
    )
    parser.add_argument(
        '--predict-sil',
        type=bool,
        default=True,
        help='set this flag for model trained with ctc loss'
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

WHISPER_DIM = {'tiny': 384,
               'base': 512,
               'small': 768,
               'medium': 1024,
               'large': 1280}

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def load_align_model_and_tokenizer(
    model_dir: str,
    args,
    device: str='cuda'
) -> AlignModel:
    assert os.path.exists(model_dir)
    with open(os.path.join(model_dir, 'args.json'), 'r') as f:
        train_args = json.load(f)
    tokenizer_name = train_args['tokenizer']
    whisper_model_name = train_args['whisper_model']
    model_path = os.path.join(model_dir, 'best_model.pt')

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    whisper_model = whisper.load_model(whisper_model_name, device=device)

    if os.path.exists(os.path.join(model_dir, 'model_args.json')):
        with open(os.path.join(model_dir, 'model_args.json'), 'r') as f:
            model_args = json.load(f)
    else:
        model_args = {'embed_dim': WHISPER_DIM[whisper_model_name],
                      'hidden_dim': 384,
                      'bidirectional': True,
                      'output_dim': len(tokenizer) + args.predict_sil,}

    bidirectional = model_args.get('bidirectional', True)

    model = AlignModel(whisper_model=whisper_model,
                       embed_dim=model_args['embed_dim'],
                       hidden_dim=model_args['hidden_dim'],
                       output_dim=model_args['output_dim'],
                       bidirectional=bidirectional,
                       device=device)
    
    if model_path is not None:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict=state_dict)
    
    return model, tokenizer


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

                logits[i][j][cur_value_list] = (cur_max * 4.0 + logits[i][j][cur_value_list]) / 5.0

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
    if use_pypinyin:
        print('Use Pypinyin to reweight, building pinyin table...')
        token_pinyin, pinyin_reverse = get_pinyin_table(tokenizer)
        print('Done.')

    total_mae = 0
    model.eval()
    model.to(device)
    pbar = tqdm(test_dataloader)

    for batch in pbar:
        audios, tokens, _, lyric_word_onset_offset, _, _ = batch

        align_logits, _ = model.frame_manual_forward(audios)

        align_logits = align_logits.cpu()
        if use_pypinyin:
            align_logits = pypinyin_reweight(align_logits, tokens, token_pinyin, pinyin_reverse)

        if predict_sil:
            align_results = perform_viterbi_sil(align_logits, tokens)
        else:
            align_results = perform_viterbi(align_logits, tokens)
        
        # print(align_logits)
        # print(lyric_word_onset_offset)
        # print(align_results)
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
    assert os.path.exists(args.model_dir)
    model, tokenizer = load_align_model_and_tokenizer(args.model_dir, args)
    whisper_tokenizer = get_tokenizer(multilingual=True, task='transcribe')
    
    assert os.path.exists(args.test_data)
    # test_dataloader = get_alignment_dataloader(data_path=args.test_data,
    #                                             tokenizer=tokenizer,
    #                                             batch_size=args.batch_size,
    #                                             use_ctc=args.predict_sil,
    #                                             is_inference=True,
    #                                             shuffle=False)

    test_dataloader = get_multitask_dataloader(args.test_data,
                                               hf_tokenizer=tokenizer,
                                               whisper_tokenizer=whisper_tokenizer,
                                               use_ctc=args.predict_sil,
                                               use_v2_dataset=True,
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
