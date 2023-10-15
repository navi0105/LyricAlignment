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
from whisper.tokenizer import get_tokenizer

from transformers import AutoTokenizer

from module.align_model import AlignModel
from dataset import get_multitask_dataloader
from utils.alignment import perform_viterbi_ctc, perform_viterbi, get_mae

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
        default=1
    )
    parser.add_argument(
        '--is-mixture',
        choices=[0, 1, 2],
        default=0,
    )
    parser.add_argument(
        '--use-ctc-loss',
        action='store_true',
        help='set this flag for model trained with ctc loss'
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
    tokenizer_name = 'bert-base-chinese'
    whisper_model_name = train_args['whisper_model']
    model_path = os.path.join(model_dir, f'{args.model_name}_model.pt')

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

@torch.no_grad()
def align_and_evaluate(
    model: AlignModel,
    tokenizer,
    test_dataloader: DataLoader, 
    use_ctc_loss: bool=False,
    device: str='cuda'
):

    # Read lookup table
    with open(f"bert_base_chinese_pronunce_table.json", 'r') as f:
        token_pinyin, pinyin_reverse, pinyin_lookup_table = json.load(f)

    total_mae = 0
    model.eval()
    model.to(device)
    pbar = tqdm(test_dataloader)
    cnt = 0

    for batch in pbar:
        audios, tokens, _, lyric_word_onset_offset, _, _ = batch

        # print (tokens)
        for i in range(len(tokens)):
            for j in range(len(tokens[i])):
                if tokens[i][j] != -100:
                    tokens[i][j] = pinyin_lookup_table[token_pinyin[tokens[i][j]]]

        # print (tokens, lyric_word_onset_offset)

        if lyric_word_onset_offset == (None,):
            continue

        align_logits, _ = model.frame_manual_forward(audios)

        align_logits = align_logits.cpu()

        if use_ctc_loss:
            align_results = perform_viterbi_ctc(align_logits, tokens)
        else:
            align_results = perform_viterbi(align_logits, tokens)
        
        mae = get_mae(lyric_word_onset_offset, align_results)
        # print (mae)
        pbar.set_postfix({"current MAE": mae})

        total_mae += mae
        cnt = cnt + 1


    # avg_mae = total_mae / len(test_dataloader)
    avg_mae = total_mae / cnt
    print("Average MAE:", avg_mae)
    # print("Weighted MAE:", weighted_mae / len(segment_cnt))
    return avg_mae

def main():
    args = parse_args()

    # print (args)
    set_seed(args.seed)

    device = args.device
    if 'cuda' in device and torch.cuda.is_available() == False:
        device = 'cpu'

    # Load Tokenizer, Model
    assert os.path.exists(args.model_dir)
    model, tokenizer = load_align_model_and_tokenizer(args.model_dir, args, device=device)
    whisper_tokenizer = get_tokenizer(multilingual=True, task='transcribe')
    
    assert os.path.exists(args.test_data)

    test_dataloader = get_multitask_dataloader(
        args.test_data,
        hf_tokenizer=tokenizer,
        whisper_tokenizer=whisper_tokenizer,
        is_mixture=args.is_mixture,
        no_timestamps=True,
        use_ctc=args.use_ctc_loss,
        batch_size=args.batch_size,
        shuffle=False
    )
    
    align_and_evaluate(model=model,
                       tokenizer=tokenizer,
                       test_dataloader=test_dataloader,
                       use_ctc_loss=args.use_ctc_loss,
                       device=device)


if __name__ == "__main__":
    main()
