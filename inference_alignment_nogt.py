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

from utils.audio import load_audio_file

from data_processor.record import read_data

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
    test_data,
    use_ctc_loss: bool=False,
    device: str='cuda'
):

    # Read lookup table
    with open(f"bert_base_chinese_pronunce_table.json", 'r') as f:
        token_pinyin, pinyin_reverse, pinyin_lookup_table = json.load(f)

    total_mae = 0
    model.eval()
    model.to(device)

    records = read_data(test_data)

    for record in tqdm(records):
        audio_path = record.audio_path
        song_id = Path(audio_path).stem

        audio = load_audio_file(audio_path)['speech']

        align_logits, _ = model.frame_manual_forward([audio,])
        align_logits = align_logits.cpu()

        tokens = tokenizer(record.text,
                            padding=True,
                            return_tensors='pt')['input_ids'][:, 1:]
        
        tokens[tokens == 0] = -100
        tokens[tokens == 102] = -100

        for i in range(len(tokens)):
            for j in range(len(tokens[i])):
                if tokens[i][j] != -100:
                    tokens[i][j] = pinyin_lookup_table[token_pinyin[tokens[i][j]]]

        if use_ctc_loss:
            align_results = perform_viterbi_ctc(align_logits, tokens)
        else:
            align_results = perform_viterbi(align_logits, tokens)

        cur_prediction = [[align_results[0][i][0], align_results[0][i][1], record.text[i]] for i in range(len(align_results[0]))]
        print (cur_prediction)

    return

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
    
    align_and_evaluate(model=model,
                       tokenizer=tokenizer,
                       test_data=args.test_data,
                       use_ctc_loss=args.use_ctc_loss,
                       device=device)


if __name__ == "__main__":
    main()
