import os
import json
import argparse
import random
import numpy as np
from typing import Iterator, Tuple, Optional
from tqdm import tqdm
import copy
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import whisper
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from module.align_model import AlignModel
from dataset import get_dataloader
from utils.alignment import perform_viterbi, get_mae

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
        '--device',
        type=str,
        default='cuda'
    )

    args = parser.parse_args()
    return args


whisper_dim = {'tiny': 384,
               'base': 512,
               'small': 768,
               'medium': 1024,
               'large': 1280}

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
        state_dict = torch.load(model_path)['model_state_dict']
        model.load_state_dict(state_dict=state_dict)
    
    return model

@torch.no_grad()
def align_and_evaluate(model: AlignModel, 
                       test_dataloader: DataLoader, 
                       device: str='cuda'):
    total_mae = 0
    model.eval()
    model.to(device)
    pbar = tqdm(test_dataloader)
    for batch in pbar:
        mel, tokens, _, lyric_word_onset_offset = batch
        mel, tokens = mel.to(device), tokens.to(device)

        align_logits, _ = model(mel)

        align_results = perform_viterbi(align_logits, tokens)
        mae = get_mae(lyric_word_onset_offset, align_results)
        pbar.set_postfix({"current MAE": mae})

        total_mae += mae

    avg_mae = total_mae / len(test_dataloader)
    print("Average MAE:", avg_mae)
    return avg_mae

def main():
    args = parse_args()

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
                             text_output_dim=len(tokenizer),
                             device=device)
    
    assert os.path.exists(args.test_data)
    test_dataloader = get_dataloader(data_path=args.test_data,
                                    tokenizer=tokenizer,
                                    batch_size=args.batch_size,
                                    shuffle=False)
    
    align_and_evaluate(model=model,
                       test_dataloader=test_dataloader,
                       device=device)


if __name__ == "__main__":
    main()
