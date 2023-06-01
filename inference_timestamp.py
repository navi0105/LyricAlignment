import argparse
import json
import os
import numpy as np
import pandas as pd
import random
from typing import List, Optional
from tqdm import tqdm
from pathlib import Path

from pypinyin import lazy_pinyin, Style

import torch
from torch.utils.data import DataLoader

from transformers import AutoTokenizer

import whisper
from whisper.tokenizer import get_tokenizer

from module.align_model import AlignModel, WHISPER_DIM
from data_processor.record import Record, read_data_from_csv, read_data_from_json
from inference_align import load_align_model_and_tokenizer
from utils.alignment import perform_viterbi, perform_viterbi_sil, get_pinyin_table, pypinyin_reweight
from utils.audio import load_audio_file, load_MIR1k_audio_file

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-f', '--data-file',
        type=str,
        required=True
    )
    parser.add_argument(
        '--model-dir',
        type=str,
        default=None,
    )
    parser.add_argument(
        '--predict-sil',
        action='store_true',
    )
    parser.add_argument(
        '--use-pypinyin',
        action='store_true',
    )
    parser.add_argument(
        '--is-mir1k',
        type=int,
        default=0,
    )
    parser.add_argument(
        '-o', '--output-dir',
        type=str,
        default='timestamps',
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=114514,
    )

    args = parser.parse_args()
    return args

def inference(
    model: AlignModel,
    tokenizer,
    records: List[Record],
    output_dir: str,
    is_mir1k: int=0,
    predict_sil: bool=False,
    use_pypinyin: bool=False,
    device: str='cuda', 
):
    print("Inference...")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if use_pypinyin:
        print('Use Pypinyin to reweight, building pinyin table...')
        token_pinyin, pinyin_reverse = get_pinyin_table(tokenizer)
        print('Done.')

    model.eval()
    model.to(device)

    pbar = tqdm(records, total=len(records))
    for idx, record in enumerate(pbar):
        audio_path = record.audio_path
        text = record.text

        if is_mir1k == 0:
            audio = load_audio_file(audio_path)['speech']
        elif is_mir1k == 1:
            audio = load_MIR1k_audio_file(audio_path, mixture=True)['speech']
        elif is_mir1k == 2:
            audio = load_MIR1k_audio_file(audio_path, mixture=False)['speech']
        else:
            raise NotImplementedError
        
        audio = [audio]
        text_tokens = tokenizer(text, return_tensors='pt')['input_ids'][:, 1: -1].to(device)

        with torch.no_grad():
            align_logits, _ = model.frame_manual_forward(audio)

        if use_pypinyin:
            align_logits = pypinyin_reweight(align_logits, text_tokens, token_pinyin, pinyin_reverse)

        if predict_sil:
            align_results = perform_viterbi_sil(align_logits, text_tokens)[0]
        else:
            align_results = perform_viterbi(align_logits, text_tokens)[0]

        # print(align_results)
        # print(text)
        # print(len(align_results), len(text))
        
        file_name = Path(audio_path).stem
        with open(os.path.join(output_dir, file_name + '.txt'), 'w') as f:
            for timestamp, char in zip(align_results, text):
                f.write(f'{timestamp[0]:.2f}\t{timestamp[1]:.2f}\t{char}\n')

    print("Done.")


def main():
    args = parse_args()

    set_seed(args.seed)
    
    device = args.device
    if 'cuda' in device and torch.cuda.is_available() == False:
        device = 'cpu'

    model, tokenizer = load_align_model_and_tokenizer(args.model_dir, args.predict_sil)
    whisper_tokenizer = get_tokenizer(multilingual=True, task='transcribe')

    assert os.path.exists(args.data_file)
    if os.path.splitext(args.data_file)[-1] == '.csv':
        records = read_data_from_csv(args.data_file)
    elif os.path.splitext(args.data_file)[-1] == '.json':
        records = read_data_from_json(args.data_file)
    else:
        raise NotImplementedError

    inference(model,
              tokenizer,
              records,
              args.output_dir,
              is_mir1k=args.is_mir1k,
              predict_sil=args.predict_sil,
              use_pypinyin=args.use_pypinyin,
              device=device,)

if __name__ == '__main__':
    main()