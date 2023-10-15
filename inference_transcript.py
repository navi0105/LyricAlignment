import argparse
import os
import json
import whisper
import torch
import pandas as pd
from typing import List, Optional, Any
from pathlib import Path
from tqdm import tqdm

from transformers import AutoTokenizer
import whisper

from module.align_model import AlignModel
from data_processor.record import Record, read_data

from utils.audio import load_audio_file

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-f', "--test-data",
        type=str,
        required=True,
        help="Data file for decode / transcription"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Whisper model name or path"
    )
    parser.add_argument(
        '--use-pretrained',
        action='store_true'
    )

    parser.add_argument(
        '--use-groundtruth',
        action='store_true'
    )

    parser.add_argument(
        '--beam_size',
        type=int,
        default=5
    )
    parser.add_argument(
        '--is-mixture',
        choices=[0, 1, 2],
        default=0,
        help="0: mono; 1: mixture; 2: mixture, but vocal only"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help=""
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="output/result.json"
    )

    args = parser.parse_args()

    return args


def transcribe(
    model: whisper.Whisper,
    records: List[Record],
    language: str='zh',
    beam_size: int=5,
    is_mixture: int=0,
    use_groundtruth: bool=True
) -> List[dict]:
    transcribe_results = []
    for record in tqdm(records):
        audio_path = record.audio_path
        song_id = Path(audio_path).stem

        if is_mixture == 0:
            audio = load_audio_file(audio_path, is_mixture)['speech']
        
        result = model.transcribe(audio=audio,
                                  task='transcribe',
                                  language=language,
                                  beam_size=beam_size)
        
        # print(result)
        if use_groundtruth:
            transcribe_results.append({'song_id': song_id,
                                       'song_path': record.audio_path,
                                       'lyric': record.text,
                                       'inference': result['text']})
        else:
            transcribe_results.append({'song_id': song_id,
                                       'song_path': record.audio_path,
                                       'inference': result['text']})

    return transcribe_results

WHISPER_DIM = {'tiny': 384,
               'base': 512,
               'small': 768,
               'medium': 1024,
               'large': 1280}

def load_align_model(
    model_dir: str,
    args,
    device: str='cuda'
) -> AlignModel:
    assert os.path.exists(model_dir)
    with open(os.path.join(model_dir, 'args.json'), 'r') as f:
        train_args = json.load(f)
    tokenizer_name = 'bert-base-chinese'
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
    
    return model

def main():
    overwrite_output = False
    args = parse_args()
    if os.path.exists(args.output) and overwrite_output == False:
        print("File Exists, Pass")
        exit()

    device = args.device
    if device == 'cuda' and torch.cuda.is_available() != True:
        device = 'cpu'

    if os.path.exists(args.model_dir) and args.use_pretrained != True:
        transcribe_model = load_align_model(model_dir=args.model_dir,
                                            args=args,
                                            device=device).whisper_model
    else:
        print('Use pretrained model')
        transcribe_model = whisper.load_model(name='medium',
                                              device=device)
    
    transcribe_model.to(device)

    assert os.path.exists(args.test_data)
    test_records = read_data(args.test_data)
    
    transcribe_results = transcribe(model=transcribe_model,
                                    records=test_records,
                                    language='zh',
                                    beam_size=args.beam_size,
                                    is_mixture=args.is_mixture,
                                    use_groundtruth=args.use_groundtruth)
    
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(transcribe_results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()