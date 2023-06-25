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
from whisper.audio import log_mel_spectrogram, pad_or_trim, N_SAMPLES, N_FRAMES
from whisper.tokenizer import get_tokenizer

from module.align_model import AlignModel
from data_processor.record import Record, read_data_from_csv, read_data_from_json

from utils.audio import load_audio_file, load_MIR1k_audio_file
from utils.alignment import get_mae

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '-f','--test-data',
        type=str,
        required=True,
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
        "--language",
        type=str,
        default='zh',
        help="Transcribe language"
    )
    parser.add_argument(
        '--is-mir1k',
        type=int,
        default=0,
        help="0 => No;\n1 => yes, mixture;\n2 => yes, vocal"
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
                      'output_dim': len(tokenizer)}

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

def get_actual_timestamp(token, tokenizer):
    assert token >= tokenizer.timestamp_begin and token <= tokenizer.timestamp_begin + 1500
    return float(tokenizer.decode_with_timestamps([token]).replace('<|', '').replace('|>', ''))

# Extract all onset / offset timestamps tokens
# Raw token format: <|startoftranscript|><|zh|><|transcribe|><|onset_1|><Char_1><|Offset_1|><|onset_2|><Char_2><|Offset_2|>
def extract_timestamp(tokens, tokenizer, start_offset: int=0.0):
    # Remove initial tokens
    tokens = tokens[3: ]
    timestamps = []
    for i in range(0, len(tokens), 3):
        print(i, len(tokens))
        onset = (tokens[i] - tokenizer.timestamp_begin) * 0.02 + start_offset
        offset = (tokens[i + 2] - tokenizer.timestamp_begin) * 0.02 + start_offset
        timestamps.append([onset, offset])

    return timestamps


def inference_segment(model, 
                      tokenizer, 
                      segment_mel, 
                      text,
                      text_idx: int,
                      begin_offset: float=0.0,
                      language: str='zh',
                      device: str='cuda',
                      last_segment: bool=False):
    mel = pad_or_trim(segment_mel, N_FRAMES).unsqueeze(0).to(device)
  
    embed = model.embed_audio(mel)

    tokens = [tokenizer.sot,
            tokenizer.special_tokens[f'<|{language}|>'],
            tokenizer.special_tokens['<|transcribe|>']]
    
    timestamps = []

    last_end_timestamp = 0.0
    while text_idx < len(text):
        try:
            # Start Timestamp
            logits = model.logits(tokens=torch.tensor(tokens).unsqueeze(0).to(device),
                                        audio_features=embed,)
            eot_prob = logits[-1, -1, tokenizer.eot]
            start_prob, start_idx = torch.max(logits[-1, -1, tokenizer.timestamp_begin: ], dim=-1)
            # print(eot_prob)
            # print(timestamp_prob)
            # output = torch.argmax(logits, dim=-1).squeeze(0).tolist()
            if last_segment != True and eot_prob > start_prob:
                break
            tokens.append(start_idx.item() + tokenizer.timestamp_begin)

            # Insert Character
            tokens += tokenizer.encode(text[text_idx])

            # End Timestamp
            logits = model.logits(tokens=torch.tensor(tokens).unsqueeze(0).to(device),
                                        audio_features=embed,)
            eot_prob = logits[-1, -1, tokenizer.eot]
            end_prob, end_idx = torch.max(logits[-1, -1, tokenizer.timestamp_begin: ], dim=-1)
            # output = torch.argmax(logits, dim=-1).squeeze(0).tolist()
            if last_segment != True and eot_prob > start_prob:
                break
            tokens.append(end_idx.item() + tokenizer.timestamp_begin)

            text_idx += 1
            last_end_timestamp = end_idx.item() * 0.02

            timestamps.append([start_idx.item() * 0.02 + begin_offset,
                            end_idx.item() * 0.02 + begin_offset])

            # Check if start timestamp equals to end timestamp
            # if start_idx == end_idx:
            #     break
            # Check if end timestamp reaches the end timestamp (29.98 / 30.00s)
            if last_segment != True and end_idx >= 1499:
                break
        except:
            break
    
    return timestamps, tokens, text_idx

@torch.no_grad()
def inference_with_timestamps(
        model: whisper.Whisper,
        tokenizer,
        records: List[Record],
        language: str='zh',
        is_mir1k: int=0,
        device: str='cuda'
):
    pred_timestamps = []
    target_timestamps = []
    for record in tqdm(records):
        audio_path = record.audio_path
        # song_id = Path(audio_path).stem
        text = record.text
        
        target_timestamps.append(record.lyric_onset_offset)

        if is_mir1k == 0:
            audio = load_audio_file(audio_path)['speech']
        elif is_mir1k == 1:
            audio = load_MIR1k_audio_file(audio_path, mixture=True)['speech']
        elif is_mir1k == 2:
            audio = load_MIR1k_audio_file(audio_path, mixture=False)['speech']
        else:
            raise ValueError
        
        mel = log_mel_spectrogram(audio)

        curr_frame = 0
        curr_text_idx = 0
        curr_timestamp = 0.0

        timestamps = []
        while curr_frame < mel.shape[-1]:
            segment_mel = mel[:, curr_frame: curr_frame + N_FRAMES]
            segment_frames = min(segment_mel.shape[-1], N_FRAMES)

            last_segment = (curr_frame + segment_frames) >= mel.shape[-1]

            segment_timestamps, segment_tokens, text_idx = inference_segment(
                model,
                tokenizer,
                segment_mel,
                text,
                curr_text_idx,
                curr_timestamp,
                device=device,
                last_segment=last_segment
            )
            # print(tokenizer.decode_with_timestamps(segment_tokens))
            # print(text_idx)
            # print(segment_timestamps)

            timestamps += segment_timestamps

            curr_frame += segment_frames
            curr_text_idx = text_idx
            curr_timestamp += segment_frames / 100

        
        # print(curr_frame)

        # Some Postprocess
        audio_len = mel.shape[-1] / 100
        for i in range(len(timestamps)):
            timestamps[i][0] = min(timestamps[i][0], audio_len)
            timestamps[i][1] = min(timestamps[i][1], audio_len)

        # print(timestamps)

        pred_timestamps.append(timestamps)

        print(len(timestamps))
        print(len(record.lyric_onset_offset))
        # mel = pad_or_trim(mel, N_FRAMES).unsqueeze(0).to(device)
  
        # embed = model.embed_audio(mel)

        # tokens = [tokenizer.sot,whisper_alignment/exp/230619_medium_opencpop_aug_demucs_multitask
        #         tokenizer.special_tokens[f'<|{language}|>'],
        #         tokenizer.special_tokens['<|transcribe|>']]
        # index = 0
        # timestamp = []
        # for char in text:
        #     if index == 0:
        #         tokens.append(tokenizer.timestamp_begin)
        #         tokens += tokenizer.encode(char)
        #     else:
        #         # Inference Start Timestamp
        #         logits = model.logits(tokens=torch.tensor(tokens).unsqueeze(0).to(device),
        #                               audio_features=embed,)
        #         eot_prob = logits[0, -1, tokenizer.eot]
        #         timestamp_prob, _ = torch.max(logits[0, -1, tokenizer.timestamp_begin: ], dim=-1)

        #         if eot_prob > timestamp_prob:whisper_alignment/exp/230619_medium_opencpop_aug_demucs_multitask
        #             tokens.append(tokenizer.eot)
        #             break

        #         output = torch.argmax(logits[:, :, tokenizer.timestamp_begin: ], dim=-1).squeeze(0).tolist()
        #         start_timestamp = output[-1] + tokenizer.timestamp_begin

        #         # logits = model.logits(tokens=torch.tensor(tokens).unsqueeze(0).to(device),
        #         #                         audio_features=embed,)
        #         # output = torch.argmax(logits, dim=-1).squeeze(0).tolist()
        #         # start_timestamp = output[-1]
        #         assert start_timestamp >= tokenizer.timestamp_begin and start_timestamp <= tokenizer.timestamp_begin + 1500
        #         # print(tokenizer.decode_with_timestamps(output))
        #         # Add last token into tokens
        #         tokens.append(start_timestamp)
        #         # Add char to tokens
        #         tokens += tokenizer.encode(char)
        #     # Inference End Timestamp
        #     logits = model.logits(tokens=torch.tensor(tokens).unsqueeze(0).to(device),
        #                         audio_features=embed,)
        #     eot_prob = logits[0, -1, tokenizer.eot]
        #     timestamp_prob, _ = torch.max(logits[0, -1, tokenizer.timestamp_begin: ], dim=-1)

        #     if eot_prob > timestamp_prob:
        #         tokens.append(tokenizer.eot)
        #         break

        #     output = torch.argmax(logits[:, :, tokenizer.timestamp_begin: ], dim=-1).squeeze(0).tolist()
        #     end_timestamp = output[-1] + tokenizer.timestamp_begin
        #     assert end_timestamp >= tokenizer.timestamp_begin and end_timestamp <= tokenizer.timestamp_begin + 1500
        #     tokens.append(end_timestamp)
        #     # print(tokenizer.decode_with_timestamps(output))
            
        #     if index == 0:
        #         timestamp.append([0, get_actual_timestamp(end_timestamp, tokenizer)])
        #     else:
        #         timestamp.append([get_actual_timestamp(start_timestamp, tokenizer),
        #                         get_actual_timestamp(end_timestamp, tokenizer)])
        #     index += 1

        ########################################################

        # for char in text:
        #     # Inference Start Timestamp
        #     logits = model.logits(tokens=torch.tensor(tokens).unsqueeze(0).to(device),
        #                             audio_features=embed,)
        #     output = torch.argmax(logits, dim=-1).squeeze(0).tolist()
        #     start_timestamp = output[-1]
        #     assert start_timestamp >= tokenizer.timestamp_begin and start_timestamp <= tokenizer.timestamp_begin + 1500
        #     # print(tokenizer.decode_with_timestamps(output))
        #     # Add last token into tokens
        #     tokens.append(start_timestamp)
        #     # Add char to tokens
        #     tokens += tokenizer.encode(char)
        #     # Inference End Timestamp
        #     logits = model.logits(tokens=torch.tensor(tokens).unsqueeze(0).to(device),
        #                         audio_features=embed,)
        #     output = torch.argmax(logits, dim=-1).squeeze(0).tolist()
        #     end_timestamp = output[-1]
        #     assert end_timestamp >= tokenizer.timestamp_begin and end_timestamp <= tokenizer.timestamp_begin + 1500
        #     tokens.append(end_timestamp)
        #     # print(tokenizer.decode_with_timestamps(output))
            
        #     timestamp.append([get_actual_timestamp(start_timestamp, tokenizer),
        #                       get_actual_timestamp(end_timestamp, tokenizer)])
        
        # logits = model.logits(tokens=torch.tensor(tokens).unsqueeze(0).to(device),
        #                         audio_features=embed,)
        # output = torch.argmax(logits, dim=-1).squeeze(0).tolist()
        # print(tokenizer.decode_with_timestamps(tokens))
        # pred_timestamps.append(timestamp)
        # # print(timestamp)
        
        # # print(tokenizer.decode_with_timestamps(tokens))
        # # print(record.lyric_onset_offset)
    
    

    print("Average MAE:", get_mae(gt=target_timestamps, predict=pred_timestamps))
    return pred_timestamps
    


def main():
    args = parse_args()

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
    
    tokenizer = get_tokenizer(multilingual=True,
                              language=args.language,
                              task='transcribe')
    transcribe_model.to(device)

    test_records = read_data_from_json(args.test_data)

    inference_with_timestamps(model=transcribe_model,
                              tokenizer=tokenizer,
                              records=test_records,
                              language=args.language,
                              is_mir1k=args.is_mir1k,
                              device=device)


if __name__ == '__main__':
    main()