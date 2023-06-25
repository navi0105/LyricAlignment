import os
import json
import argparse
import random
import numpy as np
from typing import Iterator, Tuple
from tqdm import tqdm
import itertools
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import whisper
from whisper.audio import log_mel_spectrogram, pad_or_trim, N_FRAMES
from whisper.tokenizer import get_tokenizer
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from module.align_model import AlignModel
from dataset import get_transcript_dataloader, get_multitask_dataloader

def parse_args():
    parser = argparse.ArgumentParser()

    # Data Argument
    parser.add_argument(
        '--train-data',
        nargs='+',
        required=True
    )
    parser.add_argument(
        '--dev-data',
        nargs='+'
    )

    # Model Argument
    parser.add_argument(
        '--whisper-model',
        type=str,
        default='large'
    )
    parser.add_argument(
        '--tokenizer',
        type=str,
        default='bert-base-chinese'
    )
    parser.add_argument(
        '--language',
        type=str,
        default='zh'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda'
    )
    parser.add_argument(
        '--align-model-dir',
        type=str,
        default=None,
        help="use this argument for training existed alignment model"
    )

    # Training Argument
    parser.add_argument(
        '--train-batch-size',
        type=int,
        default=4
    )
    parser.add_argument(
        '--dev-batch-size',
        type=int,
        default=16
    )
    parser.add_argument(
        '--accum-grad-steps',
        type=int,
        default=8
    )
    parser.add_argument(
        '--no-timestamps',
        action='store_true'
    )
    parser.add_argument(
        '--freeze-encoder',
        action='store_true'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=5e-6
    )
    parser.add_argument(
        '--fp16',
        type=bool,
        default=True
    )
    parser.add_argument(
        '--max-grad-norm',
        type=float,
        default=1.0
    )
    parser.add_argument(
        '--train-steps',
        type=int,
        default=1000
    )
    parser.add_argument(
        '--eval-steps',
        type=int,
        default=100
    )
    parser.add_argument(
        '--warmup-steps',
        type=int,
        default=100
    )

    parser.add_argument(
        '--save-dir',
        type=str,
        default='result'
    )
    parser.add_argument(
        '--save-all-checkpoints',
        type=bool,
        default=False
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

def save_args(args: argparse.Namespace, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(vars(args), indent=4, ensure_ascii=False))

def infinite_iter(data_loader: DataLoader) -> Iterator:
    while True:
        for batch in data_loader:
            yield batch

def load_model(
    model_dir: str,
    args,
    device: str='cuda'
) -> AlignModel:
    assert os.path.exists(model_dir)
    with open(os.path.join(model_dir, 'args.json'), 'r') as f:
        model_args = json.load(f)

    tokenizer_name = model_args['tokenizer']
    whisper_model_name = model_args['whisper_model']
    model_path = os.path.join(model_dir, 'best_model.pt')

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    whisper_tokenizer = get_tokenizer(multilingual='.en' not in whisper_model_name,
                                      task='transcribe')
    whisper_model = whisper.load_model(whisper_model_name, device=device)
    
    model = AlignModel(whisper_model=whisper_model,
                       embed_dim=WHISPER_DIM[whisper_model_name],
                       output_dim=len(tokenizer) + model_args['use_ctc_loss'],
                       freeze_encoder=True,
                       train_alignment=False,
                       train_transcribe=True,
                       device=device)
    
    if model_path is not None:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict=state_dict)
    
    args.tokenizer = tokenizer_name
    args.whisper_model = whisper_model_name

    return model, tokenizer, whisper_tokenizer, args

def train_step(
    model: AlignModel,
    train_iter: Iterator,
    optimizer: torch.optim.Optimizer, 
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    accum_grad_steps: int,
    max_grad_norm: float,
) -> Tuple[float, Iterator]:
    model.train()
    total_loss = 0

    for _ in range(accum_grad_steps):
        # mel, y_text, frame_labels, lyric_word_onset_offset
        audios, _, _, _, decoder_input, decoder_output = next(train_iter)
        decoder_input, decoder_output = decoder_input.to(model.device), decoder_output.to(model.device)

        audios = np.stack((itertools.zip_longest(*audios, fillvalue=0)), axis=1).astype('float32')
        mel = log_mel_spectrogram(audios).to(model.device)
        mel = pad_or_trim(mel, N_FRAMES)

        # Align Logits Shape: [batch size, time length, number of classes] => (N, T, C)
        # align_logits, transcribe_logits
        _, transcribe_logits = model(mel, decoder_input)

        transcribe_loss = F.cross_entropy(transcribe_logits.permute(0, 2, 1),
                                          decoder_output)

        loss = transcribe_loss / accum_grad_steps
        loss.backward()

        total_loss += transcribe_loss.item() / accum_grad_steps

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

    return total_loss

@torch.no_grad()
def evaluate(
    model: AlignModel, 
    dev_loader: DataLoader
) -> float:
    model.eval()
    total_loss = 0

    # mel, y_text, frame_labels, lyric_word_onset_offset
    for batch in tqdm(dev_loader):
        audios, _, _, _, decoder_input, decoder_output = batch
        decoder_input, decoder_output = decoder_input.to(model.device), decoder_output.to(model.device)

        audios = np.stack((itertools.zip_longest(*audios, fillvalue=0)), axis=1).astype('float32')
        mel = log_mel_spectrogram(audios).to(model.device)
        mel = pad_or_trim(mel, N_FRAMES)
        # Align Logits Shape: [batch size, time length, number of classes] => (N, T, C)
        # align_logits, transcribe_logits
        _, transcribe_logits = model(mel, decoder_input)
        
        transcribe_loss = F.cross_entropy(transcribe_logits.permute(0, 2, 1), decoder_output)

        total_loss += transcribe_loss.item()


    total_loss /= len(dev_loader)
    return total_loss

def main_loop(
    model,
    train_loader: DataLoader,
    dev_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    args: argparse.Namespace,
) -> None:
    min_loss = evaluate(model, dev_loader)
    avg_train_loss = 0
    
    print(f"Initial loss: {min_loss}")
    pbar = tqdm(range(1, args.train_steps + 1))
    train_iter = infinite_iter(train_loader)
    for step in pbar:
        train_loss = train_step(
            model,
            train_iter,
            optimizer,
            scheduler,
            args.accum_grad_steps,
            args.max_grad_norm,
        )
        pbar.set_postfix({"loss": train_loss})
        avg_train_loss += train_loss

        if step % args.eval_steps == 0:
            eval_loss = evaluate(model, dev_loader)

            tqdm.write(f"Step {step}: valid loss={eval_loss}")
            tqdm.write(f"Step {step}: train loss={avg_train_loss / args.eval_steps}")
            
            avg_train_loss = 0
        
            if eval_loss < min_loss:
                min_loss = eval_loss
                tqdm.write("Saving The Best Model")
                save_model(model, f"{args.save_dir}/best_model.pt")

            if args.save_all_checkpoints:
                save_model(model, f"{args.save_dir}/step{step}.pt")

            save_model(model, f"{args.save_dir}/last_model.pt")

def save_model(model, save_path: str) -> None:
    # save model in half precision to save space
    #model = copy.deepcopy(model).half()
    # save model weights and config in a dictionary that can be loaded with `whisper.load_model`
    torch.save(model.state_dict(), save_path)

def main():
    args = parse_args()
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    device = args.device
    if 'cuda' in device and torch.cuda.is_available() == False:
        device = 'cpu'

    if args.align_model_dir is not None:
        assert os.path.exists(args.align_model_dir)
        model, hf_tokenizer, whisper_tokenizer, args = load_model(
            model_dir=args.align_model_dir,
            args=args,
            device=device
        )
        
        # save_args(model_args, f"{args.save_dir}/args.json")
    else:
        hf_tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        whisper_tokenizer = get_tokenizer(multilingual=".en" not in args.whisper_model, task="transcribe")
        whisper_model = whisper.load_model(args.whisper_model,
                                           device=device)
        model = AlignModel(whisper_model=whisper_model,
                           embed_dim=WHISPER_DIM[args.whisper_model],
                           output_dim=len(hf_tokenizer),
                           freeze_encoder=args.freeze_encoder,
                           train_alignment=False,
                           train_transcript=True,
                           device=device)
    
    
    save_args(args, f"{args.save_dir}/args.json")
    

    print('freeze encoder', model.freeze_encoder)
    model.to(device)
    # Move rnn to cpu for reduce Vram usage
    model.align_rnn.to('cpu')

    optimizer = torch.optim.AdamW(model.whisper_model.parameters(),
                                  lr=args.lr,
                                  weight_decay=2e-5)


    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.train_steps
    )

    train_dataloader = get_multitask_dataloader(*args.train_data,
                                                hf_tokenizer=hf_tokenizer,
                                                whisper_tokenizer=whisper_tokenizer,
                                                no_timestamps=args.no_timestamps,
                                                use_v2_dataset=True,
                                                batch_size=args.train_batch_size,
                                                shuffle=True)
    
    dev_dataloader = get_multitask_dataloader(*args.dev_data,
                                                hf_tokenizer=hf_tokenizer,
                                                whisper_tokenizer=whisper_tokenizer,
                                                no_timestamps=args.no_timestamps,
                                                use_v2_dataset=True,
                                                batch_size=args.train_batch_size,
                                                shuffle=True)
    
    main_loop(
        model=model,
        train_loader=train_dataloader,
        dev_loader=dev_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        args=args
    )


if __name__ == "__main__":
    main()