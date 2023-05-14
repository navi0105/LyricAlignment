import os
import json
import argparse
import random
import numpy as np
from typing import Iterator, Tuple
from tqdm import tqdm
import copy
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import whisper
from whisper.tokenizer import get_tokenizer
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from module.align_model import AlignModel
from dataset import get_multitask_dataloader

os.environ["TOKENIZERS_PARALLELISM"]="false"

def parse_args():
    parser = argparse.ArgumentParser()
    # Data Argument
    parser.add_argument(
        '--train-data',
        type=str,
        required=True
    )
    parser.add_argument(
        '--dev-data',
        type=str
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
        '--freeze-encoder',
        action='store_true'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-5
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

def train_step(
    model: AlignModel,
    train_iter: Iterator,
    optimizer: torch.optim.Optimizer, 
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    accum_grad_steps: int,
    max_grad_norm: float
):
    model.train()
    
    device = model.device

    total_loss = 0
    total_align_ce_loss = 0
    total_align_ctc_loss = 0
    total_transcript_loss = 0
    for _ in range(accum_grad_steps):
        # mel, y_text, frame_labels, lyric_word_onset_offset
        mel, align_text, frame_labels, _, decoder_input, decoder_output = next(train_iter)

        mel = mel.to(device)
        align_text = align_text.to(device)
        frame_labels = frame_labels.to(device)
        decoder_input = decoder_input.to(device)
        decoder_output = decoder_output.to(device)

        # Align Logits Shape: [batch size, time length, number of classes] => (N, T, C)
        # align_logits, transcribe_logits
        align_logits, transcript_logits = model(mel, decoder_input)

        align_ce_loss = compute_ce_loss(align_logits, frame_labels, device=device)
        align_ctc_loss = compute_ctc_loss(align_logits[:, :, : 21128], align_text, device)

        transcript_loss = F.cross_entropy(transcript_logits.permute(0, 2, 1), decoder_output)


        loss = align_ce_loss + transcript_loss
        loss += align_ctc_loss
        
        loss /= accum_grad_steps
        loss.backward()

        total_loss += loss.item()
        total_align_ce_loss += align_ce_loss.item() / accum_grad_steps
        total_align_ctc_loss += align_ctc_loss.item() / accum_grad_steps
            
        total_transcript_loss += transcript_loss.item() / accum_grad_steps

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

    return total_loss, total_align_ce_loss, total_align_ctc_loss, total_transcript_loss


@torch.no_grad()
def evaluate(
    model: AlignModel,
    dev_loader: DataLoader
):
    model.eval()
    total_loss = 0
    total_align_ce_loss = 0
    total_align_ctc_loss = 0
    total_transcript_loss = 0

    # mel, y_text, frame_labels, lyric_word_onset_offset
    for batch in tqdm(dev_loader):
        mel, align_text, frame_labels, _, decoder_input, decoder_output = batch

        mel = mel.to(model.device)
        align_text = align_text.to(model.device)
        frame_labels = frame_labels.to(model.device)
        decoder_input = decoder_input.to(model.device)
        decoder_output = decoder_output.to(model.device)

        # TODO: Add Whisper Evaluate
        # Trainsribe Loss

        # Align Logits Shape: [batch size, time length, number of classes] => (N, T, C)
        # align_logits, transcribe_logits
        align_logits, transcript_logits = model(mel, decoder_input)

        align_ce_loss = compute_ce_loss(align_logits, frame_labels, device=model.device)
        align_ctc_loss = compute_ctc_loss(align_logits[:, :, : 21128], align_text, model.device)

        transcript_loss = F.cross_entropy(transcript_logits.permute(0, 2, 1), decoder_output)

        total_loss += align_ce_loss.item() + transcript_loss.item()
        total_loss += align_ctc_loss

        total_align_ce_loss += align_ce_loss.item()
        total_align_ctc_loss += align_ctc_loss.item()

        total_transcript_loss += transcript_loss.item()


    total_loss /= len(dev_loader)
    total_align_ce_loss /= len(dev_loader)
    total_align_ctc_loss /= len(dev_loader)

    total_transcript_loss /= len(dev_loader)
    return total_loss, total_align_ce_loss, total_align_ctc_loss, total_transcript_loss


def save_model(model, save_path: str) -> None:
    # save model in half precision to save space
    #model = copy.deepcopy(model).half()
    # save model weights and config in a dictionary that can be loaded with `whisper.load_model`
    torch.save(model.state_dict(), save_path)

def main_loop(
    model,
    train_loader: DataLoader,
    dev_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    args: argparse.Namespace,
) -> None:
    min_loss, init_align_loss, init_align_ctc_loss, init_transcript_loss = evaluate(model, dev_loader)
    avg_train_loss = 0
    avg_align_ce_loss = 0
    avg_align_ctc_loss = 0
    avg_transcript_loss = 0
    # Force Terminate if no_improve_count >= 5
    no_improve_count = 0

    print(f"Initial loss: {min_loss}, Align CE loss: {init_align_loss}, Align CTC loss: {init_align_ctc_loss}, Transcript loss: {init_transcript_loss}")
        
    pbar = tqdm(range(1, args.train_steps + 1))
    train_iter = infinite_iter(train_loader)
    for step in pbar:
        train_loss, train_align_ce_loss, train_align_ctc_loss, train_transcript_loss = train_step(
            model,
            train_iter,
            optimizer,
            scheduler,
            args.accum_grad_steps,
            args.max_grad_norm
        )
        pbar.set_postfix({"loss": train_loss,
                          "align_ce_loss": train_align_ce_loss,
                          "align_ctc_loss": train_align_ctc_loss,
                          "transcript_loss": train_transcript_loss})
        
        avg_train_loss += train_loss
        avg_align_ce_loss += train_align_ce_loss
        avg_align_ctc_loss += train_align_ctc_loss
        avg_transcript_loss += train_transcript_loss

        if step % args.eval_steps == 0:
            eval_loss, eval_align_ce_loss, eval_align_ctc_loss, eval_transcript_loss = evaluate(model, dev_loader)

            tqdm.write(f"Step {step}: valid loss={eval_loss}, valid align CE loss={eval_align_ce_loss}, valid align CTC loss={eval_align_ctc_loss}, valid transcript loss={eval_transcript_loss}")
            tqdm.write(f"Step {step}: train loss={avg_train_loss / args.eval_steps}, train align CE loss={avg_align_ce_loss / args.eval_steps}, train align CTC loss={avg_align_ctc_loss / args.eval_steps}, train transcript loss={avg_transcript_loss / args.eval_steps}")
            
            avg_train_loss = 0
            avg_align_ce_loss = 0
            avg_align_ctc_loss = 0
            avg_transcript_loss = 0
        
            if eval_loss < min_loss:
                # Reset no_improve_count
                no_improve_count = 0
                
                min_loss = eval_loss
                tqdm.write("Saving The Best Model")
                save_model(model, f"{args.save_dir}/best_model.pt")
            else:
                no_improve_count += 1
            

            if args.save_all_checkpoints:
                save_model(model, f"{args.save_dir}/step{step}.pt")

            save_model(model, f"{args.save_dir}/last_model.pt")

            # if no_improve_count >= 5:
            #     print("No improve, force terminated.")
            #     break

def compute_ce_loss(
    logits: torch.Tensor, 
    frame_labels: torch.Tensor,
    vocab_size: int=21128,
    device: str='cuda') -> torch.Tensor:
    frame_labels = frame_labels[:, : logits.shape[1]]
    if frame_labels.shape[1] < logits.shape[1]:
        frame_labels = torch.cat((frame_labels, 
                                  torch.full((frame_labels.shape[0], logits.shape[1] - frame_labels.shape[1]), 
                                             fill_value=0, 
                                             device=device)), 
                                  dim=1)
        
    # loss = F.cross_entropy(logits.permute(0, 2, 1), frame_labels)
    frame_labels[frame_labels != -100] = 1
    word_loss = F.cross_entropy(logits[:, :, 1:vocab_size].permute(0, 2, 1), frame_labels)

    silence_labels = torch.where(frame_labels == -100, 1, 0)
    silence_loss = F.binary_cross_entropy_with_logits(logits[:, :, vocab_size], silence_labels.float())

    return word_loss + silence_loss
    
def compute_ctc_loss(
    logits,
    labels,
    device: str='cuda'
):
    logits_log_sm = F.log_softmax(logits, dim=2)

    logit_lengths = torch.full(size=(logits_log_sm.shape[0], ), fill_value=logits_log_sm.shape[1], dtype=torch.long).to(device)
    target_lengths = torch.sum(labels != -100, dim=1)

    loss = F.ctc_loss(logits_log_sm.permute(1, 0, 2), labels, logit_lengths, target_lengths)
    return loss


def main():
    args = parse_args()
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    save_args(args, f"{args.save_dir}/args.json")


    device = args.device
    if 'cuda' in device and torch.cuda.is_available() == False:
        device = 'cpu'

    whisper_model = whisper.load_model(args.whisper_model, device=device)
    whisper_tokenizer = get_tokenizer(multilingual=".en" not in args.whisper_model, task="transcribe")
    hf_tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)


    multitask_model = AlignModel(whisper_model=whisper_model,
                             embed_dim=WHISPER_DIM[args.whisper_model],
                             output_dim=len(hf_tokenizer),
                             freeze_encoder=args.freeze_encoder,
                             train_alignment=True,
                             train_transcribe=True,
                             device=device).to(device)
    
    optimizer = torch.optim.AdamW([{'params': multitask_model.align_rnn.parameters(), 'lr': args.lr,},
                                    {'params': multitask_model.whisper_model.parameters(), 'lr': args.lr / 250}],
                                    lr=args.lr,
                                    weight_decay=2e-5)

    scheduler = scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.train_steps
    )

    assert os.path.exists(args.train_data)
    train_dataloader = get_multitask_dataloader(
        args.train_data,
        hf_tokenizer=hf_tokenizer,
        whisper_tokenizer=whisper_tokenizer,
        language=args.language,
        no_timestamps=True,
        batch_size=args.train_batch_size,
        shuffle=True
    )
    dev_dataloader = get_multitask_dataloader(
        args.dev_data,
        hf_tokenizer=hf_tokenizer,
        whisper_tokenizer=whisper_tokenizer,
        language=args.language,
        no_timestamps=True,
        batch_size=args.dev_batch_size,
        shuffle=False
    )

    main_loop(
        model=multitask_model,
        train_loader=train_dataloader,
        dev_loader=dev_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        args=args
    )


if __name__ == "__main__":
    main()