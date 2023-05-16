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
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from module.align_model import AlignModel
from dataset import get_alignment_dataloader

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
        '--device',
        type=str,
        default='cuda'
    )

    # Training Argument
    parser.add_argument(
        '--train-batch-size',
        type=int,
        default=2
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
        default=2000
    )
    parser.add_argument(
        '--eval-steps',
        type=int,
        default=100
    )
    parser.add_argument(
        '--warmup-steps',
        type=int,
        default=200
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
    loss_fn: dict,
    accum_grad_steps: int,
    max_grad_norm: float,
    vocab_size: int=21128,
) -> Tuple[float, Iterator]:
    model.train()
    total_loss = 0
    total_ce_loss = 0
    total_ctc_loss = 0

    for _ in range(accum_grad_steps):
        # mel, y_text, frame_labels, lyric_word_onset_offset
        mel, y_text, frame_labels, _ = next(train_iter)
        mel, y_text, frame_labels = mel.to(model.device), y_text.to(model.device), frame_labels.to(model.device)

        # Align Logits Shape: [batch size, time length, number of classes] => (N, T, C)
        # align_logits, transcribe_logits
        align_logits, _ = model(mel)

        align_ce_loss = compute_ce_loss(align_logits, frame_labels, loss_fn, device=model.device)
        align_ctc_loss = compute_ctc_loss(align_logits[:, :, : vocab_size], y_text, device=model.device)

        loss = (align_ce_loss + align_ctc_loss) / accum_grad_steps
        loss.backward()

        total_loss += loss.item()
        total_ce_loss += align_ce_loss.item() / accum_grad_steps
        total_ctc_loss += align_ctc_loss.item() / accum_grad_steps

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

    return total_loss, total_ce_loss, total_ctc_loss


@torch.no_grad()
def evaluate(
    model: AlignModel,
    dev_loader: DataLoader,
    loss_fn: dict,
    vocab_size: int=21128,
) -> float:
    model.eval()
    total_loss = 0
    total_ce_loss = 0
    total_ctc_loss = 0

    # mel, y_text, frame_labels, lyric_word_onset_offset
    for mel, y_text, frame_labels, _ in tqdm(dev_loader):
        mel, y_text, frame_labels = mel.to(model.device), y_text.to(model.device), frame_labels.to(model.device)

        # TODO: Add Whisper Evaluate
        # Trainsribe Loss

        # Align Logits Shape: [batch size, time length, number of classes] => (N, T, C)
        # align_logits, transcribe_logits
        align_logits, _ = model(mel)
        
        align_ce_loss = compute_ce_loss(align_logits, frame_labels, loss_fn, device=model.device)
        align_ctc_loss = compute_ctc_loss(align_logits[:, :, : vocab_size], y_text, device=model.device)

        total_loss += align_ce_loss.item() + align_ctc_loss.item()
        total_ce_loss += align_ce_loss.item()
        total_ctc_loss += align_ctc_loss.item()


    total_loss /= len(dev_loader)
    total_ce_loss /= len(dev_loader)
    total_ctc_loss /= len(dev_loader)
    return total_loss, total_ce_loss, total_ctc_loss


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
    loss_fn: dict,
    args: argparse.Namespace,
) -> None:
    min_loss, init_ce_loss, init_ctc_loss = evaluate(model, dev_loader, loss_fn)
    avg_train_loss = 0
    avg_ce_loss = 0
    avg_ctc_loss = 0

    # Force Terminate if no_improve_count >= 3
    no_improve_count = 0

    print(f"Initial loss: {min_loss}, Initial CE loss: {init_ce_loss}, Initial CTC loss: {init_ctc_loss}")
    pbar = tqdm(range(1, args.train_steps + 1))
    train_iter = infinite_iter(train_loader)
    for step in pbar:
        train_loss, train_ce_loss, train_ctc_loss = train_step(
            model,
            train_iter,
            optimizer,
            scheduler,
            loss_fn,
            args.accum_grad_steps,
            args.max_grad_norm,
        )
        pbar.set_postfix({"loss": train_loss,
                          "ce_loss": train_ce_loss,
                          "ctc_loss": train_ctc_loss})
        avg_train_loss += train_loss
        avg_ce_loss += train_ce_loss
        avg_ctc_loss += train_ctc_loss

        if step % args.eval_steps == 0:
            eval_loss, eval_ce_loss, eval_ctc_loss = evaluate(model, dev_loader, loss_fn)

            tqdm.write(f"Step {step}: valid loss={eval_loss}, valid CE loss={eval_ce_loss}, valid CTC loss={eval_ctc_loss}")
            tqdm.write(f"Step {step}: train loss={avg_train_loss / args.eval_steps}, train CE loss={avg_ce_loss / args.eval_steps}, train CTC loss={avg_ctc_loss / args.eval_steps}")
            
            avg_train_loss = 0
            avg_ce_loss = 0
            avg_ctc_loss = 0
        
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
            #     print("No improve, forced terminated.")
            #     break

def compute_ce_loss(
    logits: torch.Tensor, 
    frame_labels: torch.Tensor, 
    loss_fn: dict,
    vocab_size: int=21128,
    device: str='cuda'
) -> torch.Tensor:
    frame_labels = frame_labels[:, : logits.shape[1]]
    if frame_labels.shape[1] < logits.shape[1]:
        frame_labels = torch.cat((frame_labels, 
                                  torch.full((frame_labels.shape[0], logits.shape[1] - frame_labels.shape[1]), 
                                             fill_value=-100, 
                                             device=device)), 
                                  dim=1)
        
    # loss = ce_loss(logits.permute(0, 2, 1), frame_labels)
    frame_labels[frame_labels != -100] -= 1

    word_ce_loss = loss_fn['ce_loss'](logits[:, :, 1: vocab_size].transpose(1, 2), frame_labels)

    silence_label = torch.where(frame_labels == -100, 1, 0)
    silence_ce_loss = loss_fn['silence_ce_loss'](logits[:, :, vocab_size], silence_label.float())

    return word_ce_loss + silence_ce_loss
    
def compute_ctc_loss(
    logits,
    labels,
    device: str='cuda',
):
    output_log_sm = F.log_softmax(logits, dim=2)
    output_log_sm = output_log_sm.transpose(0, 1)

    # print (output_log_sm.shape, labels.shape)

    input_lengths = torch.full(size=(output_log_sm.shape[1],), fill_value=output_log_sm.shape[0], dtype=torch.long).to(device)
    target_length = torch.sum(labels != -100, dim=1)
    # print (target_length)

    cur_ctc_loss = F.ctc_loss(output_log_sm, labels, input_lengths, target_length)
    return cur_ctc_loss


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
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    align_model = AlignModel(whisper_model=whisper_model,
                             embed_dim=WHISPER_DIM[args.whisper_model],
                             output_dim=len(tokenizer) + 1,
                             freeze_encoder=args.freeze_encoder,
                             train_alignment=True,
                             device=device).to(device)
    
    if args.freeze_encoder:
        optimizer = torch.optim.AdamW([{'params': align_model.align_rnn.parameters(), 'lr': args.lr}],
                                        lr=args.lr,
                                        weight_decay=2e-5)
    else:
        optimizer = torch.optim.AdamW([{'params': align_model.align_rnn.parameters(), 'lr': args.lr,},
                                       {'params': align_model.whisper_model.parameters(), 'lr': args.lr / 100}],
                                        lr=args.lr,
                                        weight_decay=2e-5)

    scheduler = scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.train_steps
    )

    # assert os.path.exists(args.train_data)
    train_dataloader = get_alignment_dataloader(data_path=args.train_data,
                                                tokenizer=tokenizer,
                                                batch_size=args.train_batch_size,
                                                shuffle=True)
    dev_dataloader = get_alignment_dataloader(data_path=args.dev_data,
                                              tokenizer=tokenizer,
                                              batch_size=args.dev_batch_size,
                                              shuffle=False)


    loss_fn = {'ce_loss': nn.CrossEntropyLoss(),
               'silence_ce_loss': nn.BCEWithLogitsLoss()}

    main_loop(
        model=align_model,
        train_loader=train_dataloader,
        dev_loader=dev_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        args=args
    )


if __name__ == "__main__":
    main()