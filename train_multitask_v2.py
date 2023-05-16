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
from torch.nn.utils.rnn import pad_sequence
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
        nargs='+',
        type=str,
        required=True
    )
    parser.add_argument(
        '--dev-data',
        nargs='+',
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
        '--use-ctc-loss',
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

# >>> Batch handling >>>
def rebatch_handler(
    batch,
    is_multitask: bool=False
):
    mel = torch.stack([item[0] for item in batch])
    align_text = torch.stack([item[1] for item in batch])

    if is_multitask:
        frame_labels = pad_sequence([item[2] for item in batch],
                                     batch_first=True,
                                     padding_value=-100)
        lyric_onset_offset = [item[3] for item in batch]
    else:
        frame_labels = None
        lyric_onset_offset = None
    
    decoder_input = torch.stack([item[4] for item in batch])
    decoder_output = torch.stack([item[5] for item in batch])

    return mel, align_text, frame_labels, lyric_onset_offset, decoder_input, decoder_output

def split_batch(batch):
    # multitask batch => alignment + decoder transcript
    # transcript batch => decoder transcript
    multitask_batch = []
    transcript_batch = []

    # base on if frame labels exists
    # frame_labels = item[2] 
    for item in zip(*batch):
        if item[2] is not None:
            multitask_batch.append(item)
        else:
            transcript_batch.append(item)

    if len(multitask_batch) > 0:
        multitask_batch = rebatch_handler(multitask_batch, is_multitask=True)
    else:
        multitask_batch = None
    if len(transcript_batch) > 0:
        transcript_batch = rebatch_handler(transcript_batch, is_multitask=False)
    else:
        transcript_batch = None

    return multitask_batch, transcript_batch
# <<< Batch handling <<<


def train_step(
    model: AlignModel,
    train_iter: Iterator,
    optimizer: torch.optim.Optimizer, 
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    accum_grad_steps: int,
    max_grad_norm: float,
    loss_fn: dict,
    vocab_size: int=21128,
    use_ctc_loss: bool=False
):
    model.train()
    
    device = model.device

    losses = {'total': 0,
            'align_ce': 0,
            'align_ctc': 0,
            'trans_ce': 0}

    for _ in range(accum_grad_steps):
        batch = next(train_iter)
        multitask_batch, transcript_batch = split_batch(batch)

        # mel
        # y_text
        # frame_labels
        # lyric_word_onset_offset
        # decoder_input
        # decoder_output
        if multitask_batch is not None:
            multi_align_logits, multi_trans_logits = model(multitask_batch[0].to(device),
                                                           multitask_batch[4].to(device))
            multi_align_ce_loss = compute_ce_loss(multi_align_logits,
                                                  multitask_batch[2].to(device),
                                                  loss_fn=loss_fn,
                                                  compute_sil=use_ctc_loss,
                                                  device=device)
            if use_ctc_loss:
                multi_align_ctc_loss = compute_ctc_loss(multi_align_logits[:, :, : vocab_size],
                                                        multitask_batch[1].to(device),
                                                        device=device)
            multi_trans_loss = F.cross_entropy(multi_trans_logits.permute(0, 2, 1), multitask_batch[-1].to(device))

            multitask_loss = multi_align_ce_loss + multi_trans_loss
            if use_ctc_loss:
                multitask_loss += multi_align_ctc_loss
        else:
            multi_align_ce_loss = torch.tensor(0, device=device)
            multi_align_ctc_loss = torch.tensor(0, device=device)
            multi_trans_loss = torch.tensor(0, device=device)
            multitask_loss = torch.tensor(0, device=device)

        if transcript_batch is not None:
            _, trans_logits = model(transcript_batch[0].to(device),
                                    transcript_batch[4].to(device))
            transcript_loss = F.cross_entropy(trans_logits.permute(0, 2, 1), transcript_batch[-1].to(device))
        else:
            transcript_loss = torch.tensor(0, device=device)

        loss = (multitask_loss + transcript_loss) / accum_grad_steps
        loss.backward()

        losses['total'] += loss.item()
        losses['align_ce'] += multi_align_ce_loss.item() / accum_grad_steps
        losses['trans_ce'] += (multi_trans_loss.item() + transcript_loss.item()) / accum_grad_steps
        if use_ctc_loss:
            losses['align_ctc'] += multi_align_ctc_loss.item() / accum_grad_steps

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

    return losses


@torch.no_grad()
def evaluate(
    model: AlignModel,
    dev_loader: DataLoader,
    loss_fn: dict,
    vocab_size: int=21128,
    use_ctc_loss: bool=False
):
    device = model.device

    model.eval()
    losses = {'total': 0,
            'align_ce': 0,
            'align_ctc': 0,
            'trans_ce': 0}

    # mel, y_text, frame_labels, lyric_word_onset_offset, decoder_input, decoder_output
    for batch in tqdm(dev_loader):
        multitask_batch, transcript_batch = split_batch(batch)

        # mel
        # y_text
        # frame_labels
        # lyric_word_onset_offset
        # decoder_input
        # decoder_output
        if multitask_batch is not None:
            multi_align_logits, multi_trans_logits = model(multitask_batch[0].to(device),
                                                           multitask_batch[4].to(device))
            multi_align_ce_loss = compute_ce_loss(multi_align_logits,
                                                  multitask_batch[2].to(device),
                                                  loss_fn=loss_fn,
                                                  compute_sil=use_ctc_loss,
                                                  device=device)
            if use_ctc_loss:
                multi_align_ctc_loss = compute_ctc_loss(multi_align_logits[:, :, : vocab_size],
                                                        multitask_batch[1].to(device),
                                                        device=device)
            multi_trans_loss = F.cross_entropy(multi_trans_logits.permute(0, 2, 1), multitask_batch[-1].to(device))

            multitask_loss = multi_align_ce_loss + multi_trans_loss
            if use_ctc_loss:
                multitask_loss += multi_align_ctc_loss
        else:
            multi_align_ce_loss = torch.tensor(0, device=device)
            multi_align_ctc_loss = torch.tensor(0, device=device)
            multi_trans_loss = torch.tensor(0, device=device)
            multitask_loss = torch.tensor(0, device=device)

        if transcript_batch is not None:
            _, trans_logits = model(transcript_batch[0].to(device),
                                    transcript_batch[4].to(device))
            transcript_loss = F.cross_entropy(trans_logits.permute(0, 2, 1), transcript_batch[-1].to(device))
        else:
            transcript_loss = torch.tensor(0, device=device)


        losses['total'] += multitask_loss.item() + transcript_loss.item()
        losses['align_ce'] += multi_align_ce_loss.item()
        losses['trans_ce'] += (multi_trans_loss.item() + transcript_loss.item())
        if use_ctc_loss:
            losses['align_ctc'] += multi_align_ctc_loss.item()
        
    for key in losses.keys():
        losses[key] /= len(dev_loader)

    return losses


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
    init_losses = evaluate(model, 
                           dev_loader, 
                           loss_fn=loss_fn,
                           use_ctc_loss=args.use_ctc_loss)
    
    min_loss = init_losses['total']
    
    avg_losses = {'total': 0,
            'align_ce': 0,
            'align_ctc': 0,
            'trans_ce': 0}

    # Force Terminate if no_improve_count >= 5
    no_improve_count = 0

    if args.use_ctc_loss:
        print(f"Initial loss: {min_loss}, Align CE loss: {init_losses['align_ce']}, Align CTC loss: {init_losses['align_ctc']}, Transcript loss: {init_losses['trans_ce']}")
    else:
        print(f"Initial loss: {min_loss}, Align loss: {init_losses['align_ce']}, Transcript loss: {init_losses['trans_ce']}")
        
    pbar = tqdm(range(1, args.train_steps + 1))
    train_iter = infinite_iter(train_loader)
    for step in pbar:
        train_losses = train_step(
            model,
            train_iter,
            optimizer,
            scheduler,
            args.accum_grad_steps,
            args.max_grad_norm,
            loss_fn=loss_fn,
            use_ctc_loss=args.use_ctc_loss,
        )
        if args.use_ctc_loss:
            pbar.set_postfix({
                "loss": train_losses['total'],
                "align_ce_loss": train_losses['align_ce'],
                "align_ctc_loss": train_losses['align_ctc'],
                "transcript_loss": train_losses['trans_ce']
            })
        else:
            pbar.set_postfix({
                "loss": train_losses['total'],
                "align_ce_loss": train_losses['align_ce'],
                "transcript_loss": train_losses['trans_ce']
            })

        for key in avg_losses.keys():
            avg_losses[key] += train_losses[key]

        if step % args.eval_steps == 0:
            eval_losses = evaluate(
                model, 
                dev_loader, 
                loss_fn=loss_fn,
                use_ctc_loss=args.use_ctc_loss
            )

            if args.use_ctc_loss:
                tqdm.write(f"Step {step}: valid loss={eval_losses['total']}, valid align CE loss={eval_losses['align_ce']}, valid align CTC loss={eval_losses['align_ctc']}, valid transcript loss={eval_losses['trans_ce']}")
                tqdm.write(f"Step {step}: train loss={avg_losses['total'] / args.eval_steps}, train align CE loss={avg_losses['align_ce'] / args.eval_steps}, train align CTC loss={avg_losses['align_ctc'] / args.eval_steps}, train transcript loss={avg_losses['trans_ce'] / args.eval_steps}")
            else:
                tqdm.write(f"Step {step}: valid loss={eval_losses['total']}, valid align CE loss={eval_losses['align_ce']}, valid transcript loss={eval_losses['trans_ce']}")
                tqdm.write(f"Step {step}: train loss={avg_losses['total'] / args.eval_steps}, train align CE loss={avg_losses['align_ce'] / args.eval_steps}, train transcript loss={avg_losses['trans_ce'] / args.eval_steps}")

            # Reset Average Loss
            for key in avg_losses.keys():
                avg_losses[key] = 0

            if eval_losses['total'] < min_loss:
                min_loss = eval_losses['total']
                tqdm.write("Saving The Best Model")
                save_model(model, f"{args.save_dir}/best_model.pt")

            if args.save_all_checkpoints:
                save_model(model, f"{args.save_dir}/step{step}.pt")

            save_model(model, f"{args.save_dir}/last_model.pt")

def compute_ce_loss(
    logits: torch.Tensor, 
    frame_labels: torch.Tensor, 
    loss_fn: dict,
    compute_sil: bool=False,
    vocab_size: int=21128,
    device: str='cuda'
) -> torch.Tensor:
    fill_value = -100 if compute_sil else 0

    frame_labels = frame_labels[:, : logits.shape[1]]
    if frame_labels.shape[1] < logits.shape[1]:
        frame_labels = torch.cat((frame_labels, 
                                  torch.full((frame_labels.shape[0], logits.shape[1] - frame_labels.shape[1]), 
                                             fill_value=fill_value, 
                                             device=device)), 
                                  dim=1)

    if compute_sil == False:    
        loss = F.cross_entropy(logits.permute(0, 2, 1), frame_labels)
        return loss

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
    whisper_tokenizer = get_tokenizer(multilingual=".en" not in args.whisper_model, task="transcribe")
    hf_tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)


    multitask_model = AlignModel(whisper_model=whisper_model,
                             embed_dim=WHISPER_DIM[args.whisper_model],
                             output_dim=len(hf_tokenizer) + args.use_ctc_loss,
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

    # assert os.path.exists(args.train_data)
    train_dataloader = get_multitask_dataloader(
        *args.train_data,
        hf_tokenizer=hf_tokenizer,
        whisper_tokenizer=whisper_tokenizer,
        language=args.language,
        no_timestamps=True,
        batch_size=args.train_batch_size,
        shuffle=True
    )
    dev_dataloader = get_multitask_dataloader(
        *args.dev_data,
        hf_tokenizer=hf_tokenizer,
        whisper_tokenizer=whisper_tokenizer,
        language=args.language,
        no_timestamps=True,
        batch_size=args.dev_batch_size,
        shuffle=False
    )

    loss_fn = {'ce_loss': nn.CrossEntropyLoss(),
            'silence_ce_loss': nn.BCEWithLogitsLoss()}

    main_loop(
        model=multitask_model,
        train_loader=train_dataloader,
        dev_loader=dev_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        args=args
    )


if __name__ == "__main__":
    main()