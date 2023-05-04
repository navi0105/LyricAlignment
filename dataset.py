from typing import List, Optional
from data_processor.record import Record
import os

import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

from whisper import log_mel_spectrogram, pad_or_trim
from whisper.audio import N_FRAMES

from data_processor.record import read_data_from_json, read_data_from_csv

class AlignmentDataset(Dataset):
    def __init__(
        self,
        records: List[Record],
        tokenizer,
    ) -> None:
        self.records = records
        self.tokenizer = tokenizer


    def _calculate_mel(
        self,
        audio_path: str,
    ) -> torch.Tensor:
        mel = log_mel_spectrogram(audio_path)
        mel = pad_or_trim(mel, N_FRAMES)

        return mel

    def _encode_text(
        self,
        text: str
    ) -> torch.Tensor:
        return self.tokenizer.encode(text, add_special_tokens=False, return_tensors='pt').view(-1)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]

        mel = self._calculate_mel(record.audio_path)
        text_token = self._encode_text(record.text)
        lyric_onset_offset = record.lyric_onset_offset

        return (mel, text_token, lyric_onset_offset)
    
    def get_frame_label(
        self, 
        lyric_tokens, 
        lyric_word_onset_offset, 
        hop_size_second=0.02
    ):
        total_frame_num = max([lyric_word_onset_offset[i][-1][-1] for i in range(len(lyric_word_onset_offset))])
        total_frame_num = int(round(total_frame_num / hop_size_second)) + 1

        frame_labels = torch.full((len(lyric_word_onset_offset), total_frame_num), 0)

        for i in range(len(lyric_word_onset_offset)):
            for j in range(len(lyric_word_onset_offset[i])):
                onset_frame = int(round(lyric_word_onset_offset[i][j][0] / hop_size_second))
                offset_frame = int(round(lyric_word_onset_offset[i][j][1] / hop_size_second)) + 1
                frame_labels[i][onset_frame: offset_frame] = lyric_tokens[i][j]

        return frame_labels

    def collate_fn(self, data):
        x, y_text, lyric_word_onset_offset = zip(*data)

        x = pad_sequence(x, batch_first=True, padding_value=0)
        y_text = pad_sequence(y_text, batch_first=True, padding_value=0)

        y_text[y_text == 0] = -100
        y_text[y_text == 102] = -100

        frame_labels = self.get_frame_label(y_text, lyric_word_onset_offset)
        
        return x, y_text, frame_labels, lyric_word_onset_offset

def get_dataloader(
    data_path: str,
    tokenizer,
    batch_size: int=1,
    shuffle: bool=False
    ) -> DataLoader:
    assert os.path.exists(data_path)
    if os.path.splitext(data_path)[-1] == '.csv':
        records = read_data_from_csv(data_path)
    else:
        records = read_data_from_json(data_path)

    dataset = AlignmentDataset(records=records,
                               tokenizer=tokenizer)
    
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=dataset.collate_fn,
        num_workers=4,
        pin_memory=True,
    )


class TranscriptionDataset(Dataset):
    def __init__(
        self,
        records: List[Record],
    ) -> None:
        self.records = records

    def __len__(self):
        pass
    
    def __getitem__(self, index):
        pass