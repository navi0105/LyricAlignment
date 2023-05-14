from typing import List, Optional
from data_processor.record import Record
import os

import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

from whisper.tokenizer import Tokenizer
from whisper import log_mel_spectrogram, pad_or_trim
from whisper.audio import N_FRAMES

from data_processor.record import read_data_from_json, read_data_from_csv


# ALIGNMENT DATASET
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

def get_alignment_dataloader(
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

# TRANSCRIPTION DATASET
class TranscriptionDataset(Dataset):
    def __init__(
        self,
        records: List[Record],
        tokenizer: Tokenizer,
        language: str='zh',
        fp16: bool=True,
        no_timestamps: bool=True
    ):
        self.records = records
        self.tokenizer = tokenizer
        self.language = language
        self.fp16 = fp16
        self.no_timestamps = no_timestamps

    def _calculate_mel(
        self,
        audio_path: str,
    ) -> torch.Tensor:
        mel = log_mel_spectrogram(audio_path)
        mel = pad_or_trim(mel, N_FRAMES)
        if self.fp16:
            mel = mel.half()

        return mel

    def _get_special_tokens(
            self, 
            is_text_empty: bool, 
            language: str, 
            no_timestamps: bool
    ) -> List[int]:
        if is_text_empty:
            special_tokens = [self.tokenizer.sot, self.tokenizer.no_speech]
        else:
            special_tokens = [
                self.tokenizer.sot,
                self.tokenizer.special_tokens[f"<|{language}|>"],
                self.tokenizer.special_tokens["<|transcribe|>"],
            ]
            if no_timestamps:
                special_tokens.append(self.tokenizer.no_timestamps)

        return special_tokens

    def _encode_text_with_timestamps(
            self,
            text: str,
            lyric_onset_offset: List[List[float]]
    ) -> List[int]:
        
        tokens = []
        for i in range(len(lyric_onset_offset)):
            onset = lyric_onset_offset[i][0]
            offset = lyric_onset_offset[i][1]

            if onset < 0 or onset > 30:
                raise ValueError(f"Invalid timestamp: {onset}")
            if offset < 0 or offset > 30:
                raise ValueError(f"Invalid timestamp: {offset}")

            start_token = self.tokenizer.timestamp_begin + (onset * 100 // 2)
            end_token = self.tokenizer.timestamp_begin + (offset * 100 // 2)
            char_token = self.tokenizer.encode(text[i])

            tokens.append(start_token)
            tokens.extend(char_token)
            tokens.append(end_token)
        
        return tokens


    def _get_text_tokens(
            self,
            record: Record,
            no_timestmaps: bool
    ) -> List[int]:
        if no_timestmaps == False:
            text_tokens = self._encode_text_with_timestamps(record.text, record.lyric_onset_offset)
        else:
            text_tokens = self.tokenizer.encode(record.text)

        return text_tokens
    
    def _construct_decoder_output(
        self,
        special_tokens: List[int],
        text_tokens: List[int]
    ) -> List[int]:
        decoder_output = special_tokens[1:] + text_tokens + [self.tokenizer.eot]
        return decoder_output

    def __len__(self):
        return len(self.records)
    
    def __getitem__(self, index):
        record = self.records[index]
        no_timestamps = self.no_timestamps

        text_tokens = self._get_text_tokens(record, no_timestamps)
        is_text_empty = len(text_tokens) == 0
        special_tokens = self._get_special_tokens(is_text_empty, self.language, self.no_timestamps)

        decoder_input = special_tokens + text_tokens
        decoder_output = self._construct_decoder_output(special_tokens=special_tokens,
                                                        text_tokens=text_tokens)
        mel = self._calculate_mel(record.audio_path)

        return (
            mel,
            torch.tensor(decoder_input, dtype=torch.long),
            torch.tensor(decoder_output, dtype=torch.long)
        )

    def collate_fn(self, data):
        x, y_in, y_out = zip(*data)
        
        x = pad_sequence(x, batch_first=True, padding_value=0)
        y_in = pad_sequence(y_in, batch_first=True, padding_value=0)
        y_out = pad_sequence(y_out, batch_first=True, padding_value=-100)
        
        return x, y_in, y_out
    
def get_transcript_dataloader(
    data_path: str,
    tokenizer: Tokenizer,
    language: str='zh',
    no_timestamps: bool=True,
    batch_size: int=1,
    fp16: bool=True,
    shuffle: bool=False,
) -> DataLoader:
    assert os.path.exists(data_path)
    if os.path.splitext(data_path)[-1] == '.csv':
        records = read_data_from_csv(data_path)
    else:
        records = read_data_from_json(data_path)

    dataset = TranscriptionDataset(records=records,
                                   tokenizer=tokenizer,
                                   language=language,
                                   fp16=fp16,
                                   no_timestamps=no_timestamps)
    
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        collate_fn=dataset.collate_fn,
        pin_memory=False
    )


# MULTITASK DATASET

class MultitaskDataset(Dataset):
    def __init__(
        self,
        records: List[Record],
        hf_tokenizer,
        whisper_tokenizer: Tokenizer,
        language: str='zh',
        no_timestamps: bool=True
    ):
        self.records = records
        self.hf_tokenizer = hf_tokenizer
        self.whisper_tokenizer = whisper_tokenizer
        self.language = language
        self.no_timestamps = no_timestamps

    def _calculate_mel(
        self,
        audio_path: str,
    ) -> torch.Tensor:
        mel = log_mel_spectrogram(audio_path)
        mel = pad_or_trim(mel, N_FRAMES)

        return mel

    def _encode_align_text(
        self,
        text: str
    ) -> torch.Tensor:
        return self.hf_tokenizer.encode(text, add_special_tokens=False, return_tensors='pt').view(-1)

    def _get_special_tokens(
            self, 
            is_text_empty: bool, 
            language: str, 
            no_timestamps: bool
    ) -> List[int]:
        if is_text_empty:
            special_tokens = [self.whisper_tokenizer.sot, self.whisper_tokenizer.no_speech]
        else:
            special_tokens = [
                self.whisper_tokenizer.sot,
                self.whisper_tokenizer.special_tokens[f"<|{language}|>"],
                self.whisper_tokenizer.special_tokens["<|transcribe|>"],
            ]
            if no_timestamps:
                special_tokens.append(self.whisper_tokenizer.no_timestamps)

        return special_tokens
    
    def _encode_text_with_timestamps(
            self,
            text: str,
            lyric_onset_offset: List[List[float]]
    ) -> List[int]:
        
        tokens = []
        for i in range(len(lyric_onset_offset)):
            onset = lyric_onset_offset[i][0]
            offset = lyric_onset_offset[i][1]

            if onset < 0 or onset > 30:
                raise ValueError(f"Invalid timestamp: {onset}")
            if offset < 0 or offset > 30:
                raise ValueError(f"Invalid timestamp: {offset}")

            start_token = self.whisper_tokenizer.timestamp_begin + (onset * 100 // 2)
            end_token = self.whisper_tokenizer.timestamp_begin + (offset * 100 // 2)
            char_token = self.whisper_tokenizer.encode(text[i])

            tokens.append(start_token)
            tokens.extend(char_token)
            tokens.append(end_token)
        
        return tokens

    def _get_transcript_tokens(
            self,
            record: Record,
            no_timestmaps: bool
    ) -> List[int]:
        if no_timestmaps == False:
            text_tokens = self._encode_text_with_timestamps(record.text, record.lyric_onset_offset)
        else:
            text_tokens = self.whisper_tokenizer.encode(record.text)

        return text_tokens

    def _construct_decoder_output(
        self,
        special_tokens: List[int],
        text_tokens: List[int]
    ) -> List[int]:
        decoder_output = special_tokens[1:] + text_tokens + [self.whisper_tokenizer.eot]

        return decoder_output


    def __len__(self):
        return len(self.records)
    
    def __getitem__(self, index):
        record = self.records[index]

        mel = self._calculate_mel(record.audio_path)
        
        # Alignment Data
        align_text_tokens = self._encode_align_text(record.text)
        if record.lyric_onset_offset is not None:     
            lyric_onset_offset = record.lyric_onset_offset
        else:
            lyric_onset_offset = None

        # Transcription Data
        no_timestamps = self.no_timestamps
        transcript_text_tokens = self._get_transcript_tokens(record, no_timestamps)
        is_text_empty = len(transcript_text_tokens) == 0
        special_tokens = self._get_special_tokens(is_text_empty, self.language, self.no_timestamps)

        decoder_input = special_tokens + transcript_text_tokens
        decoder_output = self._construct_decoder_output(special_tokens=special_tokens,
                                                        text_tokens=transcript_text_tokens)
        
        return (
            mel,
            align_text_tokens,
            lyric_onset_offset,
            torch.tensor(decoder_input, dtype=torch.long),
            torch.tensor(decoder_output, dtype=torch.long)
        )
    
    def get_frame_label(
        self, 
        lyric_tokens, 
        lyric_word_onset_offset,
        hop_size_second: float=0.02
    ):
        total_frame_num = int(round(lyric_word_onset_offset[-1][-1] / hop_size_second)) + 1
        frame_labels = torch.full((total_frame_num,), -100)

        for j in range(len(lyric_word_onset_offset)):
            onset_frame = int(round(lyric_word_onset_offset[j][0] / hop_size_second))
            offset_frame = int(round(lyric_word_onset_offset[j][1] / hop_size_second)) + 1
            frame_labels[onset_frame:offset_frame] = lyric_tokens[j]

        return frame_labels

    def batch_get_frame_label(
        self, 
        lyric_tokens,
        lyric_word_onset_offset,
        hop_size_second: float=0.02
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
        mel, align_text, lyric_onset_offset, decoder_input, decoder_output = zip(*data)

        mel = pad_sequence(mel, batch_first=True, padding_value=0)

        # Align Token
        align_text = pad_sequence(align_text, batch_first=True, padding_value=0)
        align_text[align_text == 0] = -100
        align_text[align_text == 102] == -100

        # frame_labels = []
        # for i in range(len(data)):
        #     if lyric_onset_offset[i] is not None:
        #         frame_labels.append(self.get_frame_label(align_text[i], lyric_onset_offset[i]))
        #     else:
        #         frame_labels.append(None)
        frame_labels = self.batch_get_frame_label(align_text, lyric_onset_offset)
        
        # Transcript Token
        decoder_input = pad_sequence(decoder_input, batch_first=True, padding_value=0)
        decoder_output = pad_sequence(decoder_output, batch_first=True, padding_value=-100)

        return mel, align_text, frame_labels, lyric_onset_offset, decoder_input, decoder_output


def get_multitask_dataloader(
    *data_paths,
    hf_tokenizer,
    whisper_tokenizer: Tokenizer,
    language: str='zh',
    no_timestamps: bool=True,
    batch_size: int=1,
    shuffle: bool=False
):
    records = []
    for path in data_paths:
        assert os.path.exists(path)
        if os.path.splitext(path)[-1] == '.csv':
            records.extend(read_data_from_csv(path))
        else:
            records.extend(read_data_from_json(path))

    dataset = MultitaskDataset(
        records=records,
        hf_tokenizer=hf_tokenizer,
        whisper_tokenizer=whisper_tokenizer,
        language=language,
        no_timestamps=no_timestamps
    )

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=dataset.collate_fn,
        num_workers=4,
        pin_memory=False
    )