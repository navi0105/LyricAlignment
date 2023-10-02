from typing import List, Optional
from data_processor.record import Record
import os

import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

from whisper.tokenizer import Tokenizer
from whisper import log_mel_spectrogram, pad_or_trim
from whisper.audio import N_FRAMES, load_audio

from data_processor.record import read_data

from utils.audio import load_audio_file, load_mixture_audio_file

# Whisper Dataset
class TranscriptDataset(Dataset):
    def __init__(
        self,
        records: List[Record],
        whisper_tokenizer,
    ):
        self.records = records
        self.whisper_tokenizer = whisper_tokenizer

    def _calculate_mel(
        self,
        audio_path: str,
    ) -> torch.Tensor:
        audio = load_audio_file(audio_path)['speech']

        mel = log_mel_spectrogram(audio)
        mel = pad_or_trim(mel, N_FRAMES)

        return mel

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
        if no_timestmaps == False and record.lyric_onset_offset is not None:
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
        pass

class AlignDataset(Dataset):
    def __init__(
        self,
        records: List[Record],
        hf_tokenizer,
        use_ctc: bool=False,
    ):
        self.records = records
        self.hf_tokenizer = hf_tokenizer
        self.use_ctc = use_ctc

    def _encode_text(
        self,
        text: str
    ) -> torch.Tensor:
        return self.hf_tokenizer.encode(text, 
                                        add_special_tokens=False, 
                                        return_tensors='pt').view(-1)

    def _get_frame_label(
        self, 
        lyric_tokens, 
        lyric_word_onset_offset,
        hop_size_second: float=0.02
    ):
        fill_value = -100 if self.use_ctc else 0

        total_frame_num = int(round(lyric_word_onset_offset[-1][-1] / hop_size_second)) + 1
        frame_labels = torch.full((total_frame_num,), fill_value=fill_value)

        for j in range(len(lyric_word_onset_offset)):
            onset_frame = int(round(lyric_word_onset_offset[j][0] / hop_size_second))
            offset_frame = int(round(lyric_word_onset_offset[j][1] / hop_size_second)) + 1
            frame_labels[onset_frame:offset_frame] = lyric_tokens[j]

        return frame_labels

    def __len__(self):
        return len(self.records)
    
    def __getitem__(self, index):
        pass


class MultitaskDatasetFinal(TranscriptDataset, AlignDataset):
    def __init__(
        self, 
        records: List[Record],
        whisper_tokenizer, 
        hf_tokenizer, 
        language: str='zh',
        no_timestamps: bool=True,
        use_ctc: bool=False):
        # Inherit from TranscriptDataset and AlignDataset
        TranscriptDataset.__init__(self, 
                                records=records, 
                                whisper_tokenizer=whisper_tokenizer)
        AlignDataset.__init__(self, 
                              records=records, 
                              hf_tokenizer=hf_tokenizer, 
                              use_ctc=use_ctc)

        self.records = records
        self.whisper_tokenizer = whisper_tokenizer
        self.hf_tokenizer = hf_tokenizer
        self.language = language
        self.no_timestamps = no_timestamps

    def __len__(self):
        return len(self.records)
    
    def __getitem__(self, index):
        record = self.records[index]

        audio = load_audio(record.audio_path, sr=16000)

        text = record.text

        if record.lyric_onset_offset is not None:
            lyric_onset_offset = record.lyric_onset_offset
        else:
            lyric_onset_offset = None

        no_timestamps = self.no_timestamps
        tanscript_text_tokens = self._get_transcript_tokens(record, no_timestamps)
        is_text_empty = len(tanscript_text_tokens) == 0
        special_tokens = self._get_special_tokens(is_text_empty, self.language, self.no_timestamps)

        decoder_input = special_tokens + tanscript_text_tokens
        decoder_output = self._construct_decoder_output(special_tokens=special_tokens,
                                                        text_tokens=tanscript_text_tokens)
        
        return (
            audio,
            text,
            lyric_onset_offset,
            torch.tensor(decoder_input, dtype=torch.long),
            torch.tensor(decoder_output, dtype=torch.long)
        )

    def collate_fn(self, data):
        audio, align_text, lyric_onset_offset, decoder_input, decoder_output = zip(*data)
        
        align_text_tokens = self.hf_tokenizer(align_text,
                                                padding=True,
                                                return_tensors='pt')['input_ids'][:, 1:]
        
        align_text_tokens[align_text_tokens == 0] = -100
        align_text_tokens[align_text_tokens == 102] = -100

        frame_labels = []
        for i in range(len(data)):
            if lyric_onset_offset[i] is not None:
                frame_labels.append(self._get_frame_label(align_text_tokens[i], lyric_onset_offset[i]))
            else:
                frame_labels.append(None)

        decoder_input = pad_sequence(decoder_input, batch_first=True, padding_value=0)
        decoder_output = pad_sequence(decoder_output, batch_first=True, padding_value=-100)

        return audio, align_text_tokens, frame_labels, lyric_onset_offset, decoder_input, decoder_output

def get_multitask_dataloader(
    *data_paths,
    hf_tokenizer,
    whisper_tokenizer,
    language: str='zh',
    no_timestamps: bool=True,
    use_ctc: bool=False,
    batch_size: int=1,
    shuffle: bool=False,
) -> DataLoader:
    records = []
    for path in data_paths:
        records.extend(read_data(path))

    dataset = MultitaskDatasetFinal(
        records=records,
        hf_tokenizer=hf_tokenizer,
        whisper_tokenizer=whisper_tokenizer,
        language=language,
        no_timestamps=no_timestamps,
        use_ctc=use_ctc
    )

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        collate_fn=dataset.collate_fn,
        pin_memory=True
    )