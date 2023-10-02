import os
import json
from typing import List, Optional
from dataclasses import dataclass
from ast import literal_eval
import pandas as pd

@dataclass
class Record:
    '''
    a Dataclass for storing audio information
    audio_path: audio's absolute path
    text: audio transcription
    lyric_onset_offset: Optional, a list of list, each sublist contains two elements, the first one is the onset time of a word, the second one is the offset time of a word
                        e.g. [[0.0, 0.5], [0.5, 1.0], [1.0, 1.5]]
    '''
    audio_path: str
    text: str
    lyric_onset_offset: Optional[list]=None


def read_data(
        data_path: str,
    ) -> List[Record]:
    assert os.path.exists(data_path)
    with open(data_path, 'r') as f:
        data_list = json.load(f)

    records = []
    for data in data_list:
        record = Record(audio_path=data['song_path'],
                        text=data['lyric'])
            
        if 'on_offset' in data:
            record.lyric_onset_offset = data['on_offset']            

        records.append(record)

    return records