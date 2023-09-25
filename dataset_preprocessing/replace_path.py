import os
import sys
from pathlib import Path

import json
import pandas as pd

from tqdm import tqdm

def main():
    # argv[1] => data file
    # argv[2] => target directory
    # e.g.
    # Opencpop: python replace_path.py opencpop_train.json ./dataset/opencpop
    assert len(sys.argv) == 4
    data_path = sys.argv[1]
    output_path = sys.argv[2]
    target_dir = sys.argv[3]

    with open(data_path, 'r') as f:
        data = json.load(f)

    for i in tqdm(range(len(data))):
        data_basename = data[i]['song_id']
        data[i]['song_path'] = str(Path(target_dir).joinpath(data_basename))
        data[i]['song_path'] = str(Path(data[i]['song_path']).resolve())

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    
if __name__ == "__main__":
    main()