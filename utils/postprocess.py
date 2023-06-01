import argparse
from opencc import OpenCC
from tqdm import tqdm
import json

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-file", '-f',
        nargs='+'
    )

    args = parser.parse_args()
    return args

# Remove all english characters
def remove_english(s: str):
    processed_str = ''
    for ch in s:
        if (ord(ch) >= ord('a') and ord(ch) <= ord('z')) or (ord(ch) >= ord('A') and ord(ch) <= ord('Z')):
            continue
        processed_str += ch
    return processed_str

def main():
    args = parse_args()

    cc = OpenCC('t2s')
    for file in args.input_file:
        with open(file, 'r') as f:
            data = json.load(f)
        
        for i in tqdm(range(len(data))):
            data[i]['inference'] = cc.convert(data[i]['inference']).replace(' ', '')
            data[i]['inference'] = remove_english(data[i]['inference'])

        with open(file, 'w') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
            


if __name__ == "__main__":
    main()