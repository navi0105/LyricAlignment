import argparse
import evaluate
import os
import pandas as pd
import json
from typing import List

from utils.CER import CER
from utils.alignment import get_mae_v2

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", "--result-file",
        type=str,
        required=True,
        help="Json file"
    )
    parser.add_argument(
        "--ref-text-key",
        type=str,
        default='lyric',
        help=""
    )
    parser.add_argument(
        "--pred-text-key",
        type=str,
        default='inference',
        help=""
    )
    parser.add_argument(
        '--ref-timestamp-key',
        type=str,
        default='onset_offset'
    )
    parser.add_argument(
        '--pred-timestamp-key',
        type=str,
        default='inference_onset_offset'
    )

    args = parser.parse_args()
    return args

def compute_cer(
    reference: List[str], 
    prediction: List[str]):
    CER_weighted = 0.0
    op_count = {'substitution': 0,
                'insertion': 0,
                'deletion': 0}
    for ref, pred in zip(reference, prediction):
        try:
            cer, nb_map = CER(hypothesis=list(pred),
                            reference=list(ref))

        except:
            cer, nb_map = CER(hypothesis=[],
                              reference=list(ref))
            
        CER_weighted += cer
        op_count['substitution'] += nb_map['S']
        op_count['insertion'] += nb_map['I']
        op_count['deletion'] += nb_map['D']
    
    print('=' * 30)
    print("CER (Weighted):", CER_weighted / len(reference))
    print("Wrong Operations:")
    for key, value in op_count.items():
        print(f"{key}: {value}")
    print('-' * 30)
    # weighted evaluate
    metric = evaluate.load("cer")
    CER_unweighted = metric.compute(references=reference,
                                  predictions=prediction)
    
    print("CER (Unweighted):", CER_unweighted)
    print("=" * 30)

def main():
    args = parse_args()

    assert os.path.exists(args.result_file)
    with open(args.result_file, 'r') as f:
        results = json.load(f)

    # CER
    compute_cer(reference=[result[args.ref_text_key] for result in results],
                prediction=[result[args.pred_text_key] for result in results])
    
    avg_mae = get_mae_v2(gt=[result[args.ref_timestamp_key] for result in results],
                         predict=[result[args.pred_timestamp_key] for result in results])
    print("Average MAE:", avg_mae)

if __name__ == "__main__":
    main()