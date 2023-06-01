import numpy as np
from pypinyin import lazy_pinyin, Style

import torch
import torch.nn.functional as F
from typing import List

from transformers import AutoTokenizer

import sys
sys.path.insert(0, '..')
from data_processor.record import read_data_from_json

def perform_viterbi(prediction, labels, hop_size_second=0.02):

    log_prediction = F.log_softmax(prediction, dim=2)

    prediction = torch.clip(prediction, min=-1000)
    # print (log_prediction.shape, labels.shape)

    predicted_onset_offset = []

    for i in range(prediction.shape[0]):
        cur_label = [labels[i][j] for j in range(len(labels[i])) if labels[i][j] != -100]
        # print (cur_label)
        # blank row: dp_matrix[-1]
        dp_matrix = torch.full((prediction.shape[1], len(cur_label) * 2 + 1), -10000000.0)
        # blank first
        dp_matrix[0][0] = prediction[i][0][0]
        dp_matrix[0][1] = prediction[i][0][cur_label[0]]

        dp_path = [[i,] for i in range(len(cur_label) * 2 + 1)]
        new_dp_path = [[] for i in range(len(cur_label) * 2 + 1)]

        for j in range(1, prediction.shape[1]):
            for k in range(len(cur_label) * 2 + 1):
                if k == 0:
                    # blank
                    new_dp_path[k] = list(dp_path[k])
                    new_dp_path[k].append(k)
                    dp_matrix[j][k] = dp_matrix[j-1][k] + prediction[i][j][0]
                elif k == 1:
                    if dp_matrix[j-1][k] > dp_matrix[j-1][k-1]:
                        new_dp_path[k] = list(dp_path[k])
                        new_dp_path[k].append(k)
                        dp_matrix[j][k] = dp_matrix[j-1][k] + prediction[i][j][cur_label[0]]
                    else:
                        new_dp_path[k] = list(dp_path[k-1])
                        new_dp_path[k].append(k)
                        dp_matrix[j][k] = dp_matrix[j-1][k-1] + prediction[i][j][cur_label[0]]

                elif k % 2 == 0:
                    # blank
                    if dp_matrix[j-1][k] > dp_matrix[j-1][k-1]:
                        new_dp_path[k] = list(dp_path[k])
                        new_dp_path[k].append(k)
                        dp_matrix[j][k] = dp_matrix[j-1][k] + prediction[i][j][0]
                    else:
                        new_dp_path[k] = list(dp_path[k-1])
                        new_dp_path[k].append(k)
                        dp_matrix[j][k] = dp_matrix[j-1][k-1] + prediction[i][j][0]

                else:
                    if dp_matrix[j-1][k] > dp_matrix[j-1][k-1] and dp_matrix[j-1][k] > dp_matrix[j-1][k-2]:
                        new_dp_path[k] = list(dp_path[k])
                        new_dp_path[k].append(k)
                        dp_matrix[j][k] = dp_matrix[j-1][k] + prediction[i][j][cur_label[k // 2]]

                    elif dp_matrix[j-1][k-1] > dp_matrix[j-1][k] and dp_matrix[j-1][k-1] > dp_matrix[j-1][k-2]:
                        # k-1 -> k
                        new_dp_path[k] = list(dp_path[k-1])
                        new_dp_path[k].append(k)
                        dp_matrix[j][k] = dp_matrix[j-1][k-1] + prediction[i][j][cur_label[k // 2]]
                    else:
                        # k-2 (last character) -> k
                        new_dp_path[k] = list(dp_path[k-2])
                        new_dp_path[k].append(k)
                        dp_matrix[j][k] = dp_matrix[j-1][k-2] + prediction[i][j][cur_label[k // 2]]

            dp_path = list(new_dp_path)

        if dp_matrix[-1][-1] > dp_matrix[-1][-2]:
            correct_path = dp_path[-1]
            # print (dp_matrix[-1][-1] / len(correct_path), correct_path)
        else:
            correct_path = dp_path[-2]
            # print (dp_matrix[-1][-2] / len(correct_path), correct_path)

        cur_predicted_onset_offset = []
        cur_pos = 0
        for k in range(len(cur_label)):
            first_index = correct_path.index(k * 2 + 1)
            last_index = len(correct_path) - correct_path[::-1].index(k * 2 + 1) - 1
            cur_predicted_onset_offset.append([float(first_index) * hop_size_second, float(last_index + 1) * hop_size_second])
        
        predicted_onset_offset.append(list(cur_predicted_onset_offset))
    return predicted_onset_offset

def perform_viterbi_sil(prediction, labels, hop_size_second=0.02):

    log_prediction = F.log_softmax(prediction[:,:,1:-1], dim=2)

    silence_prediction = F.sigmoid(prediction[:,:,-1:])
    voiced_prediction = 1.0 - silence_prediction

    log_silence_prediction = torch.log(silence_prediction)
    log_voiced_prediction = torch.log(voiced_prediction)

    log_prediction = log_prediction + log_voiced_prediction
    log_prediction = torch.clip(log_prediction, min=-1000)

    log_silence_prediction = torch.clip(log_silence_prediction, min=-1000)
    # print (log_prediction.shape, labels.shape)
    # print (log_silence_prediction)

    predicted_onset_offset = []

    for i in range(log_prediction.shape[0]):
        cur_label = [labels[i][j] for j in range(len(labels[i])) if labels[i][j] != -100]
        # print (cur_label)
        # blank row: dp_matrix[-1]
        dp_matrix = torch.full((log_prediction.shape[1], len(cur_label) * 2 + 1), -10000000.0)
        # blank first
        # dp_matrix[0][0] = log_prediction[i][0][0]
        dp_matrix[0][0] = log_silence_prediction[i][0][0]
        dp_matrix[0][1] = log_prediction[i][0][cur_label[0] - 1]

        # print (dp_matrix[0,0:2])

        # print (prediction[i,:,0])

        dp_path = [[i,] for i in range(len(cur_label) * 2 + 1)]
        new_dp_path = [[] for i in range(len(cur_label) * 2 + 1)]

        for j in range(1, log_prediction.shape[1]):
            for k in range(len(cur_label) * 2 + 1):
                if k == 0:
                    # blank
                    new_dp_path[k] = list(dp_path[k])
                    new_dp_path[k].append(k)
                    # dp_matrix[j][k] = dp_matrix[j-1][k] + log_prediction[i][j][0]
                    dp_matrix[j][k] = dp_matrix[j-1][k] + log_silence_prediction[i][j][0]

                elif k == 1:
                    if dp_matrix[j-1][k] > dp_matrix[j-1][k-1]:
                        new_dp_path[k] = list(dp_path[k])
                        new_dp_path[k].append(k)
                        dp_matrix[j][k] = dp_matrix[j-1][k] + log_prediction[i][j][cur_label[0] - 1]
                    else:
                        new_dp_path[k] = list(dp_path[k-1])
                        new_dp_path[k].append(k)
                        dp_matrix[j][k] = dp_matrix[j-1][k-1] + log_prediction[i][j][cur_label[0] - 1]

                elif k % 2 == 0:
                    # blank
                    if dp_matrix[j-1][k] > dp_matrix[j-1][k-1]:
                        new_dp_path[k] = list(dp_path[k])
                        new_dp_path[k].append(k)
                        # dp_matrix[j][k] = dp_matrix[j-1][k] + log_prediction[i][j][0]
                        dp_matrix[j][k] = dp_matrix[j-1][k] + log_silence_prediction[i][j][0]
                    else:
                        new_dp_path[k] = list(dp_path[k-1])
                        new_dp_path[k].append(k)
                        # dp_matrix[j][k] = dp_matrix[j-1][k-1] + log_prediction[i][j][0]
                        dp_matrix[j][k] = dp_matrix[j-1][k-1] + log_silence_prediction[i][j][0]

                else:
                    if (dp_matrix[j-1][k-2] >= dp_matrix[j-1][k-1] and dp_matrix[j-1][k-2] >= dp_matrix[j-1][k] 
                        and cur_label[k // 2] != cur_label[k // 2 - 1]):
                        # k-2 (last character) -> k
                        new_dp_path[k] = list(dp_path[k-2])
                        new_dp_path[k].append(k)
                        dp_matrix[j][k] = dp_matrix[j-1][k-2] + log_prediction[i][j][cur_label[k // 2] - 1]

                    elif dp_matrix[j-1][k] > dp_matrix[j-1][k-1]:
                        # k -> k
                        new_dp_path[k] = list(dp_path[k])
                        new_dp_path[k].append(k)
                        dp_matrix[j][k] = dp_matrix[j-1][k] + log_prediction[i][j][cur_label[k // 2] - 1]

                    else:
                        # k-1 -> k
                        new_dp_path[k] = list(dp_path[k-1])
                        new_dp_path[k].append(k)
                        dp_matrix[j][k] = dp_matrix[j-1][k-1] + log_prediction[i][j][cur_label[k // 2] - 1]
                        

            dp_path = list(new_dp_path)
            # if j < 5:
            #     print (dp_path, dp_matrix[j])

        if dp_matrix[-1][-1] > dp_matrix[-1][-2]:
            correct_path = dp_path[-1]
            # print (dp_matrix[-1][-1] / len(correct_path), correct_path)
        else:
            correct_path = dp_path[-2]
            # print (dp_matrix[-1][-2] / len(correct_path), correct_path)

        cur_predicted_onset_offset = []
        cur_pos = 0
        # print (correct_path)
        for k in range(len(cur_label)):
            first_index = correct_path.index(k * 2 + 1)
            last_index = len(correct_path) - correct_path[::-1].index(k * 2 + 1) - 1
            cur_predicted_onset_offset.append([float(first_index) * hop_size_second, float(last_index + 1) * hop_size_second])
        
        predicted_onset_offset.append(list(cur_predicted_onset_offset))
    return predicted_onset_offset

def get_mae(gt, predict):
    error = 0.0
    cnt = 0
    for i in range(len(gt)):
        for j in range(len(gt[i])):
            error = error + abs(gt[i][j][0] - predict[i][j][0]) + abs(gt[i][j][1] - predict[i][j][1])
            cnt = cnt + 2.0

    error = error / cnt
    return error

def get_pinyin_table(tokenizer):
    def handle_error(chars):
        return ['bad', 'bad']

    tokens = tokenizer.convert_ids_to_tokens(np.arange(0, len(tokenizer), 1).astype(int))
    # print (tokens)
    token_pinyin = []
    pinyin_reverse = {}
    for i in range(len(tokens)):
        try:
            cur_pinyin = lazy_pinyin(tokens[i], style=Style.NORMAL, errors=handle_error)
        except:
            cur_pinyin = ['bad', 'bad']
        if len(cur_pinyin) == 1:
            token_pinyin.append(cur_pinyin[0])
            if cur_pinyin[0] not in pinyin_reverse.keys():
                pinyin_reverse[cur_pinyin[0]] = [i,]
            else:
                pinyin_reverse[cur_pinyin[0]].append(i)
        else:
            token_pinyin.append('bad')

    return token_pinyin, pinyin_reverse

def pypinyin_reweight(
    logits: torch.Tensor,
    labels,
    token_pinyin,
    pinyin_reverse,
):
    pinyin_reverse_keys = list(pinyin_reverse.keys())

    cur_same_pronun_token = []
    for k in range(len(pinyin_reverse_keys)):
        cur_same_pronun_token.append(torch.tensor(pinyin_reverse[pinyin_reverse_keys[k]]))

    for i in range(len(logits)):

        effective_pronun = []
        for k in range(len(labels[i])):
            # print(labels[i][k])
            if labels[i][k] == -100:
                continue

            # print (labels[i][k], token_pinyin[labels[i][k]])
            try:
                cur_key = token_pinyin[labels[i][k]]
                cur_key_index = pinyin_reverse_keys.index(cur_key)
                if cur_key_index not in effective_pronun:
                    effective_pronun.append(cur_key_index)
            except ValueError:
                print(labels[i][k])
                print(token_pinyin[labels[i][k]])

        for j in range(len(logits[i])):
            cur_frame_best = torch.max(logits[i][j])
            # for k in range(len(pinyin_reverse_keys)):
            for k in effective_pronun:
                # selected = torch.index_select(logits[i][j], dim=0, index=cur_same_pronun_token[k])
                cur_value_list = cur_same_pronun_token[k]
                selected = logits[i][j][cur_value_list]
                # print (selected.shape)
                cur_max = torch.max(selected)

                logits[i][j][cur_value_list] = (cur_max * 4.0 + logits[i][j][cur_value_list]) / 5.0

    return logits

# def batch_get_frame_label(
#         lyric_tokens,
#         lyric_word_onset_offset,
#         hop_size_second: float=0.02
#     ):
#         fill_value = -100
#         # fill_value = -100

#         total_frame_num = max([lyric_word_onset_offset[i][-1][-1] for i in range(len(lyric_word_onset_offset))])
#         total_frame_num = int(round(total_frame_num / hop_size_second)) + 1

#         frame_labels = torch.full((len(lyric_word_onset_offset), total_frame_num), fill_value=fill_value)

#         for i in range(len(lyric_word_onset_offset)):
#             for j in range(len(lyric_word_onset_offset[i])):
#                 onset_frame = int(round(lyric_word_onset_offset[i][j][0] / hop_size_second))
#                 offset_frame = int(round(lyric_word_onset_offset[i][j][1] / hop_size_second)) + 1
#                 frame_labels[i][onset_frame: offset_frame] = lyric_tokens[i][j]

#         return frame_labels

# def get_ce_weight(
#     data_path: str,
#     tokenizer,
# ):
#     records = read_data_from_json(data_path)
#     freq = torch.full((len(tokenizer),), 0.001)
#     for i in range(len(records)):
#         if not hasattr(records[i], "lyric_onset_offset"):
#             continue

#         target_transcription = [records[i].text]
#         labels = tokenizer(target_transcription, 
#                            padding=True, 
#                            return_tensors="pt").input_ids[:,1:]

#         labels[labels == 0] = -100
#         labels[labels == 102] = -100

#         lyric_word_onset_offset = [records[i].lyric_onset_offset]
#         frame_labels = batch_get_frame_label(labels, lyric_word_onset_offset)

#         for j in range(len(frame_labels)):
#             for k in range(len(frame_labels[j])):
#                 freq[int(frame_labels[j][k])] += 1

#     freq = freq / torch.sum(freq)

#     return 1.0 / freq

        