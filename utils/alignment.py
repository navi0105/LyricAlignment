import numpy as np
from pypinyin import lazy_pinyin, Style

import torch
import torch.nn.functional as F
from typing import List

from numba import jit

import sys
sys.path.insert(0, '..')

def perform_viterbi(prediction, labels, hop_size_second=0.02):
    log_prediction = F.log_softmax(prediction, dim=2)

    silence_prediction = log_prediction[:,:,0:1]

    log_prediction = torch.clip(log_prediction, min=-1000)[:,:,1:]

    log_silence_prediction = torch.clip(silence_prediction, min=-1000)

    predicted_onset_offset = []

    for i in range(log_prediction.shape[0]):
        cur_label = np.array([labels[i][j] for j in range(len(labels[i])) if labels[i][j] != -100])
        # print (cur_label)
        # blank row: dp_matrix[-1]
        dp_matrix = np.array([[-10000000.0 for k in range(len(cur_label) * 2 + 1)] for j in range(log_prediction.shape[1])])

        backtrace_dp_matrix = np.array([[0 for k in range(len(cur_label) * 2 + 1)] for j in range(log_prediction.shape[1])])

        cur_log_prediction = log_prediction[i].numpy()
        cur_log_silence_prediction = log_silence_prediction[i].numpy()

        dp_matrix[0][0] = cur_log_silence_prediction[0][0]
        dp_matrix[0][1] = cur_log_prediction[0][cur_label[0] - 1]

        # print (time.time())
        dp_matrix, backtrace_dp_matrix = run_viterbi_core(dp_matrix, backtrace_dp_matrix, cur_log_prediction, cur_log_silence_prediction, cur_label)
        # print (time.time())
        if dp_matrix[-1][-1] > dp_matrix[-1][-2]:
            # Go backtrace

            # Get dp_matrix.shape[1] but dp_matrix is not numpy array XD
            correct_path = [len(dp_matrix[0]) - 1, ]
            cur_k = backtrace_dp_matrix[-1][-1]
            # print (backtrace_dp_matrix[-1][-1])
            for j in range(len(dp_matrix)-2, -1, -1):
                correct_path.append(cur_k)
                cur_k = backtrace_dp_matrix[j][cur_k]
        else:

            correct_path = [len(dp_matrix[0]) - 2, ]
            cur_k = backtrace_dp_matrix[-1][-2]
            # print (backtrace_dp_matrix[-1][-2])
            for j in range(len(dp_matrix)-2, -1, -1):
                correct_path.append(cur_k)
                cur_k = backtrace_dp_matrix[j][cur_k]

        correct_path.reverse()

        cur_predicted_onset_offset = []
        cur_pos = 0
        # print (correct_path)
        for k in range(len(cur_label)):
            first_index = correct_path.index(k * 2 + 1)
            last_index = len(correct_path) - correct_path[::-1].index(k * 2 + 1) - 1
            cur_predicted_onset_offset.append([float(first_index) * hop_size_second, float(last_index + 1) * hop_size_second])
        
        predicted_onset_offset.append(list(cur_predicted_onset_offset))
    return predicted_onset_offset

@jit(nopython=True)
def run_viterbi_core(dp_matrix, backtrace_dp_matrix, cur_log_prediction, cur_log_silence_prediction, cur_label):
    # print (cur_label.shape[0] * 2 + 1, log_prediction.shape[1])
    for j in range(1, cur_log_prediction.shape[0]):
        for k in range(cur_label.shape[0] * 2 + 1):
            if k == 0:
                # blank
                backtrace_dp_matrix[j][k] = k
                # dp_matrix[j][k] = dp_matrix[j-1][k] + log_prediction[i][j][0]
                dp_matrix[j][k] = dp_matrix[j-1][k] + cur_log_silence_prediction[j][0]

            elif k == 1:
                if dp_matrix[j-1][k] > dp_matrix[j-1][k-1]:
                    backtrace_dp_matrix[j][k] = k
                    dp_matrix[j][k] = dp_matrix[j-1][k] + cur_log_prediction[j][cur_label[0] - 1]
                else:
                    backtrace_dp_matrix[j][k] = k - 1
                    dp_matrix[j][k] = dp_matrix[j-1][k-1] + cur_log_prediction[j][cur_label[0] - 1]

            elif k % 2 == 0:
                # blank
                if dp_matrix[j-1][k] > dp_matrix[j-1][k-1]:
                    backtrace_dp_matrix[j][k] = k
                    # dp_matrix[j][k] = dp_matrix[j-1][k] + log_prediction[i][j][0]
                    dp_matrix[j][k] = dp_matrix[j-1][k] + cur_log_silence_prediction[j][0]
                else:
                    backtrace_dp_matrix[j][k] = k - 1
                    # dp_matrix[j][k] = dp_matrix[j-1][k-1] + log_prediction[i][j][0]
                    dp_matrix[j][k] = dp_matrix[j-1][k-1] + cur_log_silence_prediction[j][0]

            else:
                if (dp_matrix[j-1][k-2] >= dp_matrix[j-1][k-1] and dp_matrix[j-1][k-2] >= dp_matrix[j-1][k] 
                    and cur_label[k // 2] != cur_label[k // 2 - 1]):
                    # k-2 (last character) -> k
                    backtrace_dp_matrix[j][k] = k - 2
                    dp_matrix[j][k] = dp_matrix[j-1][k-2] + cur_log_prediction[j][cur_label[k // 2] - 1]

                elif dp_matrix[j-1][k] > dp_matrix[j-1][k-1]:
                    # k -> k
                    backtrace_dp_matrix[j][k] = k
                    dp_matrix[j][k] = dp_matrix[j-1][k] + cur_log_prediction[j][cur_label[k // 2] - 1]
                else:
                    # k-1 -> k
                    backtrace_dp_matrix[j][k] = k - 1
                    dp_matrix[j][k] = dp_matrix[j-1][k-1] + cur_log_prediction[j][cur_label[k // 2] - 1]

    return dp_matrix, backtrace_dp_matrix

def perform_viterbi_ctc(prediction, labels, hop_size_second=0.02):

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
        cur_label = np.array([labels[i][j] for j in range(len(labels[i])) if labels[i][j] != -100])
        # print (cur_label)
        # blank row: dp_matrix[-1]
        dp_matrix = np.array([[-10000000.0 for k in range(len(cur_label) * 2 + 1)] for j in range(log_prediction.shape[1])])

        backtrace_dp_matrix = np.array([[0 for k in range(len(cur_label) * 2 + 1)] for j in range(log_prediction.shape[1])])

        cur_log_prediction = log_prediction[i].numpy()
        cur_log_silence_prediction = log_silence_prediction[i].numpy()

        dp_matrix[0][0] = cur_log_silence_prediction[0][0]
        dp_matrix[0][1] = cur_log_prediction[0][cur_label[0] - 1]

        # print (time.time())
        dp_matrix, backtrace_dp_matrix = run_viterbi_core(dp_matrix, backtrace_dp_matrix, cur_log_prediction, cur_log_silence_prediction, cur_label)
        # print (time.time())
        if dp_matrix[-1][-1] > dp_matrix[-1][-2]:
            # Go backtrace

            # Get dp_matrix.shape[1] but dp_matrix is not numpy array XD
            correct_path = [len(dp_matrix[0]) - 1, ]
            cur_k = backtrace_dp_matrix[-1][-1]
            # print (backtrace_dp_matrix[-1][-1])
            for j in range(len(dp_matrix)-2, -1, -1):
                correct_path.append(cur_k)
                cur_k = backtrace_dp_matrix[j][cur_k]
        else:

            correct_path = [len(dp_matrix[0]) - 2, ]
            cur_k = backtrace_dp_matrix[-1][-2]
            # print (backtrace_dp_matrix[-1][-2])
            for j in range(len(dp_matrix)-2, -1, -1):
                correct_path.append(cur_k)
                cur_k = backtrace_dp_matrix[j][cur_k]

        correct_path.reverse()
        # print (correct_path)

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