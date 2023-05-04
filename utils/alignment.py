import torch
import torch.nn.functional as F

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

def get_mae(gt, predict):
    error = 0.0
    cnt = 0
    for i in range(len(gt)):
        for j in range(len(gt[i])):
            error = error + abs(gt[i][j][0] - predict[i][j][0]) + abs(gt[i][j][1] - predict[i][j][1])
            cnt = cnt + 2.0

    error = error / cnt
    return error