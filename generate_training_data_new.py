from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import numpy as np
import os
import pandas as pd
import torch

DATA_LEN = 15
PRED_LEN = 3

GAME_DATA_LENS = 1000

def generate_graph_seq2seq_io_data(
        actions, actions_prev, x_offsets, y_offsets, add_time_in_day=True, add_day_in_week=False, scaler=None
):
    """
    Generate samples from
    :param df:
    :param x_offsets:
    :param y_offsets:
    :param add_time_in_day:
    :param add_day_in_week:
    :param scaler:
    :return:
    # x1: (epoch_size, input_length, num_nodes, input_dim)  current input data.
    # x2: (epoch_size, input_length, num_nodes, input_dim)  prev input data
    # y: (epoch_size, output_length, num_nodes, output_dim)
    """

    data = np.concatenate(actions, axis=0)
    data_prev = np.concatenate(actions_prev, axis=0)

    x1, x2, y = data[:, 0:DATA_LEN-PRED_LEN, :], data_prev[:, 0:DATA_LEN-PRED_LEN, :], data[:, DATA_LEN-PRED_LEN:DATA_LEN, :]
    x1 = np.expand_dims(x1, axis=3)
    x2 = np.expand_dims(x2, axis=3)
    y = np.expand_dims(y, axis=3)

    '''
    # t is the index of the last observation. todo
    min_t = abs(min(x_offsets))
    max_t = abs(len(actions) - abs(max(y_offsets)))  # Exclusive
    for t in range(min_t, max_t):
        x_t = data[t + x_offsets, ...]
        y_t = data[t + y_offsets, ...]
        x.append(x_t)
        y.append(y_t)
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    '''
    return x1, x2, y

def read_data(fileName, dataNums):
    actions = []  # if start from 3 -> +15 timestep
    actions_prev = []  # then start from 2 -> +15
    for i in range(dataNums):
        data = pd.read_json(path_or_buf=fileName + str(i) + ".json", orient=dict)["actions"]
        lens = len(data)
        action_prev = []
        # don't know if the maximum step is fixed to be 27. Let it be dynamic and not to create problem.
        for j in range(lens):
            action = []
            if j < PRED_LEN + 1:  # check
                continue
            elif j < DATA_LEN:
                for k in range(DATA_LEN - j):
                    action.append(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
                for k in range(j):
                    action.append(np.array(data[k][0]))
            else:
                for k in range(DATA_LEN):
                    action.append(np.array(data[j + k - DATA_LEN][0]))  # check
            actions.append(np.expand_dims(np.asarray(action), axis=0))
            if len(action_prev) == 0:
                actions_prev.append(np.expand_dims(np.zeros((DATA_LEN, 11)), axis=0))
            else:
                actions_prev.append(np.expand_dims(np.asarray(action_prev), axis=0))
            action_prev = action
    print("the shape of whole dataset is", np.concatenate(actions, axis=0).shape)
    print("the shape of whole prev dataset is", np.concatenate(actions_prev, axis=0).shape)
    return actions, actions_prev

def generate_train_val_test(args):

    actions, actions_prev = read_data(args.action_filename, GAME_DATA_LENS)

    x_offsets = np.concatenate((np.arange(-11, 1, 1),))   #
    # Predict the next one hour
    # y_offsets = np.sort(np.arange(1, 13, 1))  # 第一个参数为起点第二个参数为终点（终点不输出）第三个参数为步长
    y_offsets = np.arange(1, 3, 1)
    # x: (num_samples, input_length, num_nodes, input_dim)
    # y: (num_samples, output_length, num_nodes, output_dim)
    x1, x2, y = generate_graph_seq2seq_io_data(
        actions,
        actions_prev,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        # add_time_in_day=True,
        # add_day_in_week=False,
    )

    print("x1 shape: ", x1.shape, "x2 shape: ", x2.shape, ", y shape: ", y.shape)    #输出矩阵维度和长度
    # Write the data into npz file.
    # num_test = 6831, using the last 6831 examples as testing.
    # for the rest: 7/8 is used for training, and 1/8 is used for validation.
    num_samples = x1.shape[0]
    num_test = round(num_samples * 0.2)
    num_train = round(num_samples * 0.7)
    num_val = num_samples - num_test - num_train

    # train
    x1_train, x2_train, y_train = x1[:num_train], x2[:num_train], y[:num_train]
    # val
    x1_val, x2_val, y_val = (
        x1[num_train: num_train + num_val],
        x2[num_train: num_train + num_val],
        y[num_train: num_train + num_val],
    )
    # val add adv agent data in val
    # adv_actions, adv_actions_prev = read_data(args.adv_action_filename, GAME_DATA_LENS)
    # adv_x1, adv_x2, adv_y = generate_graph_seq2seq_io_data(
    #     adv_actions,
    #     adv_actions_prev,
    #     x_offsets=x_offsets,
    #     y_offsets=y_offsets
    # )
    # x1_val, x2_val, y_val = (
    #     adv_x1[: num_val],
    #     adv_x2[: num_val],
    #     adv_y[: num_val],
    # )

    # test
    x1_test, x2_test, y_test = x1[-num_test:], x2[-num_test:], y[-num_test:]

    # temp:
    temp_actions, temp_actions_prev = read_data(args.action_filename, 1)
    temp_x1, temp_x2, temp_y = generate_graph_seq2seq_io_data(
        temp_actions,
        temp_actions_prev,
        x_offsets=x_offsets,
        y_offsets=y_offsets
    )
    x1_temp, x2_temp, y_temp = (
        temp_x1,
        temp_x2,
        temp_y,
    )
    temp_adv_actions, temp_adv_actions_prev = read_data(args.adv_action_filename, 1)
    temp_adv_x1, temp_adv_x2, temp_adv_y = generate_graph_seq2seq_io_data(
        temp_adv_actions,
        temp_adv_actions_prev,
        x_offsets=x_offsets,
        y_offsets=y_offsets
    )
    x1_temp_adv, x2_temp_adv, y_temp_adv = (
        temp_adv_x1,
        temp_adv_x2,
        temp_adv_y,
    )

    for cat in ["train", "val", "test", "temp", "temp_adv"]:
        _x1, _x2, _y = locals()["x1_" + cat], locals()["x2_" + cat], locals()["y_" + cat]
        print(cat, "x1: ", _x1.shape, "x2: ", _x2.shape, "y:", _y.shape)
        np.savez_compressed(
            os.path.join(args.output_dir, "%s.npz" % cat),
            x1=_x1,
            x2=_x2,
            y=_y,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
        )


def main(args):
    print("Generating training data")
    generate_train_val_test(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", type=str, default="data/MARLActionPrediction/save/", help="Output directory."
    )
    parser.add_argument(
        "--traffic_df_filename",
        type=str,
        default="data/DCRNN/metr-la.h5",
        help="Raw traffic readings.",
    )
    parser.add_argument(
        "--action_filename",
        type=str,
        default="data/MARLActionPrediction/MARLActionPrediction/gamedata/",
        help="Raw normal agent data.",
    )
    parser.add_argument(
        "--adv_action_filename",
        type=str,
        default="data/MARLActionPrediction/MARLActionPrediction/gamedata/",
        help="Raw adversarial agent data.",
    )
    args = parser.parse_args()
    main(args)
