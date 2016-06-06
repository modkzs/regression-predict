# -*- coding: utf-8 -*-
from . import global_setting
import numpy as np
__author__ = 'yixuanhe'


def get(x):
    return x


def extract_feature(filename, feature, sp=" ", get_value=get, exclude=[]):
    """
    This is the basic frame work for data extract and test. When you think it's no need to deal with this data, please
    return True in the func.

    :param filename: train file or test file location
    :param feature: The BasicFeature class, used to extract feature
    :param sp: data separator
    :param get_value: the function extract data from different format like 1:2, 2 and so on
    :param exclude: the feature that we don't want to use
    :param predict: the predict function, no need to use when get training data.You can add output to time
                    series in this function
    :return:
    """

    flag = True
    X = []
    Y = []
    with open(filename) as f:
        for l in f.readlines():
            data = l.replace(" \n", "").split(sp)
            data = [get_value(d) for d in data]

            data, control = feature.value_func(data, flag)

            if not control:
                continue

            data, control = feature.line_func(data, flag)
            if not control:
                continue

            data, control = feature.global_func(data)
            if not control:
                continue

            control = feature.time_func(data)
            if not control:
                continue

            row = []
            length = len(data)
            for i in range(global_setting.begin, length):
                if i in exclude:
                    continue
                row.append(data[i])

            y = data[global_setting.y_pos]

            X.append(row)
            Y.append(y)

    return np.array(X), np.array(Y)


def test_model(filename, feature, predict, sp=" ", get_value=get, exclude=[]):
    """
    This is the basic frame work for data extract and test. When you think it's no need to deal with this data, please
    return True in the func.

    :param filename: train file or test file location
    :param feature: The BasicFeature class, used to extract feature
    :param predict: predict func
    :param sp: data separator
    :param get_value: the function extract data from different format like 1:2, 2 and so on
    :param exclude: the feature that we don't want to use
    :param predict: the predict function, no need to use when get training data.You can add output to time
                    series in this function
    :return:
    """

    Y = []
    flag = False
    with open(filename) as f:
        for l in f.readlines():
            data = l.replace(" \n", "").split(sp)
            data = [get_value(d) for d in data]

            data, control = feature.value_func(data, flag)

            data, control = feature.line_func(data, flag)

            data, control = feature.global_func(data)

            row = []
            length = len(data)
            for i in range(global_setting.begin, length):
                if i in exclude:
                    continue
                row.append(data[i])

            y = predict(row)
            y = feature.deal_predict(y, data[global_setting.song_pos], data[global_setting.y_pos])
            Y.append(y)

    return np.array(Y)


