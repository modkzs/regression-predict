# -*- coding: utf-8 -*-
from sklearn.svm import SVR
import xgboost as xgb
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
import math
import time
from MultiAdaBoostRegressor import MultiAdaBoostRegressor
__author__ = 'yixuanhe'


def get_avg(path):
    avg = {}

    with open(path) as f:
        for l in f.readlines():
            data = l.split(" ")
            if data[6] > '20150701':
                songid = data[1]
                play = int(data[2])
                avg[songid] = avg.get(songid, 0) + play

    for key in avg:
        avg[key] = avg[key]/31.0

    return avg


def load_data(path, avg={}):
    X = []
    Y = []
    weight = []

    with open(path) as f:
        for l in f.readlines():
            row = []
            data = l.split(" ")
            songid = data[1]
            Y.append(int(data[2]) - avg.get(songid, 0))

            data[6] = time.mktime(time.strptime(data[6]+" 0:00:00", '%Y%m%d %H:%M:%S'))
            data[7] = time.mktime(time.strptime(data[7]+" 0:00:00", '%Y%m%d %H:%M:%S'))

            for d in data[3:]:
                row.append(float(d))
            X.append(row)
            weight.append(avg.get(songid, 0.0))

    return np.array(X), np.array(Y), np.array(weight)


def load_data_with_avg(path, avg={}):
    X = []
    Y = []

    with open(path) as f:
        for l in f.readlines():
            row = []
            data = l.split(" ")
            songid = data[1]
            Y.append(int(data[2]))

            data[6] = time.mktime(time.strptime(data[6]+" 0:00:00", '%Y%m%d %H:%M:%S'))
            data[7] = time.mktime(time.strptime(data[7]+" 0:00:00", '%Y%m%d %H:%M:%S'))

            for d in data[3:]:
                row.append(float(d))
            row.append(avg.get(songid, 0))
            X.append(row)

    return np.array(X), np.array(Y)


def load_data_no_cut(path, avg={}):
    X = []
    Y = []

    with open(path) as f:
        for l in f.readlines():
            row = []
            data = l.split(" ")
            songid = data[1]
            Y.append(int(data[2]))

            data[6] = time.mktime(time.strptime(data[6]+" 0:00:00", '%Y%m%d %H:%M:%S'))
            data[7] = time.mktime(time.strptime(data[7]+" 0:00:00", '%Y%m%d %H:%M:%S'))

            for d in data[3:]:
                row.append(float(d))
            row.append(avg.get(songid, 0))
            X.append(row)

    return np.array(X), np.array(Y)


def train_gbrt(data, avg={}):
    test_X, test_Y = load_data(data, avg)
    gbrt = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=0, loss='ls')
    gbrt.fit(test_X, test_Y)
    return gbrt


def train_xgboost(data, avg={}):
    test_X, test_Y = load_data_no_cut(data, avg)
    bst = xgb.XGBModel(max_depth=6, learning_rate=0.1, silent=True, objective='reg:linear',
             subsample=0.7, reg_alpha=0.5, reg_lambda=0.3, n_estimators=80)
    # bst.set_params(**param)
    bst.fit(test_X, test_Y)

    return bst


def train_svm(data):
    test_X, test_Y = load_data(data)
    svr = SVR(kernel='rbf', C=100, gamma=1)
    print("start train")
    svr.fit(test_X, test_Y)
    print("train finish")
    return svr


def train_cart(data):
    test_X, test_Y = load_data(data)
    clf = DecisionTreeRegressor(max_depth=4)
    clf.fit(test_X, test_Y)
    return clf


def train_adboost_cart(data, avg={}):
    test_X, test_Y = load_data(data, avg)
    adaboost = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), loss="square", learning_rate=0.01, n_estimators=500)
    adaboost.fit(test_X, test_Y)
    return adaboost


def train_multi_adboost_cart(data):
    test_X, test_Y = load_data(data)
    adaboost = MultiAdaBoostRegressor([
        DecisionTreeRegressor(max_depth=4),
        GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=0, loss='ls'),
        xgb.XGBModel(max_depth=4, learning_rate=0.6, silent=True, objective='reg:linear', subsample=0.7,
                     reg_alpha=0.5, reg_lambda=0.3, n_estimators=30)
    ], loss="square", learning_rate=0.01, n_estimators=500)
    adaboost.fit(test_X, test_Y)
    return adaboost


def test_multi_adboost_cart(data):
    test_X, test_Y = load_data(data)
    adaboost = MultiAdaBoostRegressor([
        DecisionTreeRegressor(max_depth=4),
        GradientBoostingRegressor(n_estimators=1, learning_rate=0.1, max_depth=4, random_state=0, loss='ls'),
        xgb.XGBModel(max_depth=4, learning_rate=0.6, silent=True, objective='reg:linear', subsample=0.7,
                     reg_alpha=0.5, reg_lambda=0.3, n_estimators=1)
    ], loss="square", learning_rate=0.01, n_estimators=4)
    adaboost.fit(test_X, test_Y)
    return adaboost


def get_time(ts):
    time_array = time.localtime(ts)
    time_str = time.strftime("%Y%m%d", time_array)
    return time_str


def test(bst, test, origin, avg={}):
    test_X, test_Y = load_data(test)
    predict = bst.predict(test_X)

    plays_prediction = {}

    with open(test) as f:
        i = 0
        for l in f.readlines():
            songid = l.split(" ")[1]
            artist = l.split(" ")[0]
            date = l.split(" ")[6]
            key = artist + "-" + date
            plays_prediction[key] = plays_prediction.get(key, 0.0) + predict[i] + avg.get(songid, 0)
            i += 1

    plays_actual = {}
    with open(origin) as f:
        i = 0
        for l in f.readlines():
            real = float(l.split(" ")[2])
            artist = l.split(" ")[0]
            date = l.split(" ")[6]
            key = artist + "-" + date
            plays_actual[key] = plays_actual.get(key, 0.0) + real
            i += 1

    sigma = {}
    Phi = {}
    for k in plays_actual:
        artist = k.split("-")[0]
        sigma[artist] = sigma.get(artist, 0) + ((plays_prediction.get(k, 0.0) - plays_actual.get(k, 0.0)) / plays_actual.get(k, 0.0)) ** 2
        Phi[artist] = Phi.get(artist, 0.0) + plays_actual[k]

    for k in sigma:
        sigma[k] = math.sqrt(sigma.get(k, 0.0) / 30)
        Phi[k] = math.sqrt(Phi[k])

    F = 0
    i = 0
    total = 0
    for k in sigma:
        F += (1 - sigma[k]) * Phi[k]
        total += Phi[k]
        i += 1

    print(F)
    print(total)


def test_model(func):
    print('***********************')
    print(getattr(func, '__name__'))
    print('***********************')
    model = func("data/train_data_cut")
    # model = train_gbrt()
    # model = test_multi_adboost_cart()
    print("cut-test")
    test(model, 'data/test_data', "data/test")
    print("cut-origin")
    test(model, 'data/test_cut', "data/test_cut")

    model = func("data/train_data_full")
    # model = train_gbrt()
    # model = test_multi_adboost_cart()
    print("full-test")
    test(model, 'data/test_data', "data/test")
    print("full-origin")
    test(model, 'data/test_full', "data/test_full")

    model = func("data/train_data_full_no_start")
    # model = train_gbrt()
    # model = test_multi_adboost_cart()
    print("full_no_start-test")
    test(model, 'data/test_data', "data/test")
    print("full_no_start-origin")
    test(model, 'data/test_full_no_start', "data/test_full_no_start")


if __name__ == "__main__":
    # test_model(train_xgboost)
    # test_model(train_adboost_cart)
    # test_model(train_gbrt)
    avg = {}
    avg = get_avg("data/train_data_full_no_start")
    model = train_gbrt("data/train_data_full_no_start", avg)
    test(model, 'data/test_full_no_start', "data/test_full_no_start", avg)

    # model2 = train_cart()
    # print("start cart")
    # test(model2)
