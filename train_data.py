# -*- coding: utf-8 -*-
import operator
import xgboost as xgb
import math
import time
import numpy as np
__author__ = 'yixuanhe'


def get_time(ts):
    time_array = time.localtime(ts)
    time_str = time.strftime("%Y%m%d", time_array)
    return time_str


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
        l9 = ['11', '0', '100', '3', '14', '2', '1', '4', '12']
        l10 = ['1', '3', '2']
        for l in f.readlines():
            row = []
            data = l.split(" ")
            songid = data[1]
            Y.append(int(data[2]))

            data[6] = time.mktime(time.strptime(data[6]+" 0:00:00", '%Y%m%d %H:%M:%S'))
            data[7] = time.mktime(time.strptime(data[7]+" 0:00:00", '%Y%m%d %H:%M:%S'))

            for l in l9:
                if data[9] == l:
                    data.append('1')
                else:
                    data.append('0')

            for l in l10:
                if data[10] == l:
                    data.append('1')
                else:
                    data.append('0')

            i = 3
            for d in data[3:]:
                if i == 9 or i == 10:
                    continue
                row.append(float(d))
                i += 1
            row.append(avg.get(songid, 0))
            X.append(row)
            weight.append(avg.get(songid, 0.0))

    return np.array(X), np.array(Y), np.array(weight)


def logregobj(preds, dtrain):
    labels = dtrain.get_label()
    preds = 1.0 / (1.0 + np.exp(-preds))
    grad = preds - labels
    hess = preds * (1.0-preds)
    return grad, hess


def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    # return a pair metric_name, result
    # since preds are margin(before logistic transformation, cutoff at 0)
    return 'error', float(sum(labels != (preds > 0.0))) / len(labels)


def train(train_file, eval_file):
    # X, Y, weight = load_data(train_file, get_avg(train_file))
    # dtrain = xgb.DMatrix(X, label=Y)
    # dtrain.save_binary('data/dtrain_no_weight')
    # dtrain.save_binary('data/dtrain_dump')
    dtrain = xgb.DMatrix('data/dtrain_dump')
    # X, Y, weight = load_data(eval_file, get_avg(eval_file))
    # dtest = xgb.DMatrix(X, label=Y)
    # dtest.save_binary('data/dtest_dump')
    dtest = xgb.DMatrix('data/dtest_dump')

    watchlist = [(dtrain, 'train'), (dtest, 'eval')]

    print('start running example to start from a initial prediction')

    param = {'max_depth': 6, 'learning_rate': 0.1, 'silent': 1, 'objective': 'reg:linear',
             'subsample': 0.7, 'alpha': 0.3, 'lambda': 0.8}
    num_round = 40
    bst = xgb.train(param, dtrain, num_round, watchlist, early_stopping_rounds=5)
    return bst


def test(bst, test, origin, avg={}):
    dtest = xgb.DMatrix(test)
    predict = bst.predict(dtest)

    plays_prediction = {}

    with open(origin) as f:
        i = 0
        for l in f.readlines():
            # songid = l.split(" ")[1]
            artist = l.split(" ")[0]
            date = l.split(" ")[6]
            key = artist + "-" + date
            plays_prediction[key] = plays_prediction.get(key, 0.0) + predict[i]
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


def write_result(bst, test_file, write_file, avg={}):
    X, Y, weight = load_data(test_file, get_avg("data/train_data_full_no_start"))
    dtest = xgb.DMatrix(X, label=Y)
    predict = bst.predict(dtest)

    plays_prediction = {}

    with open(test_file) as f:
        i = 0
        for l in f.readlines():
            artist = l.split(" ")[0]
            date = l.split(" ")[6]
            key = artist + "-" + date
            plays_prediction[key] = plays_prediction.get(key, 0.0) + predict[i]
            i += 1

    with open(write_file, "w") as f:
        for k in plays_prediction:
            artist, date = k.split("-", 1)
            times = plays_prediction[k]
            f.write(artist + "," + str(times) + "," + date + "\n")


model = train("data/train_data_full_no_start", "data/test_full_no_start")
test(model, 'data/dtest_dump', "data/test_full_no_start")
write_result(model, 'data/pose_data', 'data/mars_tianchi_artist_plays_predict.csv')
# avg = get_avg('data/train_data_full_no_start')
# sorted_x = sorted(avg.items(), key=operator.itemgetter(1), reverse=True)
# i = 0
# for k in sorted_x:
#     print(k)
#     i += 1
#     if i > 10:
#         break
