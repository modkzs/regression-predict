# -*- coding: utf-8 -*-
import xgboost as xgb
import math
import time
import numpy as np
import os
from sklearn import linear_model
from sklearn.ensemble import AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

__author__ = 'yixuanhe'
times = {}
interval = 15
cut_avg = False
last = {}
l9 = ['11', '0', '100', '3', '14', '2', '1', '4', '12']
l10 = ['1', '3', '2']
feature_not_used = [6, 7, 9, 10]


def get_avg(path, date='20150701'):
    avg = {}

    with open(path) as f:
        for l in f.readlines():
            data = l.split(" ")
            if data[6] > date:
                songid = data[1]
                play = float(data[2])
                avg[songid] = avg.get(songid, 0) + play

    for key in avg:
        avg[key] = avg[key] / 31.0

    return avg


def load_data(path, avg={}):
    X = []
    Y = []
    weight = []

    with open(path) as f:
        i = 0
        for l in f.readlines():
            i += 1
            data = l.replace(" \n", "").split(" ")
            songid = data[1]

            plays = float(data[2])
            if cut_avg:
                plays -= avg.get(songid, 0.0)
            tmp = times.get(songid, [0 for i in range(interval + 1)])
            tmp.append(plays)
            times[songid] = tmp

            row = []

            if songid not in last:
                last[songid] = plays;
                continue

            if cut_avg:
                Y.append(float(data[2]) - avg.get(songid, 0.0))
            else:
                Y.append(float(data[2]) - last[songid])

            data[5] = math.exp(10 - abs(float(data[5])))
            data[6] = time.mktime(time.strptime(data[6] + " 0:00:00", '%Y%m%d %H:%M:%S'))
            data[7] = time.mktime(time.strptime(data[7] + " 0:00:00", '%Y%m%d %H:%M:%S'))

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

            for p in range(-interval, 0, 1):
                data.append(times[songid][p + 1] - times[songid][p])

            # for p in times[songid][-interval - 1:-1:]:
            #     data.append(p)

            i = 3
            for d in data[3:]:
                if i in feature_not_used:
                    i += 1
                    continue
                row.append(float(d))
                i += 1
            row.append(avg.get(songid, 0))
            X.append(row)
            weight.append(avg.get(songid, 0.0))
            last[songid] = float(data[2])

    return np.array(X), np.array(Y), np.array(weight)


def get_times(path):
    with open(path) as f:
        i = 0
        for l in f.readlines():
            i += 1
            data = l.split(" ")
            songid = data[1]
            plays = float(data[2])
            tmp = times.get(songid, [0, 0, 0, 0, 0])
            tmp.append(plays)
            times[songid] = tmp


def train(train_file, resave=False, avg={}, train_data='data/dtrain_serise_15', test_data='data/dtest_serise_15'):
    if not os.path.exists(train_data) or resave:
        X, Y, weight = load_data(train_file, avg)
        dtrain = xgb.DMatrix(X, label=Y)
        dtrain.save_binary(train_data)
    else:
        dtrain = xgb.DMatrix(train_data)

    # if not os.path.exists(test_data) or resave:
    #     X, Y, weight = load_data(eval_file, get_avg(eval_file))
    #     dtest = xgb.DMatrix(X, label=Y)
    #     dtest.save_binary(test_data)
    # else:
    #     dtest = xgb.DMatrix(test_data)

    watchlist = [(dtrain, 'train')]

    print('start running example to start from a initial prediction')

    param = {'max_depth': 6, 'learning_rate': 0.1, 'silent': 1, 'objective': 'reg:linear',
             'subsample': 0.7, 'alpha': 0.3, 'lambda': 1, 'booster': 'gblinear'}
    num_round = 30
    bst = xgb.train(param, dtrain, num_round, watchlist)  # , early_stopping_rounds=5)
    return bst


def train_svm(train_file):
    test_X, test_Y, weight = load_data(train_file, get_avg(train_file))
    svr = SVR(kernel='rbf', C=100, gamma=1)
    print("start train")
    svr.fit(test_X, test_Y)
    print("train finish")
    return svr


def train_cart(train_file, avg={}):
    test_X, test_Y, weight = load_data(train_file, avg)
    clf = DecisionTreeRegressor(max_depth=5)
    clf.fit(test_X, test_Y)
    return clf


def train_liner(path, avg={}):
    X, Y, weight = load_data(path, avg)
    np.save("train_X", X)
    np.save("train_Y", Y)
    clf = linear_model.LinearRegression()
    clf.fit(X, Y)
    print("train over")
    return clf


def get_test_result(bst, origin, avg):
    Y = []
    if len(times) == 0:
        get_times("data/train_data_full_start")

    with open(origin) as f:
        j = 0
        for l in f.readlines():
            j += 1
            data = l.replace(" \n", "").split(" ")
            songid = data[1]

            if songid not in times:
                times[songid] = [0 for x in range(interval)]

            row = []

            data[5] = math.exp(10 - abs(float(data[5])))
            data[6] = time.mktime(time.strptime(data[6] + " 0:00:00", '%Y%m%d %H:%M:%S'))
            data[7] = time.mktime(time.strptime(data[7] + " 0:00:00", '%Y%m%d %H:%M:%S'))

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

            if len(times[songid]) < interval:
                for i in range(interval - len(times[songid])):
                    data.append(0)

            # for p in times[songid][-interval::]:
            #     data.append(p)
            for p in range(-interval, 0, 1):
                data.append(times[songid][p + 1] - times[songid][p])

            i = 3
            for d in data[3:]:
                if i in feature_not_used:
                    i += 1
                    continue
                row.append(float(d))
                i += 1
            row.append(avg.get(songid, 0))

            if bst.__class__.__name__ == "Booster":
                dtest = xgb.DMatrix(row)
            else:
                dtest = np.array(row).reshape(1, (len(row)))

            y = bst.predict(dtest)  # , ntree_limit=bst.best_ntree_limit)
            val = float(y[0])
            if cut_avg:
                val = y[0] + avg.get(songid, 0)
            val = int(math.floor(val + 0.5 + last.get(songid, 0)))
            if val < 0:
                val = 0
            last[songid] = val

            Y.append(val)

            tmp = times.get(songid, [])
            tmp.append(val)
            times[songid] = tmp

        return Y


def test(bst, origin, avg={}):
    predict = get_test_result(bst, origin, avg)

    plays_prediction = {}
    plays_actual = {}

    with open(origin) as f:
        i = 0
        for l in f.readlines():
            songid = l.split(" ")[1]
            real = float(l.split(" ")[2])
            artist = l.split(" ")[0]
            date = l.split(" ")[6]
            key = artist + "-" + date
            plays_prediction[key] = plays_prediction.get(key, 0) + predict[i]
            plays_actual[key] = plays_actual.get(key, 0) + real
            i += 1

    # plays_actual = sorted(plays_actual.items(), key=operator.itemgetter(0), reverse=False)
    #
    # with open("compare_song", "w") as f:
    #     for k in plays_actual:
    #         f.write(k[0] + ", " + str(k[1]) + ", " + str(plays_prediction[k[0]])+"\n")

    sigma = {}
    Phi = {}
    for k in plays_actual:
        artist = k.split("-")[0]
        if plays_actual.get(k, 0.0) == 0:
            continue
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


def train_adboost_cart(train_file):
    test_X, test_Y, weight = load_data(train_file, get_avg(train_file))
    adaboost = AdaBoostRegressor(DecisionTreeRegressor(max_depth=5), loss="square", learning_rate=0.01,
                                 n_estimators=500)
    adaboost.fit(test_X, test_Y)
    return adaboost


def draw(data):
    import matplotlib.pyplot as plt
    x = [i for i in range(len(data))]
    plt.plot(x, data)
    plt.show()


def write_result(bst, test_file, write_file, avg={}):
    predict = get_test_result(bst, test_file, get_avg(test_file))

    plays_prediction = {}

    with open(test_file) as f:
        i = 0
        for l in f.readlines():
            artist = l.split(" ")[0]
            date = l.split(" ")[6]
            key = artist + "-" + date
            plays_prediction[key] = plays_prediction.get(key, 0) + predict[i]
            i += 1

    with open(write_file, "w") as f:
        for k in plays_prediction:
            artist, date = k.split("-", 1)
            times = plays_prediction[k]
            f.write(artist + "," + str(times) + "," + date + "\n")


avg = get_avg("data/train_data_full_start_avg")
model = train("data/train_data_full_start_avg", resave=True, avg=avg)
# model = train_svm("data/train_data_full_no_start")
# model = train_adboost_cart("data/train_data_full_start")
# model = train_cart("data/train_data_full_start")
# model = train_liner("data/train_data_full_start", avg)
test(model, "data/test_full_start", avg)

with open("data/top1", "w") as f:
    for i in times['6d3cf538a1fc9adc4911c9359833eab7']:
        f.write(str(i) + "\n")

with open("data/top2", "w") as f:
    for i in times['7ec488fc483386cdada5448864e82990']:
        f.write(str(i) + "\n")

# model = train_cart("data/deal_mars_tianchi_full_no_start", '20150801')
# write_result(model, 'data/pose_data', 'data/mars_tianchi_artist_plays_predict.csv')
