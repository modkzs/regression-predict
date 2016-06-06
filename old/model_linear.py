# -*- coding: utf-8 -*-
import math
import time
import numpy as np
from sklearn.gaussian_process import GaussianProcess
from sklearn import linear_model
from datetime import datetime


__author__ = 'yixuanhe'
times = {}
interval = 15
cut_avg = False
feature_not_used = [6, 7, 9, 10]
l9 = ['11', '0', '100', '3', '14', '2', '1', '4', '12']
l10 = ['1', '3', '2']
# l9 = []
# l10 = []
holiday = set([
    "20150301",
    "20150307",
    "20150308",
    "20150314",
    "20150315",
    "20150321",
    "20150322",
    "20150328",
    "20150329",
    "20150404",
    "20150405",
    "20150406",
    "20150411",
    "20150412",
    "20150418",
    "20150419",
    "20150425",
    "20150426",
    "20150501",
    "20150502",
    "20150503",
    "20150509",
    "20150510",
    "20150516",
    "20150517",
    "20150523",
    "20150524",
    "20150530",
    "20150531",
    "20150606",
    "20150607",
    "20150613",
    "20150614",
    "20150620",
    "20150621",
    "20150622",
    "20150627",
    "20150628",
    "20150704",
    "20150705",
    "20150711",
    "20150712",
    "20150718",
    "20150719",
    "20150725",
    "20150726",
    "20150801",
    "20150802",
    "20150808",
    "20150809",
    "20150815",
    "20150816",
    "20150822",
    "20150823",
    "20150829",
    "20150830",
    "20150903",
    "20150904",
    "20150905",
    "20150912",
    "20150913",
    "20150919",
    "20150920",
    "20150926",
    "20150927",
    "20151001",
    "20151002",
    "20151003",
    "20151004",
    "20151005",
    "20151006",
    "20151007",
    "20151011",
    "20151017",
    "20151018",
    "20151024",
    "20151025",
    "20151031"
])


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
        j = 0
        for l in f.readlines():
            j += 1
            data = l.replace(" \n", "").split(" ")
            songid = data[1]
            plays = float(data[2])
            tmp = times.get(songid, [])
            tmp.append(plays)
            times[songid] = tmp

            if songid not in times:
                continue

            if len(times[songid]) < interval + 1:
                continue

            row = []
            if cut_avg:
                Y.append(float(data[2]) - avg.get(songid, 0.0))
            else:
                Y.append(float(data[2]))

            data[5] = math.exp(10 - abs(int(data[5])))
            d = datetime(int(data[6][0:4]), int(data[6][4:6]), int(data[6][6:8])).weekday()

            for i in range(0, 8):
                if i == d:
                    data.append(1)
                else:
                    data.append(0)

            if data[6] in holiday:
                data.append(1)
            else:
                data.append(0)

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

            for p in times[songid][-interval - 1:-1:]:
                data.append(p)

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

    return np.array(X), np.array(Y), np.array(weight)


def train_gaussian(path):
    X, Y, weight = load_data(path, get_avg(path))
    np.save("train_X", X)
    np.save("train_Y", Y)
    gp = GaussianProcess()
    gp.fit(X, Y)
    return gp


def train_liner(path, avg={}):
    X, Y, weight = load_data(path, avg)
    np.save("train_X", X)
    np.save("train_Y", Y)
    clf = linear_model.LinearRegression(normalize=True)
    clf.fit(X, Y)
    print("train over")
    return clf


def train_ridge(path, avg={}):
    X, Y, weight = load_data(path, avg)
    np.save("train_X", X)
    np.save("train_Y", Y)
    clf = linear_model.Ridge(alpha=0.1, max_iter=100000)
    clf.fit(X, Y, weight)
    print("train over")
    return clf


def train_lasso(path, avg={}):
    X, Y, weight = load_data(path, avg)
    np.save("train_X", X)
    np.save("train_Y", Y)
    clf = linear_model.Lasso(alpha=0.5, normalize=True, max_iter=100000)
    clf.fit(X, Y)
    print("train over")
    return clf


def get_times(path):
    with open(path) as f:
        i = 0
        for l in f.readlines():
            i += 1
            data = l.split(" ")
            songid = data[1]
            plays = int(data[2])
            tmp = times.get(songid, [])
            tmp.append(plays)
            times[songid] = tmp


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
            d = datetime(int(data[6][0:4]), int(data[6][4:6]), int(data[6][6:8])).weekday()

            for i in range(0, 8):
                if i == d:
                    data.append(1)
                else:
                    data.append(0)

            if data[6] in holiday:
                data.append(1)
            else:
                data.append(0)

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

            for p in times[songid][-interval::]:
                data.append(p)

            i = 3
            for d in data[3:]:
                if i in feature_not_used:
                    i += 1
                    continue
                row.append(float(d))
                i += 1
            row.append(avg.get(songid, 0))

            dtest = np.array(row).reshape(1, -1)

            y = bst.predict(dtest)
            val = float(y[0])
            if cut_avg:
                val = y[0] + avg.get(songid, 0)
            val = int(math.floor(val + 0.5))

            Y.append(val)

            tmp = times.get(songid, [])
            tmp.append(val)
            times[songid] = tmp

        return Y


def test(bst, test, avg={}):
    predict = get_test_result(bst, test, avg)

    plays_prediction = {}

    with open(test) as f:
        i = 0
        for l in f.readlines():
            artist = l.split(" ")[0]
            date = l.split(" ")[6]
            key = artist + "-" + date
            plays_prediction[key] = plays_prediction.get(key, 0.0) + predict[i]
            i += 1

    plays_actual = {}
    with open(test) as f:
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
        sigma[artist] = sigma.get(artist, 0) + ((plays_prediction.get(k, 0.0) - plays_actual.get(k,
                                                                                                 0.0)) / plays_actual.get(
            k, 0.0)) ** 2
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
    predict = get_test_result(bst, test_file, avg)

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
# model = train_liner("data/train_data_full_start", avg)
model = train_ridge("data/train_data_full_start_avg", avg)
# model = train_lasso("data/train_data_full_start", avg)
test(model, "data/test_full_start", avg)

print(model.coef_)

# with open("data/ltop1", "w") as f:
#     for i in times['6d3cf538a1fc9adc4911c9359833eab7']:
#         f.write(str(i) + "\n")
#
# with open("data/ltop2", "w") as f:
#     for i in times['7ec488fc483386cdada5448864e82990']:
#         f.write(str(i) + "\n")

# avg = get_avg("data/deal_mars_tianchi_full_start_avg", "20150801")
# model = train_ridge("data/deal_mars_tianchi_full_start_avg", avg)
# print("train over")
# write_result(model, 'data/pose_data', 'data/mars_tianchi_artist_plays_predict.csv', avg)

