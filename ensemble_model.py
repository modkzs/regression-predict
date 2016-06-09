# -*- coding: utf-8 -*-
from sklearn.svm import SVR

from feature.TimeSeriesFeature import TimeSeriesFeature
from sklearn import linear_model
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
from sklearn.ensemble.forest import RandomForestRegressor
import math
import numpy as np

__author__ = 'yixuanhe'


def train_cart(X, Y):
    clf = DecisionTreeRegressor(max_depth=5)
    clf.fit(X, Y)
    return clf


class TimeSeriesFeatureWithSmooth(TimeSeriesFeature):
    def __init__(self, cut_avg, interval, train_path, test_path, date="20150701", exclude=[]):
        super().__init__(cut_avg, interval, train_path, test_path, date, exclude)
        self.avg_days = 2

    def line_func(self, data, flag):
        # avg = 0
        # num = 0
        # if data[1] in self.times:
        #     for i in self.times[data[1]][-self.avg_days::]:
        #         avg += i
        #         num += 1
        # avg += data[2]
        # avg /= (num+1)
        # data[2] = avg

        if self.avg.get(data[1], 0) != 0 and flag:
            avg = self.avg.get(data[1])

            if self.cut_avg:
                val = avg
            else:
                val = 2 * avg

            if data[2] > val:
                data[2] = (data[2] - val) / 2 + val

        return super().line_func(data, flag)

    def deal_predict(self, y, id, actual):
        super().deal_predict(y, id, actual)

        if self.avg.get(id, 0) != 0:
            avg = self.avg.get(id, 0)

            if self.cut_avg:
                val = avg
            else:
                val = 2 * avg

            if y > val:
                y = (y - val) * 2 + val

        return y


class TimeSeriesFeatureRMSLE(TimeSeriesFeature):
    def line_func(self, data, flag):
        data[2] = math.log(data[2]+1)
        return super().line_func(data, flag)

    def deal_predict(self, y, id):
        super().deal_predict(y, id)
        y = math.exp(y) - 1

        return y


def train_ridge(X, Y):
    clf = linear_model.Ridge(alpha=0.8, max_iter=100000)
    clf.fit(X, Y)
    return clf


def train_liner(X, Y):
    clf = linear_model.LinearRegression()
    clf.fit(X, Y)
    return clf


def train_adaboost_liner(X, Y):
    adaboost = AdaBoostRegressor(linear_model.LinearRegression())
    adaboost.fit(X, Y)
    return adaboost


def train_random_forest(X, Y):
    rf = RandomForestRegressor(n_estimators=20)
    rf.fit(X, Y)
    return rf


def train_xgboost(X, Y):
    # dtrain = xgb.DMatrix(X, label=Y)
    # watchlist = [(dtrain, 'train')]
    #
    # print('start running example to start from a initial prediction')
    #
    # param = {'max_depth': 5, 'learning_rate': 0.03, 'silent': 1, 'objective': 'reg:linear',
    #          'subsample': 0.7, 'alpha': 0.8, 'lambda': 0.8, 'booster': 'gblinear'}
    # num_round = 200
    # # bst = xgb.cv(param, dtrain, num_round, nfold=10, metrics={'error'}, seed=0)
    # bst = xgb.train(param, dtrain, num_round)

    bst = xgb.XGBRegressor(max_depth=6, learning_rate=0.02, n_estimators=300, silent=True, objective='reg:linear',
                           subsample=0.7, reg_alpha=0.8, reg_lambda=0.8)
    bst.fit(X, Y)

    return bst


def single_predict(model, X):
    if model.__class__.__name__ == "Booster":
        dtest = xgb.DMatrix(X)
    else:
        dtest = X

    Y = model.predict(dtest)
    return Y


def train_gbrt(X, Y):
    gbrt = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=0, loss='ls')
    gbrt.fit(X, Y)
    return gbrt


def train(model_feature, check, weight=None):
    feature_sample = {}
    models = {}
    weights = {}
    # actual = []
    for f in model_feature.values():
        X, Y = f.extract()
        feature_sample[f] = [X, Y]
        # actual = Y
    print("feature extraction over")

    # length = len(actual)
    # Y = [[] for i in range(length)]
    for func in model_feature:
        feature_gen = model_feature[func]
        feature = feature_sample[feature_gen]
        model = func(feature[0], feature[1])
        models[model] = feature_gen
        weights[model] = feature_gen.get_rmse(model)
        print(func.__name__ + ": " + str(weights[model]))

    print("train over")

    ys = []
    for model in models:
        feature_gen = model_feature[func]
        feature = feature_sample[feature_gen]
        Y = model.predict(feature[0])
        actual = feature[1]
        ys.append(Y)
    ys.append(actual)

    with open("data/ensemble_train", "w") as f:
        for i in range(len(ys[0])):
            line = ""
            for y in ys:
                print(type(y[i]))
                line = line + str(y[i]) + " "
            f.write(line + "\n")

    # final_model = train_liner(Y, actual)

    result = []
    w = []
    for model in models:
        feature_gen = models[model]
        model_predict = feature_gen.predict_result(model)
        result.append(model_predict)
        print(model)
        w.append(1/weights[model])

    with open("data/ensemble_result", "w") as f, open("data/test_full_start") as r:
        i = 0
        for l in r.readlines():
            data = l.split(" ")
            line = data[0] + " " + data[1] + " " + data[6] + " "
            for re in result:
                line = line + str(re[i]) + " "
            line += data[2]
            i += 1
            f.write(line + "\n")

    avg = [1 for func in model_feature]

    final = []
    length = len(w)

    if not weight is None:
        w = weight

    for j in range(len(result[0])):
        total = 0
        v = 0
        for i in range(length):
            v += w[i]*result[i][j]
            total += w[i]
        v /= total
        final.append(int(math.floor(v+0.5)))

    print("weighted:")
    check.check_result(None, final)

    final = []
    w = avg
    length = len(w)
    for j in range(len(result[0])):
        total = 0
        v = 0
        for i in range(length):
            v += w[i]*result[i][j]
            total += w[i]
        v /= total
        final.append(int(math.floor(v+0.5)))

    print("avg:")
    check.check_result(None, final)

    return final


def train_svm(X, Y):
    svr = SVR(kernel='rbf', C=100, gamma=1, verbose=True, cache_size=1024)
    print("start train")
    svr.fit(X, Y)
    print("train finish")
    return svr


def high_level_test(train_path, test_path, func):
    train_X = []
    train_Y = []
    with open(train_path) as f:
        for l in f.readlines():
            data = l.strip().split(" ")
            train_Y.append(math.log(float(data[-1])+1))
            train_X.append([(max(float(x), 0)) for x in data[:-1]])
    model = func(train_X, train_Y)

    predict = {}
    actual = {}
    with open(test_path) as f:
        for l in f.readlines():
            data = l.strip().split(" ")
            Y = int(data[-1])
            X = [max(float(x), 0) for x in data[3:-1]]
            X = np.array(X).reshape(1, (len(X)))
            y = math.exp(model.predict(X))-1
            artist = data[0]
            date = data[2]
            key = artist + "-" + date
            predict[key] = predict.get(key, 0) + y
            actual[key] = actual.get(key, 0) + Y

        sigma = {}
        Phi = {}
        for k in actual:
            artist = k.split("-")[0]
            if actual.get(k, 0.0) == 0:
                continue
            sigma[artist] = sigma.get(artist, 0) + ((math.floor(predict.get(k, 0.0)+0.5) - actual.get(k, 0.0)) / actual.get(k, 0.0)) ** 2
            Phi[artist] = Phi.get(artist, 0.0) + actual[k]

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

        print(model.__class__.__name__)
        print(F)
        print(total)

        return F


if __name__ == "__main__":
    for func in [train_ridge, train_liner, train_xgboost, train_random_forest, train_cart, train_gbrt]:
        print(func)
        high_level_test("data/ensemble_train", "data/ensemble_result", func)
    # cut_avg = False
    # interval = 15
    #
    # feature_not_used = [9, 10]
    # Fs = []
    #
    # tsfws = TimeSeriesFeatureWithSmooth(cut_avg, interval, 'data/train_data_full_start_avg',
    #                                     'data/test_full_start_avg', exclude=feature_not_used)
    # tsf = TimeSeriesFeature(cut_avg, interval, 'data/train_data_full_start_avg',
    #                         'data/test_full_start_avg', exclude=feature_not_used)
    #
    # # tsf = TimeSeriesFeature(cut_avg, interval, 'data/deal_mars_tianchi_full_start_avg',
    # #                         'data/pose_data', exclude=feature_not_used)
    # # tsfws = TimeSeriesFeatureWithSmooth(cut_avg, interval, 'data/deal_mars_tianchi_full_start_avg',
    # #                                     'data/pose_data', exclude=feature_not_used)
    #
    # model_feature = {train_xgboost: tsf, train_liner: tsfws, train_ridge: tsfws, train_cart: tsf}
    # result = train(model_feature, tsfws)
    # tsfws.write_result('data/mars_tianchi_artist_plays_predict_ensemble.csv', result)
