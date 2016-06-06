# -*- coding: utf-8 -*-
from feature.TimeSeriesFeature import TimeSeriesFeature
from sklearn import linear_model
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
from sklearn.ensemble.forest import RandomForestRegressor
import math

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
    dtrain = xgb.DMatrix(X, label=Y)
    watchlist = [(dtrain, 'train')]

    print('start running example to start from a initial prediction')

    param = {'max_depth': 5, 'learning_rate': 0.03, 'silent': 1, 'objective': 'reg:linear',
             'subsample': 0.7, 'alpha': 0.8, 'lambda': 0.8, 'booster': 'gblinear'}
    num_round = 200
    # bst = xgb.cv(param, dtrain, num_round, nfold=10, metrics={'error'}, seed=0)
    bst = xgb.train(param, dtrain, num_round)

    return bst


def single_predict(model, X):
    if model.__class__.__name__ == "Booster":
        dtest = xgb.DMatrix(X)
    else:
        dtest = X

    Y = model.predict(dtest)
    return Y


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
        # cur = single_predict(model, feature[0])
        # for i in range(length):
        #     Y[i].append(cur[i])
        weights[model] = feature_gen.get_rmse(model)
        print(func.__name__ + ": " + str(weights[model]))

    print("train over")

    # final_model = train_liner(Y, actual)

    result = []
    w = []
    for model in models:
        feature_gen = models[model]
        result.append(feature_gen.predict_result(model))
        print(model)
        w.append(1/weights[model])

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


if __name__ == "__main__":
    cut_avg = False
    interval = 15

    feature_not_used = [9, 10]
    Fs = []

    tsfws = TimeSeriesFeatureWithSmooth(cut_avg, interval, 'data/train_data_full_start_avg',
                                        'data/test_full_start_avg', exclude=feature_not_used)
    tsf = TimeSeriesFeature(cut_avg, interval, 'data/train_data_full_start_avg',
                            'data/test_full_start_avg', exclude=feature_not_used)

    tsf = TimeSeriesFeature(cut_avg, interval, 'data/deal_mars_tianchi_full_start_avg',
                            'data/pose_data', exclude=feature_not_used)
    tsfws = TimeSeriesFeatureWithSmooth(cut_avg, interval, 'data/deal_mars_tianchi_full_start_avg',
                                        'data/pose_data', exclude=feature_not_used)

    model_feature = {train_xgboost: tsf, train_liner: tsfws, train_ridge: tsfws, train_cart: tsf}
    result = train(model_feature, tsfws)
    tsfws.write_result('data/mars_tianchi_artist_plays_predict_ensemble.csv', result)


# if __name__ == "__main__":
#     cut_avg = False
#     interval = 15
#
#     feature_not_used = [9, 10]
#     Fs = []
#
#     tsfws = TimeSeriesFeatureWithSmooth(cut_avg, interval, 'data/deal_mars_tianchi_full_start_avg',
#                                         'data/pose_data', exclude=feature_not_used)
#     tsf = TimeSeriesFeature(cut_avg, interval, 'data/deal_mars_tianchi_full_start_avg',
#                             'data/pose_data', exclude=feature_not_used)
#     X_tsf, Y_tsf = tsf.extract()
#     X_tsfws, Y_tsfws = tsfws.extract()
#     length = len(X_tsfws)
#     print("feature extraction over")
#
#     line = train_liner(X_tsfws, Y_tsfws)
#     F = tsfws.get_rmse(line)
#     print("liner:" + str(F))
#     Fs.append(F)
#
#     xgb = train_xgboost(X_tsf, Y_tsf)
#     F = tsf.get_rmse(xgb)
#     print("xgboost:" + str(F))
#     Fs.append(F)
#
#     ridge = train_ridge(X_tsf, Y_tsf)
#     F = tsf.get_rmse(ridge)
#     print("ridge:" + str(F))
#     Fs.append(F)
#     print("train over")
#
#     cart = train_cart(X_tsfws, Y_tsfws)
#     F = tsf.get_rmse(ridge)
#     print("ridge:" + str(F))
#     Fs.append(F)
#     print("train over")
#
#     total = 0
#     for f in Fs:
#         total += length/f
#     for i in range(len(Fs)):
#         Fs[i] = (length/Fs[i])/total
#
#     print("begin predict")
#     xgb_result = tsfws.predict_result(xgb)
#     print("xbg over")
#     line_result = tsf.predict_result(line)
#     print("line over")
#     ridge_result = tsf.predict_result(ridge)
#     print("ridge over")
#     cart_result = tsfws.predict_result(cart)
#     print("cart over")
#
#     length = len(xgb_result)
#     final = []
#     for i in range(length):
#         final.append(math.floor((line_result[i]*Fs[0] + xgb_result[i]*Fs[1] +
#                                  ridge_result[i]*Fs[2] + cart_result[i]*Fs[3])+0.5))
#
#     print("start check")
#     tsf.check_result(None, final)
#     print("over ")
#
#     length = len(xgb_result)
#     final = []
#     for i in range(length):
#         final.append(math.floor((xgb_result[i] + line_result[i] + ridge_result[i] + cart_result[i])/4+0.5))
#
#     print("start check")
#     tsf.write_result()
#     print("over ")
#
#     # tsfws = TimeSeriesFeatureWithSmooth(cut_avg, interval, 'data/deal_mars_tianchi_full_start_avg',
#     #                                     'data/pose_data', exclude=feature_not_used)
#     # tsf = TimeSeriesFeature(cut_avg, interval, 'data/deal_mars_tianchi_full_start_avg',
#     #                         'data/pose_data', exclude=feature_not_used)
#     #
#     # X_tsf, Y_tsf = tsf.extract()
#     # X_tsfws, Y_tsfws = tsfws.extract()
#     # print("feature extraction over")
#     # xgb = train_xgboost(X_tsf, Y_tsf)
#     # xgb_result = tsf.predict_result(xgb)
#     # liner = train_liner(X_tsfws, Y_tsfws)
#     # liner_result = tsfws.predict_result(liner)
#     # ridge = train_ridge(X_tsfws, Y_tsfws)
#     # ridge_result = tsfws.predict_result(ridge)
#     # print("train over")
#     #
#     # total = 0
#     # for f in Fs:
#     #     total += 1/f
#     # for i in range(len(Fs)):
#     #     Fs[i] /= total
#     #
#     # for k in xgb_result:
#     #     xgb_result[k] = math.floor((xgb_result[k]*Fs[0] + liner_result[k]*Fs[1] + ridge_result[k]*Fs[1])+0.5)
#     #     liner_result[k] = math.floor((xgb_result[k] + liner_result[k] + ridge_result[k])/3+0.5)
#     #
#     # tsf.write_result('data/mars_tianchi_artist_plays_predict_weight.csv', xgb_result)
#     # tsf.write_result('data/mars_tianchi_artist_plays_predict_avg.csv', xgb_result)
#     #
#     # print("write over")
