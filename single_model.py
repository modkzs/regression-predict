# -*- coding: utf-8 -*-
from feature.TimeSeriesFeature import TimeSeriesFeature
from sklearn import linear_model
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
from sklearn.ensemble.forest import RandomForestRegressor
import numpy as np
import matplotlib
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

    def deal_predict(self, y, id, actual):
        super().deal_predict(y, id, actual)
        return math.exp(y) - 1


class TimeSeriesFeatureWithSqrt(TimeSeriesFeature):
    def line_func(self, data, flag):
        data[2] = math.sqrt(data[2])
        return super().line_func(data, flag)

    def deal_predict(self, y, id, actual):
        super().deal_predict(y, id, actual)
        return y**2


def train_ridge(X, Y):
    clf = linear_model.Ridge(alpha=0.8, max_iter=100000)
    clf.fit(X, Y)
    return clf


def train_liner(X, Y):
    clf = linear_model.LinearRegression()
    clf.fit(X, Y)
    return clf


def train_bagging_xgboost(X, Y):
    adaboost = BaggingRegressor(xgb.XGBRegressor(max_depth=6, learning_rate=0.02, n_estimators=300, silent=True,
                                                 objective='reg:linear', subsample=0.7, reg_alpha=0.8,
                                                 reg_lambda=0.8, booster="gblinear")
                                , max_features=0.7, n_estimators=30)
    adaboost.fit(X, Y)
    return adaboost


def train_adaboost_xgboost(X, Y):
    adaboost = AdaBoostRegressor(xgb.XGBRegressor(max_depth=6, learning_rate=0.02, n_estimators=300, silent=True,
                                                 objective='reg:linear', subsample=0.7, reg_alpha=0.8,
                                                 reg_lambda=0.8, booster="gblinear"), loss='exponential')
    adaboost.fit(X, Y)
    return adaboost


def train_random_forest(X, Y):
    rf = RandomForestRegressor(n_estimators=20)
    rf.fit(X, Y)
    return rf


def train_bagging_cart(X, Y):
    adaboost = BaggingRegressor(DecisionTreeRegressor(max_depth=5) , max_features=0.7, n_estimators=30)
    adaboost.fit(X, Y)
    return adaboost


def train_xgboost(X, Y):
    # dtrain = xgb.DMatrix(X, label=Y)
    # watchlist = [(dtrain, 'train')]
    #
    # print('start running example to start from a initial prediction')
    #
    # param = {'max_depth': 6, 'learning_rate': 0.02, 'silent': 1, 'objective': 'reg:linear',
    #          'subsample': 0.7, 'alpha': 0.8, 'lambda': 0.8, 'booster': 'gblinear'}
    # num_round = 300
    # # bst = xgb.cv(param, dtrain, num_round, nfold=10, metrics={'error'}, seed=0)
    # bst = xgb.train(param, dtrain, num_round, watchlist)

    bst = xgb.XGBRegressor(max_depth=5, learning_rate=0.03, n_estimators=200, silent=True, objective='reg:linear',
                           subsample=0.7, reg_alpha=0.8, reg_lambda=0.8, colsample_bytree=0.75, booster="gblinear")
    bst.fit(X, Y)

    return bst
    # bst = xgb.XGBRegressor(max_depth=6, learning_rate=0.1, n_estimators=100, silent=True, objective='reg:linear',
    #                        subsample=0.7, reg_alpha=0.8, reg_lambda=0.8)


if __name__ == "__main1__":
    cut_avg = False
    interval = 15

    # feature_not_used = [9, 10]
    # tsf = TimeSeriesFeatureWithSmooth(cut_avg, interval, 'data/train_data_full_start',
    #                                   'data/test_full_start', exclude=feature_not_used)
    # X, Y = tsf.extract()
    param = {'max_depth': 6, 'learning_rate': 0.1, 'silent': 1, 'objective': 'reg:linear',
             'subsample': 0.7, 'alpha': 0.5, 'lambda': 0.8, 'booster': 'gblinear'}
    num_round = 80
    # dtrain = xgb.DMatrix(X, label=Y)
    # watchlist = [(dtrain, 'train')]

    dtrain= xgb.DMatrix('train')

    bst = xgb.cv(param, dtrain, num_round, nfold=10, metrics={'error'}, seed=0)
    bst.to_csv('cv_result')


if __name__ == "__main__":
    cut_avg = False
    interval = 15

    feature_not_used = [9, 10]

    tsf = TimeSeriesFeature(cut_avg, interval, 'data/train_p2',
                            'data/test_p2', exclude=feature_not_used)

    X_tsf, Y_tsf = tsf.extract()
    print("feature extract over")
    model = train_xgboost(X_tsf, Y_tsf)
    print("train over")
    if hasattr(model, 'feature_importances_'):
        print(model.feature_importances_)
    elif hasattr(model, 'coef_'):
        print(model.coef_)
    F = tsf.check_result(model)

    # tsf = TimeSeriesFeature(cut_avg, interval, 'data/deal_mars_tianchi_full_start_avg',
    #                         'data/pose_data', exclude=feature_not_used)
    #
    # X_tsf, Y_tsf = tsf.extract()
    # print("feature extraction over")
    # xgb = train_bagging_xgboost(X_tsf, Y_tsf)
    # print("train over")
    # xgb_result = tsf.predict_result(xgb)
    # tsf.write_result('data/mars_tianchi_artist_plays_predict.csv', xgb_result)
    #
    # print("write over")
