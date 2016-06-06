# -*- coding: utf-8 -*-
from abc import abstractmethod
import xgboost as xgb
import numpy as np
import math
__author__ = 'yixuanhe'


class WeightAvgEnsembleModel:
    def __init__(self, X, Y, is_train=True, fold=10, func=[], weight=[]):
        self.funcs = func
        self.models = []
        self.weight = weight
        self.total = 0
        self.is_train = is_train

        if is_train:
            length = len(X)
            train_len = int((fold-1)/fold * length)
            self.train_X = [x[:train_len] for x in X]
            self.train_Y = [y[:train_len] for y in Y]
            self.test_X = [x[train_len:] for x in X]
            self.test_Y = [y[train_len:] for y in Y]
        else:
            self.train_X = X
            self.train_Y = Y
            self.test_X = None
            self.test_Y = None

    def add_model(self, func):
        self.funcs.append(func)

    @abstractmethod
    def fit(self):
        for f in self.funcs:
            model = f(self.train_X, self.train_Y)
            self.models.append(model)
            if self.is_train:
                weight = self.judge(model)
                self.weight.append(weight)
                self.total += weight

    def predict(self, X):
        y = 0
        length = range(self.models)
        for i in range(length):
            y += self.predict(self.models[i], X) * self.weight[i]

        y /= self.total
        return int(math.floor(y+0.5))

    @staticmethod
    def single_predict(model, X):
        if model.__class__.__name__ == "Booster":
            dtest = xgb.DMatrix(X)
        else:
            dtest = np.array(X).reshape(1, (len(X)))

        y = model.predict(dtest)
        return y

    def judge(self, model):
        length = len(self.train_X)
        loss = 0
        for i in length:
            X = self.test_X[i]
            Y = self.test_Y[i]
            predict = self.single_predict(model, X)
            loss += (Y-predict)**2

        return loss

    def get_weight(self):
        return 1/self.weight
