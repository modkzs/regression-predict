# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod
import math
import time
from . import global_setting
import xgboost as xgb
import numpy as np
from .feature_extract import test_model
import copy
__author__ = 'yixuanhe'


class BasicFeature(metaclass=ABCMeta):
    """
    Basic class used in feature extraction.
    value_func : the function casting data into its own type like int
    line_func: the function dealing with single line, which means no need to deal with last line
    time_func: the function dealing with time series. we may need output or input before, deal it here
    global_func: the func dealing with whole sample space like avg, min, max
    """
    def __init__(self, cut_avg):
        self.times = {}
        self.avg = {}
        self.cut_avg = cut_avg
        self.is_rmse = False
        self.rmse = 0

    def value_func(self, data, flag):
        if self.cut_avg:
            data[2] = float(data[2]) - self.avg.get(data[1], 0)
        else:
            data[2] = float(data[2])
        data[7] = math.exp(10 - abs(float(data[7])))
        data[5] = time.mktime(time.strptime(data[6] + " 0:00:00", '%Y%m%d %H:%M:%S'))
        data[6] = time.mktime(time.strptime(data[7] + " 0:00:00", '%Y%m%d %H:%M:%S'))
        for i in range(2, len(data)):
            data[i] = float(data[i])

        return data, True

    def line_func(self, data, flag):
        for l in global_setting.l9:
            if data[9] == l:
                data.append(1)
            else:
                data.append(0)

        for l in global_setting.l10:
            if data[10] == l:
                data.append(1)
            else:
                data.append(0)

        if data[5] in global_setting.holiday:
            data.append(1)
        else:
            data.append(0)

        return data, True

    def time_func(self, data):
        pass

    @abstractmethod
    def global_func(self, data):
        pass

    def deal_predict(self, y, id, actual):
        if self.is_rmse:
            self.rmse += (y-actual)**2
        return y

    def get_avg(self, path, date='20150701'):
        with open(path) as f:
            for l in f.readlines():
                data = l.split(" ")
                if data[6] > date:
                    songid = data[1]
                    play = float(data[2])
                    self.avg[songid] = self.avg.get(songid, 0) + play

        for key in self.avg:
            self.avg[key] /= 31.0

        return self.avg

    def get_times(self, path):
        with open(path) as f:
            i = 0
            for l in f.readlines():
                i += 1
                data = l.split(" ")
                song_id = data[1]
                plays = int(data[2])
                tmp = self.times.get(song_id, [])
                tmp.append(plays)
                self.times[song_id] = tmp

    @staticmethod
    def single_predict(model, X):
        if model.__class__.__name__ == "Booster":
            dtest = xgb.DMatrix(X)
        else:
            dtest = np.array(X).reshape(1, (len(X)))

        y = model.predict(dtest)

        # val = math.floor(y[0]+0.5)
        return int(y)

    def get_rmse(self, model):
        self.is_rmse = True
        self.test(model)
        return self.rmse

    def check_result(self, model, result=None):
        if result is None:
            times = copy.deepcopy(self.times)
            predict = self.test(model)
            self.times = times
        else:
            predict = result

        plays_prediction = {}
        plays_actual = {}

        with open(self.test_path) as f:
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
            sigma[artist] = sigma.get(artist, 0) + ((math.floor(plays_prediction.get(k, 0.0)+0.5) - plays_actual.get(k, 0.0)) / plays_actual.get(k, 0.0)) ** 2
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

        print(model.__class__.__name__)
        print(F)
        print(total)

        return F

    def predict_result(self, model):
        times = copy.deepcopy(self.times)
        predict = self.test(model)
        self.times = times

        return predict

    def check(self, result):
        plays_prediction = result
        plays_actual = {}

        with open(self.test_path) as f:
            i = 0
            for l in f.readlines():
                songid = l.split(" ")[1]
                real = float(l.split(" ")[2])
                artist = l.split(" ")[0]
                date = l.split(" ")[6]
                key = artist + "-" + date
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

    def write_result(self, write_path, predict):
        plays_prediction = {}

        with open(self.test_path) as f:
            i = 0
            for l in f.readlines():
                songid = l.split(" ")[1]
                artist = l.split(" ")[0]
                date = l.split(" ")[6]
                key = artist + "-" + date
                plays_prediction[key] = plays_prediction.get(key, 0) + predict[i]
                i += 1

        with open(write_path, "w") as f:
            for k in plays_prediction:
                artist, date = k.split("-", 1)
                play = int(math.floor(plays_prediction.get(k, 0.0)+0.5))
                f.write(artist + "," + str(play) + "," + date + "\n")

    def models_test(self, model, X):
        y = self.single_predict(model, X)
        return y

    def test(self, model):
        predict = lambda x: self.models_test(model, x)
        return test_model(self.test_path, self, predict=predict, exclude=self.exclude)
