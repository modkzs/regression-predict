# -*- coding: utf-8 -*-
from .BasicFeature import BasicFeature
from . import global_setting
from .feature_extract import extract_feature
from .feature_extract import test_model
import math
import copy
__author__ = 'yixuanhe'


class TimeSeriesFeature(BasicFeature):
    def __init__(self, cut_avg, interval, train_path, test_path, date="20150701", exclude=[]):
        super().__init__(cut_avg)
        self.interval = interval
        self.get_avg(train_path, date=date)
        self.train_path = train_path
        self.test_path = test_path
        self.exclude = exclude

    def time_func(self, data):
        plays = data[global_setting.y_pos]
        songid = data[global_setting.song_pos]

        tmp = self.times.get(songid, [])
        tmp.append(plays)
        self.times[songid] = tmp

        if len(self.times[songid]) < self.interval + 1:
            return False

        return True

    def global_func(self, data):
        songid = data[global_setting.song_pos]
        data.append(self.avg.get(songid, 0.0))
        times = self.times.get(songid, [0 for i in range(self.interval)])
        if len(times) < self.interval:
            for i in range(self.interval - len(self.times[songid])):
                data.append(0)

        data.extend(times[-self.interval::])

        return data, True

    def extract(self):
        return extract_feature(self.train_path, self, exclude=self.exclude)

    def extract_eval(self, path):
        times = copy.copy(self.times)
        X, Y = extract_feature(path, self, exclude=self.exclude)
        self.times = times
        return X, Y

    def deal_predict(self, y, id, actual):
        times = self.times.get(id, [0 for i in range(self.interval)])
        times.append(y)
        self.times[id] = times

        if self.cut_avg:
            y += self.avg.get()

        return super().deal_predict(y, id, actual)
