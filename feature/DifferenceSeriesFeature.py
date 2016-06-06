# -*- coding: utf-8 -*-
from .BasicFeature import BasicFeature
from . import global_setting
from .feature_extract import extract_feature
from .feature_extract import test_model
__author__ = 'yixuanhe'


class DifferenceSeriesFeature(BasicFeature):
    def __init__(self, cut_avg, interval):
        self.interval = interval
        super().__init__(cut_avg)
        self.get_avg()

    def time_func(self, data):
        plays = data[global_setting.y_pos]
        songid = data[global_setting.song_pos]

        if self.cut_avg:
            plays -= self.avg.get(songid, 0.0)
        tmp = global_setting.times.get(songid, [])
        tmp.append(plays)
        self.times[songid] = tmp

        for p in self.times[songid][-self.interval - 1:-1:]:
            data.append(p)

        if len(self.times[songid]) < self.interval + 1:
            return data, False

        return data, True

    def global_func(self, data):
        songid = data[global_setting.song_pos]
        data.append[self.avg.get(songid)]
        data.extend[self.times.get(songid)[-self.interval::]]
        return data, True

    def extract(self, path):
        return extract_feature(path, self)

    def test(self, path, predict):
        return test_model(path, self, predict=predict)

    def deal_predict(self, y):
        self.interval.append(y)
