# -*- coding: utf-8 -*-
import math

import operator

__author__ = 'yixuanhe'


def get_daily_play(filename):
    play = {}

    with open(filename) as f:
        for l in f.readlines():
            data = l.split(",")
            play[data[0] + "-" + data[2]] = float(data[1])

    return play


def get_F(base, cur):
    plays_actual = get_daily_play(base)
    plays_prediction = get_daily_play(cur)

    sigma = {}
    Phi = {}
    for k in plays_prediction:
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


get_F("data/best.csv", "data/mars_tianchi_artist_plays_predict_ensemble.csv")
