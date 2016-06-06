# -*- coding: utf-8 -*-
__author__ = 'yixuanhe'
import time


def get_time(ts):
    time_array = time.localtime(ts)
    time_str = time.strftime("%Y%m%d", time_array)
    return time_str


def get_test_data(train_name, test_name):
    songs = {}
    with open(train_name) as f:
        for l in f.readlines():
            data = l.replace("\n", "").split(" ")
            id = data[1]
            t = data[6]

            if id not in songs:
                songs[id] = l
            else:
                r_time = data[6]
                if t > r_time:
                    songs[id] = l

    with open(test_name, "w") as f:
        begin = time.mktime(time.strptime("2015-09-01 0:00:00", '%Y-%m-%d %H:%M:%S'))
        day = 24*60*60
        for k in songs:
            l = songs[k]
            data = l.split(" ")
            t1 = time.mktime(time.strptime(data[7] + " 0:00:00", '%Y%m%d %H:%M:%S'))
            # ts = int(data[5])
            for i in range(0, 60):
                t = int(begin + i*day)
                data[6] = get_time(t)
                data[5] = str((t - t1)/day)
                w = ""
                for d in data:
                    w += d + " "
                f.write(w.strip()+"\n")


def deal_with_data(origin, write_file):
    play_time = {}
    with open(origin) as f:
        with open(write_file, "w") as w:
            for l in f.readlines():
                data = l.split(" ")
                songid = data[1]
                play = int(data[2])
                cur = data[6]
                begin = data[7]

                if play > 0 or begin <= cur:
                    if songid not in play_time:
                        play_time[songid] = cur
                    elif play_time[songid] < cur:
                        play_time[songid] = cur

                if songid not in play_time:
                    continue

                if cur >= play_time[songid]:
                    w.write(l)


deal_with_data('data/deal_mars_tianchi_full', 'data/deal_mars_tianchi_full_start')
# get_test_data("data/train_full_start", "data/test_full_start")










