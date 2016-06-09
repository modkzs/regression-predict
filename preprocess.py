# -*- coding: utf-8 -*-
__author__ = 'yixuanhe'
import time


def get_time(ts):
    time_array = time.localtime(ts)
    time_str = time.strftime("%Y%m%d", time_array)
    return time_str


def get_test_data(train_name, test_name):
    """
    get test data that we need to test between 9 and 10
    :param train_name:
    :param test_name:
    :return:
    """
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
    """
    cut not necessary data which before data it publish or first listened
    :param origin:
    :param write_file:
    :return:
    """
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


def devide(origin_file, test_file, train_file, date):
    """
    cut train and test data
    :param origin_file:
    :param test_file:
    :param train_file:
    :param date:
    :return:
    """
    with open(origin_file, "r") as origin, open(test_file, "w") as test, open(train_file, "w") as train:
            for line in origin.readlines():
                feature = line.split(" ")
                # deal with the empty row
                if len(feature) == 1:
                    continue
                ts = feature[6]
                # time_array = time.localtime(ts)
                # time_str = time.strftime("%Y%m%d", time_array)

                if ts >= date:
                    test.write(line)
                else:
                    train.write(line)


def avg_data(path, avg_path, avg_days=3):
    """
    get avg data
    :param path:
    :param avg_path:
    :param avg_days:
    :return:
    """
    plays = {}
    with open(path, "w") as w:
        with open(avg_path) as f:
            for l in f.readlines():
                data = l.replace("\n", "").split(" ")
                songid = data[1]
                play = int(data[2])

                tmp = plays.get(songid, [])
                tmp.append(play)
                plays[songid] = tmp

                avg = 0
                num = 0
                for i in plays[songid][-avg_days::]:
                    avg += i
                    num += 1
                avg /= num

                data[2] = str(avg)
                for d in data:
                    w.write(d + " ")
                w.write("\n")


deal_with_data('data/data', 'data/data_start')
devide("data/data_start", "data/test_p2", "data/train_p2", "20150801")
get_test_data('data/data', 'data/pose')











