# -*- coding: utf-8 -*-
import time
__author__ = 'yixuanhe'


def devide(origin_file, test_file, train_file, date):
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


# devide("data/deal_mars_tianchi_full", "data/test_full", "data/train_data_full", "20150801")
# devide("data/deal_mars_tianchi_full_no_start", "data/test_full_no_start", "data/train_data_full_no_start", "20150801")
# devide("data/deal_mars_tianchi_cut", "data/test_cut", "data/train_data_cut", "20150801")
devide("data/deal_mars_tianchi_full_start_avg", "data/test_full_start_avg", "data/train_data_full_start_avg", "20150801")
