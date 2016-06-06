# -*- coding: utf-8 -*-
__author__ = 'yixuanhe'

avg_days = 3
plays = {}

with open("data/deal_mars_tianchi_full_start_avg", "w") as w:
    with open("data/deal_mars_tianchi_full_start") as f:
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
