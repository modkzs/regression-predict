# -*- coding: utf-8 -*-
__author__ = 'yixuanhe'

import numpy as np
import pickle as p
import time, datetime

stime = int(time.mktime(time.strptime('20150301', '%Y%m%d')) / 86400)
T = [
 '20150901',
 '20150902',
 '20150903',
 '20150904',
 '20150905',
 '20150906',
 '20150907',
 '20150908',
 '20150909',
 '20150910',
 '20150911',
 '20150912',
 '20150913',
 '20150914',
 '20150915',
 '20150916',
 '20150917',
 '20150918',
 '20150919',
 '20150920',
 '20150921',
 '20150922',
 '20150923',
 '20150924',
 '20150925',
 '20150926',
 '20150927',
 '20150928',
 '20150929',
 '20150930',
 '20151001',
 '20151002',
 '20151003',
 '20151004',
 '20151005',
 '20151006',
 '20151007',
 '20151008',
 '20151009',
 '20151010',
 '20151011',
 '20151012',
 '20151013',
 '20151014',
 '20151015',
 '20151016',
 '20151017',
 '20151018',
 '20151019',
 '20151020',
 '20151021',
 '20151022',
 '20151023',
 '20151024',
 '20151025',
 '20151026',
 '20151027',
 '20151028',
 '20151029',
 '20151030']

songs = dict()
artists = dict()

# f1 = open('data/mars_tianchi_songs.csv', 'r')
# f2 = open('data/mars_tianchi_user_actions.csv', 'r')
# f3 = open('data/artist_feature.txt', 'r')
# fs = open('data/lyl/song_data.npy', 'wb')
# fa = open('data/lyl/artist_data.npy', 'wb')
#
# for line in f3:
#     d = line.split(' ')
#     a_id = d[0]
#     cc = dict()
#     cc['play_avg_num'] = float(d[2].split(':')[1])
#     cc['user_number'] = int(d[4].split(':')[1])
#     cc['avg_time_during'] = int(d[6].split(':')[1])
#     cc['gender'] = int(d[8].split(':')[1])
#     cc['a_id'] = a_id
#     cc['play_seq'] = np.zeros([183, 3])
#     artists[a_id] = cc
#
# for line in f1:
#     d = line.split(',')
#     s_id = d[0]
#     cc = dict()
#     int_time = int(time.mktime(time.strptime(d[2], '%Y%m%d')) / 86400)
#     cc['publish_time_int'] = int_time - stime
#     cc['publish_time_str'] = d[2]
#     cc['song_init_plays'] = int(d[3])
#     cc['language'] = int(d[4])
#     cc['play_seq'] = np.zeros([183, 3])
#     cc['start_no_zero'] = 183
#     cc['artist'] = artists[d[1]]
#     cc['s_id'] = s_id
#     songs[s_id] = cc
#
# for line in f2:
#     d = line.split(',')
#     s_id = d[1]
#     cc = songs[s_id]
#     dd = int(time.mktime(time.strptime(d[4].strip(), '%Y%m%d')) / 86400) - stime
#     if d[3] == '1':
#         cc['play_seq'][dd, 0] += 1
#         cc['start_no_zero'] = min(dd, cc['start_no_zero'])
#     elif d[3] == '2':
#         cc['play_seq'][dd, 1] += 1
#     elif d[3] == '3':
#         cc['play_seq'][dd, 2] += 1
#
# for k, v in songs.items():
#     artist = v['artist']
#     artist['play_seq'] += v['play_seq']
#
#
# p.dump(songs, fs)
# p.dump(artists, fa)
# f1.close()
# f2.close()
# f3.close()
# fa.close()
# fs.close()


pkl_file = open('data/lyl/artist_data.npy', 'rb')
artists = p.load(pkl_file)
pkl_file.close()
f = open('data/lyl/mars_tianchi_artist_plays_predict_5.csv', 'w')

for k, v in artists.items():
    p = np.mean(v['play_seq'][-7:, 0])
    for i in T:
        s = k + ',' + str(int(p) - 10) + ',' + i + '\n'
        f.write(s)
f.close()


