# -*- coding: utf-8 -*
import os
import sys
import timeit
import pandas as pd

from numpy import array
'''
data loading and preview
'''
start_time = timeit.default_timer()

# data loading using pandas
# show data sketch
# with open("../../data/fresh_comp_offline/tianchi_fresh_comp_train_user.csv", 'r') as data_file_user:
#     chunks_user = pd.read_csv(data_file_user, iterator = True))
# with open("../../data/fresh_comp_offline/tianchi_fresh_comp_train_item.csv", mode = 'r') as data_file_item:
#     chunks_item = pd.read_csv(data_file_item, iterator = True)
# chunk_user = chunks_user.get_chunk(5)
# chunk_item = chunks_item.get_chunk(5)
# print(chunk_user)
# print(chunk_item)


'''
data pre_analysis
'''

################################
# calculation of CTR
################################

count_all = 0
count_4 = 0  # the count of behavior_type = 4
for df in pd.read_csv(open("sample_train_user.csv", 'r'),
                      chunksize=100000):
    try:
        count_user = df['behavior_type'].value_counts()
        count_all += count_user[1] + count_user[2] + count_user[3] + count_user[4]
        count_4 += count_user[4]
    except StopIteration:
        print("Iteration is stopped.")
        break
# CTR
ctr = count_4 / count_all
print(ctr)

################################
# visualization month record based on date(11-18->12-18)
################################

count_day = {}  # using dictionary for date-count pairs
for i in range(31):  # for speed up the program, initial dictionary here
    if i <= 12:
        date = '2014-11-%d' % (i + 18)
    else:
        date = '2014-12-%d' % (i - 12)
    count_day[date] = 0

batch = 0
dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d %H')
for df in pd.read_csv(open("sample_train_user.csv", 'r'),
                      parse_dates=['time'], index_col=['time'], date_parser=dateparse,
                      chunksize=100000):
    try:
        for i in range(31):
            if i <= 12:
                date = '2014-11-%d' % (i + 18)
            else:
                date = '2014-12-%d' % (i - 12)

            print(df)
            count_day[date] += df['2014-11-18'].shape[0]
        batch += 1
        print('chunk %d done.' % batch)

    except StopIteration:
        print("finish data process")
        break
