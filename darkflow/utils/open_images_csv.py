"""
parse Google's Open Images csv annotations
"""
import os
import sys
import glob
import csv
import functools
from collections import defaultdict
import numpy as np

def _pp(l): # pretty printing 
    for i in l: print('{}: {}'.format(i,l[i]))

# returns data for training
# ANN will be the annotation csv file
def open_images_csv(ANN, pick, exclusive = False):
    print('Parsing for {} {}'.format(
            pick, 'exclusively' * int(exclusive)))

    dumps = []

    with open(ANN, 'r') as f:
        csvreader = csv.reader(f, delimiter=',')

        size = len(list(csvreader))
        f.seek(0)

        prev_img = None
        objs = []
        for i, row in enumerate(csvreader):

            # progress bar      
            sys.stdout.write('\r')
            percentage = 1. * (i+1) / size
            progress = int(percentage * 20)
            bar_arg = [progress*'=', ' '*(19-progress), percentage*100]
            bar_arg += [row[0]]
            sys.stdout.write('[{}>{}]{:.0f}%  {}'.format(*bar_arg))
            sys.stdout.flush()

            # actual parsing
            img_name = row[0]

            if prev_img is not None and prev_img != img_name:
                dumps.append([prev_img, [w, h , objs]])
                objs = []

            w = int(row[7])
            h = int(row[8])

            class_name = row[2]
            if class_name not in pick:
                print("found stray", class_name)
                continue

            xmin = int(float(row[3]) *float(w))
            xmax = int(float(row[4]) * float(w))
            ymin = int(float(row[5]) * float(h))
            ymax = int(float(row[6]) * float(h))
            
            objs.append([class_name, xmin, ymin, xmax, ymax])

            prev_img = img_name
  
    # gather all stats
    stat = dict()
    for dump in dumps:
        all = dump[1][2]
        for current in all:
            if current[0] in pick:
                if current[0] in stat:
                    stat[current[0]]+=1
                else:
                    stat[current[0]] =1

    print('\nStatistics:')
    _pp(stat)
    print('Dataset size: {}'.format(len(dumps)))

    return dumps

# get ground truth in eval.py desired format   
# ANN is the csv file with all the annotations
# each row: [0]img [2]class [3]xmin [4]xmax [5]ymin [6]ymax
def open_images_csv_gt(ANN):
    index = 0
    previous = None
    truth = defaultdict(dict)

    with open(ANN, 'r') as f:
        csvreader = csv.reader(f, delimiter=',')
        for row in csvreader:
            name = row[0].split('.')[0]

            w = row[7]
            h = row[8]

            xmin = int(float(row[3]) * float(w))
            xmax = int(float(row[4]) * float(w))
            ymin = int(float(row[5]) * float(h))
            ymax = int(float(row[6]) * float(h))

            if name in truth and row[2] in truth[name]:
                truth[name][row[2]]['bboxs'] = np.vstack((truth[name][row[2]]['bboxs'], np.array([xmin, xmax, ymin, ymax])))
                truth[name][row[2]]['is_dets'] = np.append(truth[name][row[2]]['is_dets'], 0)
            else:
                truth[name][row[2]] = {'bboxs': np.array([[xmin, xmax, ymin, ymax]], dtype=int), \
                                       'is_dets': np.array([0])}
   
    return truth
