"""
parse BBox label tool annotations
https://github.com/puzzledqs/BBox-Label-Tool
"""

import csv
import os
import sys
import xml.etree.ElementTree as ET
import glob
from collections import defaultdict
import numpy as np

def _pp(l): # pretty printing 
    for i in l: print('{}: {}'.format(i,l[i]))

def bbox_label_tool(ANN, pick, exclusive = False):
    print('Parsing for {} {}'.format(
            pick, 'exclusively' * int(exclusive)))

    dumps = list()
    cur_dir = os.getcwd()
    os.chdir(ANN)
    annotations = os.listdir('.')
    annotations = glob.glob(str(annotations)+'*.txt')
    size = len(annotations)

    for i, file in enumerate(annotations):
        # progress bar      
        sys.stdout.write('\r')
        percentage = 1. * (i+1) / size
        progress = int(percentage * 20)
        bar_arg = [progress*'=', ' '*(19-progress), percentage*100]
        bar_arg += [file]
        sys.stdout.write('[{}>{}]{:.0f}%  {}'.format(*bar_arg))
        sys.stdout.flush()
        
        # actual parsing 
        with open(file, 'r') as f:
            csvreader = csv.reader(f, delimiter=' ')
            jpg = file.split('.')[0] + '.jpg'
            w = 1200
            h = 900
            objs = list()
            head = True

            for row in csvreader:
                if head:
                    head = False
                    continue

                current = list()
                name = row[-1]
                if name not in pick:
                        continue

                xn = int(row[0])
                xx = int(row[2])
                yn = int(row[1])
                yx = int(row[3])
                current = [name,xn,yn,xx,yx]
                objs += [current]

        add = [[jpg, [w, h, objs]]]
        dumps += add
            
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

    os.chdir(cur_dir)
    return dumps    

def bbox_label_gt(ANN):
    if not os.path.isdir(ANN):
        raise IOError('not a directory')

    anns = glob.glob(os.path.join(ANN, '*.txt'))

    truth = defaultdict(dict)

    for file in anns:
        name = file.split('/')[-1]
        name = name.split('.')[0]
        with open(file, 'r') as f:
            csvreader = csv.reader(f, delimiter=' ')
            head = True
            for row in csvreader:
                if head:
                    head = False
                    continue


                xmin = int(row[0])
                xmax = int(row[2])
                ymin = int(row[1])
                ymax = int(row[3])
                            
                if name in truth and row[4] in truth[name]:
                    truth[name][row[4]]['bboxs'] = np.vstack((truth[name][row[4]]['bboxs'], np.array([xmin, xmax, ymin, ymax])))
                    truth[name][row[4]]['is_dets'] = np.append(truth[name][row[4]]['is_dets'], 0)

                else:       
                    truth[name][row[4]] = {'bboxs': np.array([[xmin, xmax, ymin, ymax]], dtype=int), \
                                        'is_dets': np.array([0])}
                
    return truth                                    