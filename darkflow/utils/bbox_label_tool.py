"""
parse BBox label tool annotations
https://github.com/puzzledqs/BBox-Label-Tool
"""

import os
import sys
import xml.etree.ElementTree as ET
import glob

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
            csvreader = csv.reader(f, delimiter=',')
            jpg = file.split('.')[0] + '.jpg'
            w = 640
            h = 480
            all = list()
            head = True

            for row in csvreader:
                if head:
                    head = False
                    continue

                current = list()
                elms = row.split(' ')
                name = elms[-1]
                if name not in pick:
                        continue

                xn = int(elms[0])
                xx = int(elms[2])
                yn = int(elms[1])
                yx = int(elms[3])
                current = [name,xn,yn,xx,yx]
                all += [current]

        add = [[jpg, [w, h, all]]]
        dumps += add
        in_file.close()

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