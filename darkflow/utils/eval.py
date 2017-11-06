# Evalute recall for open images
import numpy as np
import csv
import json

import argparse
import os
import glob
import functools

def get_images_detection(out_path, confidence_thres):
    files = os.path.join(out_path, '*.json')
    files = glob.glob(files)

    f_names = []
    det = {}

    for f in files:
        name = f.split('/')[-1]
        name = name.split('.')[0]
        f_names.append(name)

        with open(f, 'r') as j:
            data = json.load(j)
            det[name] = {}
            for obj in data:
                if obj['confidence'] < confidence_thres:
                    continue

                if obj['label'] in det[name]:
                    det[name][obj['label']].append([obj['topleft']['x'], 
                        obj['bottomright']['x'], obj['topleft']['y'], obj['bottomright']['y']])
                else:
                    det[name].update({obj['label']: [[obj['topleft']['x'], 
                        obj['bottomright']['x'], obj['topleft']['y'], obj['bottomright']['y']]]})

    return f_names, det

# ann_csv row: [0]img [2]class [3]xmin [4]xmax [5]ymin [6]ymax
# return dict[img][obj][bbox1, bbox2, ...]
def get_ground_truth(ann, files):
    # files = sorted(files, key=functools.cmp_to_key(cmp))
    # print(files)
    truth = {}

    if ann[-1] != '/': ann += '/'
    annotations = glob.glob(str(ann)+'*.txt')

    for a in annotations:
        with open(a, 'r') as f:
            csvreader = csv.reader(f, delimiter=' ')
            obj = {}
            name = (a.split('/')[-1]).split('.')[0]
            # print(name)
            head = True
            for row in csvreader:

                if head:
                    head = False
                    continue
                
                xmin = int(row[0])
                xmax = int(row[2])
                ymin = int(row[1])
                ymax = int(row[3])

                if row[4] in obj:
                    obj[row[4]]. append([xmin, xmax, ymin, ymax])
                else:
                    obj.update({row[4]: [[xmin, xmax, ymin, ymax]]})

                #     if name in truth:
                #         if row[2] in truth[name]:
                #             truth[name][row[2]].append(list(map(int, [xmin, xmax, ymin, ymax])))
                #         else:
                #             truth[name].update({row[2]: [list(map(int, [xmin, xmax, ymin, ymax]))]})
                #     else:
                #         truth[name] = {row[2]: [list(map(int, [xmin, xmax, ymin, ymax]))]}
                
                # elif cmp(name, files[index]) > 0:
                #     while cmp(name, files[index]) > 0:
                #         index += 1
                #     continue
            truth[name] = obj

    # pp(truth)
    return truth

def get_recall(truth, det, overlap_thres):
    tp = {}
    fp = {}
    objs = {}
    for k in truth:
        tp.update({k: 0})
        fp.update({k: 0})
        objs.update({k: 0})

    for img in det:
        for obj_n in det[img]:
            if img in truth:
                if obj_n in truth[img]:
                    taken = set()
                    for bb_det in det[img][obj_n]:
                        found = False
                        for i,bb_truth in enumerate(truth[img][obj_n]):
                            if i in taken:
                                # print(img, obj_n, taken)
                                continue

                            # bb_truth = truth[img][obj]
                            # bb_det = det[img][obj]

                            # compute overlap
                            # intersection
                            ixmin = np.maximum(bb_truth[0], bb_det[0])
                            ixmax = np.minimum(bb_truth[1], bb_det[1])
                            iymin = np.maximum(bb_truth[2], bb_det[2])
                            iymax = np.maximum(bb_truth[3], bb_det[3])
                            iw = np.maximum(ixmax - ixmin + 1., 0.)
                            ih = np.maximum(iymax - iymin + 1., 0.)
                            inter = iw * ih

                            # union
                            union = ((bb_det[1] - bb_det[0] + 1.) * (bb_det[3] - bb_det[2] + 1.) +
                               (bb_truth[1] - bb_truth[0] + 1.) *
                               (bb_truth[3] - bb_truth[2] + 1.) - inter)

                            overlap = inter / union
                            if overlap > overlap_thres:
                                tp[img] += 1
                                found = True
                                taken.add(i)
                                # print(img, obj_n)
                                break
                        if not found:
                            fp[img] += 1
                else:
                    fp[img] += len(det[img][obj_n])

    for img in truth:
        for obj_n in truth[img]:
            # print(img, obj_n, truth[img][obj_n])
            objs[img] += len(truth[img][obj_n])

    # print(tp)
    # print(objs)

    sum_tp = sum(tp.values())
    sum_fp = sum(fp.values())
    all_obj = sum(objs.values())

    print('recall:', sum_tp, '/', all_obj, '   precision:', sum_tp, '/', sum_tp + sum_fp)
    recall = float(sum_tp) / all_obj
    precision = float(sum_tp) / (sum_tp + sum_fp)
    return recall, precision

def pp(truth):
    for k in truth:
        print(k)
        for obj in truth[k]:
            print("    ", obj, truth[k][obj])

def cmp(x, y):
    # num_x = int(x, 16) + 0x200
    # num_y = int(y, 16) + 0x200
    num_x = int(x[9:])
    num_y = int(y[9:])

    return num_x - num_y

def main(ann, out_path, overlap_thres, confidence_thres):
    files, det = get_images_detection(out_path, confidence_thres)
    if len(files) == 0:
        print('no files found')
        return
    truth = get_ground_truth(ann, files)
    recall, precision = get_recall(truth, det, overlap_thres)
    print('Recall:', recall, '   Precision:', precision)
    # print(len(files))

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('ann', help='ground truth annotation')
    p.add_argument('out_path', help='detection results folder')
    p.add_argument('--overlap_thres', '-ot', type=int, default=0.5)
    p.add_argument('--confidence_thres', '-ct', type=float, default=0.03)

    args = p.parse_args()

    main(args.ann, args.out_path, args.overlap_thres, args.confidence_thres)
