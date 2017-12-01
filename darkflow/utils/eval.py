# Evalute recall for open images
import numpy as np
import csv
import json

import argparse
import os
import glob
import functools
from collections import defaultdict
import matplotlib.pyplot as plt

# return full path of files and file index.
def get_files(out_path):
    if not os.path.isdir(out_path):
        raise IOError('Not a folder', out_path)
    files = os.path.join(out_path, '*.json')
    files = glob.glob(files)

    image_ids = []

    for f in files:
        name = f.split('/')[-1]
        name = name.split('.')[0]
        image_ids.append(name)

    return files, image_ids

# return dict[img][obj][[bbox1, conf1], [bbox2, conf2], ...]
def get_images_detection(files, image_ids):

    det = {}

    for f, name in zip(files, image_ids):

        with open(f, 'r') as j:
            data = json.load(j)
            det[name] = {}
            for obj in data:


                if obj['label'] in det[name]:
                    det[name][obj['label']].append([[obj['topleft']['x'], 
                        obj['bottomright']['x'], obj['topleft']['y'], 
                        obj['bottomright']['y']], obj['confidence']])
                else:
                    det[name].update({obj['label']: [[[obj['topleft']['x'], 
                        obj['bottomright']['x'], obj['topleft']['y'], 
                        obj['bottomright']['y']], obj['confidence']]]})

    return det

# ann_csv row: [0]img [2]class [3]xmin [4]xmax [5]ymin [6]ymax
# return dict
# {img:
#     {obj:
#         {bboxs: [bbox1, [bbox2], ...]
#         is_dets: [isDetected1, isDetected2, ...] (all init to 0)
#         }
#     }
# }
def get_ground_truth(ann_csv, files):
    files = sorted(files, key=functools.cmp_to_key(cmp))
    index = 0
    previous = None
    # truth = {}
    truth = defaultdict(dict)

    with open(ann_csv, 'r') as f:
        csvreader = csv.reader(f, delimiter=',')
        for row in csvreader:
            name = row[0].split('.')[0]
            if previous is not None and name != previous:
                index += 1
                previous = None

            if index == len(files):
                break

            if name == files[index]:
                previous = name

                w = row[7]
                h = row[8]

                xmin = int(float(row[3]) * float(w))
                xmax = int(float(row[4]) * float(w))
                ymin = int(float(row[5]) * float(h))
                ymax = int(float(row[6]) * float(h))

                if name in truth and row[2] in truth[name]:
                    # truth[name][row[2]] = np.vstack((truth[name][row[2]], np.array([[xmin, xmax, ymin, ymax], 0])))
                    truth[name][row[2]]['bboxs'] = np.vstack((truth[name][row[2]]['bboxs'], np.array([xmin, xmax, ymin, ymax])))
                    truth[name][row[2]]['is_dets'] = np.append(truth[name][row[2]]['is_dets'], 0)
                    # else:
                    #     truth[name].update({row[2]: np.expand_dims(np.array([[xmin, xmax, ymin, ymax], 0], dtype=int), axis=0)})
                else:
                    truth[name].update({row[2]: {'bboxs': np.array([[xmin, xmax, ymin, ymax]], dtype=int), \
                                           'is_dets': np.array([0])}})
               
            elif cmp(name, files[index]) > 0:
                while cmp(name, files[index]) > 0:
                    index += 1
                continue


    return truth

# returns a np array [recall, precision]
def get_recall_precision(truth, det, gt_obj_num, overlap_thres, confidence_thres, verbose=False):
    tp = {}
    fp = {}
    for k in truth:
        tp.update({k: 0})
        fp.update({k: 0})

    for img in det:
        for obj_n in det[img]:
            if img in truth:
                if obj_n in truth[img]:
                    for bb_det, conf in det[img][obj_n]:
                        if conf < confidence_thres:
                            continue
                        iou_max = None
                        iou_max_ind = None
                        # print(truth[img][obj_n])
                        for i, (bb_truth, isDetected) in enumerate(zip(truth[img][obj_n]['bboxs'], truth[img][obj_n]['is_dets'])):
                            if bool(isDetected):
                                continue

                            # compute overlap
                            # intersection
                            ixmin = np.maximum(bb_truth[0], bb_det[0])
                            ixmax = np.minimum(bb_truth[1], bb_det[1])
                            iymin = np.maximum(bb_truth[2], bb_det[2])
                            iymax = np.minimum(bb_truth[3], bb_det[3])
                            iw = np.maximum(ixmax - ixmin + 1., 0.)
                            ih = np.maximum(iymax - iymin + 1., 0.)
                            inter = iw * ih

                            # union
                            union = ((bb_det[1] - bb_det[0] + 1.) * (bb_det[3] - bb_det[2] + 1.) +
                               (bb_truth[1] - bb_truth[0] + 1.) *
                               (bb_truth[3] - bb_truth[2] + 1.) - inter)

                            overlap = inter / union
                            if overlap > overlap_thres and (iou_max is None or iou_max < overlap):
                                iou_max = overlap
                                iou_max_ind = i

                        if iou_max is None:
                            fp[img] += 1
                        else:
                            tp[img] += 1
                            truth[img][obj_n]['is_dets'][iou_max_ind] = int(True)
                else:
                    fp[img] += len(det[img][obj_n])

    sum_tp = sum(tp.values())
    sum_fp = sum(fp.values())

    recall = float(sum_tp) / gt_obj_num
    precision = float(sum_tp) / np.maximum(sum_tp + sum_fp, np.finfo(np.float64).eps)

    if isVerbose:
        print('Recall:', sum_tp, '/', gt_obj_num, '   Precision:', sum_tp, '/', sum_tp + sum_fp)
    print('Recall:', round(recall, 5), '   Precision:', round(precision, 5))
   
    return np.array([recall, precision])

# sort by classes
# if list of classes given returns dict {c: num of obj with class c},
# else returns total num of gt objs
def get_gt_obj_num(truth, classes=None):

    if classes is None:
        objs = 0
    else:
        objs = {}

        for c in classes:
            objs.update({c: 0})

    for img in truth:
        for obj_n in truth[img]:
            num = len(truth[img][obj_n]['is_dets'])
            if classes is None:
                objs += num
            else:
                objs[obj_n] += num 
    
    return objs

def get_classes(label):
    classes = []
    with open(label, 'r') as f:
        csvreader = csv.reader(f, delimiter=',')
        for row in csvreader:
            classes.append(row[0])
    return classes

def get_ap(rec, pre):
    # # first append sentinel values at the end
    # mrec = np.concatenate(([0.], rec, [1.]))
    # mpre = np.concatenate(([0.], pre, [0.]))
    # print(len(rec))
    mrec = rec
    mpre = pre

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec

    # plt.figure()
    # plt.title('pre-recall')
    # # plt.scatter(mrec, mpre,s=5)
    # plt.plot(mrec,mpre)
    # plt.figure()
    # plt.title('Recall')
    # plt.plot(mrec)
    # # plt.scatter(np.arange(len(rec)),rec,s=5)
    # plt.figure()
    # plt.title('Precision')
    # plt.plot(mpre)
    # # plt.scatter(np.arange(len(pre)),pre,s=5)
    # plt.show()

    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

# returns np array containing ap for each class (in the order of classes list) + map. len = len(classes) + 1
def get_map(truth, det, classes, gt_obj_num, overlap_thres, verbose=True):
    ap = np.zeros(len(classes))
    for j, c in enumerate(classes):
        ids, bbs_det, _ = reorg_det(det, c)

        num = len(ids)
        tp = np.zeros(num)
        fp = np.zeros(num)

        for i in range(num):
            if c in truth[ids[i]]:
                bb_det = bbs_det[i]
                bb_truth = truth[ids[i]][c]['bboxs']
                isDetected = truth[ids[i]][c]['is_dets']     # check if gt object has been detected
                # print(bb_truth, bb_det)
                #TODO refactor this
                # compute overlaps
                # intersection
                ixmin = np.maximum(bb_truth[:, 0], bb_det[0])
                ixmax = np.minimum(bb_truth[:, 1], bb_det[1])
                iymin = np.maximum(bb_truth[:, 2], bb_det[2])
                iymax = np.minimum(bb_truth[:, 3], bb_det[3])
        
                iw = np.maximum(ixmax - ixmin + 1., 0.)
                ih = np.maximum(iymax - iymin + 1., 0.)
                inters = iw * ih
                # print(ixmin, iymin, ixmax, iymax, iw, ih, inters)

                # union
                uni = ((bb_det[1] - bb_det[0] + 1.) * (bb_det[3] - bb_det[2] + 1.) +
                       (bb_truth[:, 1] - bb_truth[:, 0] + 1.) *
                       (bb_truth[:, 3] - bb_truth[:, 2] + 1.) - inters)

                overlaps = inters / uni
                # print(overlaps)
                overlaps *= (isDetected ^ 1)     # overlap becomes 0 if gt object has been detected
                # print(len(overlaps), len(isDetected))
                # print(overlaps)

                ovmax = np.max(overlaps)
                max_ind = np.argmax(overlaps)

                if ovmax > overlap_thres:
                    tp[i] = 1
                    truth[ids[i]][c]['is_dets'][max_ind] = int(True)
                else:
                    fp[i] = 1

            else:
                fp[i] = 1

        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        recall = tp / float(gt_obj_num[c])
        precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)

        ap[j] = get_ap(recall, precision)
        if verbose:
            print('ap for', c, round(ap[j],5))

    summed = np.sum(ap)
    map_ = summed / len(classes)
    print('map', round(map_,5))

    return np.append(ap, map_)

# keep only detections of class c
# sort by confidence (higher to lower)
# returns 3 separate lists, where the same index corresponds to the same obj:
#     image_ids, bounding boxes, confidence
def reorg_det(det, c):
    ids = np.array([], dtype=str)
    bbs = np.array([], dtype=int).reshape(0,4)
    confs = np.array([], dtype=float)
    
    for img in det:
        if c in det[img]:
            for bbox, conf in det[img][c]:
                ids = np.append(ids, str(img))
                bbs = np.vstack((bbs, [bbox]))
                confs = np.append(confs, float(conf))

    sorted_ind = np.argsort(-confs)
    confs = -1 * np.sort(-confs)
    bbs = bbs[sorted_ind, :]
    ids = ids[sorted_ind]

    return ids, bbs, confs

def pp(truth):
    for k in truth:
        print(k)
        for obj in truth[k]:
            print("    ", obj, truth[k][obj])

def cmp(x, y):
    num_x = int(x, 16) + 0x200
    num_y = int(y, 16) + 0x200

    return num_x - num_y

# return np array of results
# if files not found, IOError raised
def evaluate(ann_csv, path, classes, overlap_thres, cal_recall, confidence_thres, isVerbose):
    print(path)
    try:
        files, image_ids = get_files(path)
        det = get_images_detection(files, image_ids)
    except IOError:
        raise

    if len(files) == 0:
        raise IOError('no files found in', path)

    truth = get_ground_truth(ann_csv, image_ids)
    # print(truth)

    if cal_recall:
        gt_obj_num = get_gt_obj_num(truth)
        return get_recall_precision(truth, det, gt_obj_num, overlap_thres, confidence_thres, verbose=isVerbose)

    else:
        gt_obj_num = get_gt_obj_num(truth, classes=classes)
        return get_map(truth, det, classes, gt_obj_num, overlap_thres, verbose=isVerbose)

    # get_map
    # print(len(files))

def main(ann_csv, out_path, label, overlap_thres, recall, confidence_thres, isVerbose, csv_file):
    classes = get_classes(label)

    if not os.path.isdir(out_path):
        paths = glob.glob(out_path)
    else:
        paths = [out_path]

    if csv_file is not None:
        file = open(csv_file, 'w')
        writer = csv.writer(file, delimiter=',')
        if recall:
            writer.writerow(['Folder', 'Recall', 'Precision'])
        else:
            writer.writerow((['Folder'] + classes + ['mAP']))

    for path in paths:
        try:
            results = evaluate(ann_csv, path, classes, overlap_thres, recall, confidence_thres, isVerbose)
            if csv_file is not None:
                f_name = path.split('/')[-1]
                writer.writerow(np.append(f_name, results))

        except IOError as err:
            print(err, err.__cause__)
            continue 

    file.close()

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('ann_csv', help='ground truth annotation')
    p.add_argument('out_path', help='detection results folder. Can contain wildcards i.e out_\*')
    p.add_argument('--label', type=str, default='labels.txt', 
                    help='class labels (not necessary if only calculate recall')
    p.add_argument('--overlap_thres', '-ot', type=float, default=0.5)
    p.add_argument('--recall', '-r', default=False, action='store_true',
                    help='whether to calculate recall and precision at desired threshold')
    p.add_argument('--confidence_thres', '-ct', type=float, default=0.03) # only used with --recall above
    p.add_argument('--verbose', '-v', default=False, action='store_true')
    p.add_argument('--csv', '-c', type=str, default=None,
                    help='csv file to save to')
    
    args = p.parse_args()
    main(args.ann_csv, args.out_path, args.label, args.overlap_thres, args.recall, args.confidence_thres, args.verbose, args.csv)
