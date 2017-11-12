import tensorflow.contrib.slim as slim
import pickle
import tensorflow as tf
from ..yolo.misc import show
import numpy as np
import os
import math

def expit_tensor(x):
	return 1. / (1. + tf.exp(-x))

def loss(self, net_out):
    """
    Takes net.out and placeholders value
    returned in batch() func above,
    to build train_op and loss
    """
    # meta
    m = self.meta
    sprob = float(m['class_scale'])
    sconf = float(m['object_scale'])
    snoob = float(m['noobject_scale'])
    scoor = float(m['coord_scale'])
    H, W, _ = m['out_size']
    B, C = m['num'], m['classes']
    HW = H * W # number of grid cells
    anchors = m['anchors']

    print('{} loss hyper-parameters:'.format(m['model']))
    print('\tH       = {}'.format(H))
    print('\tW       = {}'.format(W))
    print('\tbox     = {}'.format(m['num']))
    print('\tclasses = {}'.format(m['classes']))
    print('\tscales  = {}'.format([sprob, sconf, snoob, scoor]))

    size1 = [None, HW, B, C]
    size2 = [None, HW, B]

    # return the below placeholders
    _probs = tf.placeholder(tf.float32, size1)
    _confs = tf.placeholder(tf.float32, size2)
    _coord = tf.placeholder(tf.float32, size2 + [4])
    # weights term for L2 loss
    _proid = tf.placeholder(tf.float32, size1)
    # material calculating IOU
    _areas = tf.placeholder(tf.float32, size2)
    _upleft = tf.placeholder(tf.float32, size2 + [2])
    _botright = tf.placeholder(tf.float32, size2 + [2])

    self.placeholders.update({
        'probs':_probs, 'confs':_confs, 'coord':_coord, 'proid':_proid,
        'areas':_areas, 'upleft':_upleft, 'botright':_botright
    })

    # Extract the coordinate prediction from net.out
    net_out_reshape = tf.reshape(net_out, [-1, H, W, B, (4 + 1 + C)])
    coords = net_out_reshape[:, :, :, :, :4] # first 4 of last dim are coordinates
    coords = tf.reshape(coords, [-1, H*W, B, 4]) # change to row major order like placeholder inputs
    adjusted_coords_xy = expit_tensor(coords[:,:,:,0:2]) # get sigmoid of xy
    adjusted_coords_wh = tf.sqrt(tf.exp(coords[:,:,:,2:4]) * np.reshape(anchors, [1, 1, B, 2]) / np.reshape([W, H], [1, 1, 1, 2])) # sqrt(e^prediction * prior) sqrt reason in yolov1? not sure how priors calculated
    coords = tf.concat([adjusted_coords_xy, adjusted_coords_wh], 3) # join back

    adjusted_c = expit_tensor(net_out_reshape[:, :, :, :, 4]) # sigmoid of to
    adjusted_c = tf.reshape(adjusted_c, [-1, H*W, B, 1]) # row major order indexing like placeholder input

    adjusted_prob = tf.nn.softmax(net_out_reshape[:, :, :, :, 5:]) # not sure
    adjusted_prob = tf.reshape(adjusted_prob, [-1, H*W, B, C])

    adjusted_net_out = tf.concat([adjusted_coords_xy, adjusted_coords_wh, adjusted_c, adjusted_prob], 3) # xywhc, prob

    wh = tf.pow(coords[:,:,:,2:4], 2) * np.reshape([W, H], [1, 1, 1, 2]) # pow 2 cos sqrt just now
    area_pred = wh[:,:,:,0] * wh[:,:,:,1] # 0 batch, 1 index, 2 B
    centers = coords[:,:,:,0:2]
    floor = centers - (wh * .5) # predicted top left
    ceil  = centers + (wh * .5) # predicted bottom right

    # calculate the intersection areas
    intersect_upleft   = tf.maximum(floor, _upleft)
    intersect_botright = tf.minimum(ceil , _botright)
    intersect_wh = intersect_botright - intersect_upleft
    intersect_wh = tf.maximum(intersect_wh, 0.0)
    intersect = tf.multiply(intersect_wh[:,:,:,0], intersect_wh[:,:,:,1]) # area of intersect

    # calculate the best IOU, set 0.0 confidence for worse boxes
    iou = tf.truediv(intersect, _areas + area_pred - intersect) # intersect / all areas outside intersect # what if divided by 0? D: # bigger better
    best_box = tf.equal(iou, tf.reduce_max(iou, [2], True)) # reduce in B dim # best_box is array of True/False. True if that index has max in the B dim
    best_box = tf.to_float(best_box) # True to 1.0, False to 0.0
    confs = tf.multiply(best_box, _confs) # * probability an object at that index?

    # take care of the weight terms
    conid = snoob * (1. - confs) + sconf * confs # scale with noobject and object parameters. # not sure which part of paper
    weight_coo = tf.concat(4 * [tf.expand_dims(confs, -1)], 3) # x 4?
    cooid = scoor * weight_coo # * coordinate scale
    weight_pro = tf.concat(C * [tf.expand_dims(confs, -1)], 3) # x classes? 
    proid = sprob * weight_pro # * class scale

    self.fetch += [_probs, confs, conid, cooid, proid]
    true = tf.concat([_coord, tf.expand_dims(confs, 3), _probs ], 3) # ground truth, not scaled # why confs not _confs
    wght = tf.concat([cooid, tf.expand_dims(conid, 3), proid ], 3) # predictions, scaled

    print('Building {} loss'.format(m['model']))
    loss = tf.pow(adjusted_net_out - true, 2) # loss function in yolo v1. xywh P(object) C
    loss = tf.multiply(loss, wght) # not sure?
    loss = tf.reshape(loss, [-1, H*W*B*(4 + 1 + C)])
    loss = tf.reduce_sum(loss, 1)
    self.loss = .5 * tf.reduce_mean(loss) # why need .5?
    tf.summary.scalar('{} loss'.format(m['model']), self.loss)