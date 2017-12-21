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
    sprob = float(m['class_scale']) # 1
    sconf = float(m['object_scale']) # 1
    snoob = float(m['noobject_scale']) # 1, reduce loss from this? paper put 0.5
    scoor = float(m['coord_scale']) # 5, increase loss from this, so gradient not 0
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
    
    adjusted_coords_xy = tf.nn.sigmoid(coords[:,:,:,0:2])
    
    ad_wh = tf.exp(tf.clip_by_value(coords[:,:,:,2:4], -1e3, 10)) * np.reshape(anchors, [1, 1, B, 2]) / np.reshape([W, H], [1, 1, 1, 2]) # get sigmoid of xy # cx, cy removed when reading in data already # sqrt(e^prediction * prior) sqrt reason in yolov1. 
    adjusted_coords_wh = tf.sqrt(tf.clip_by_value(ad_wh, 1e-10, 1e5)) # following https://github.com/tensorflow/tensorflow/issues/4914
    coords = tf.concat([adjusted_coords_xy, adjusted_coords_wh], 3) # join back

    # adjusted_c = expit_tensor(net_out_reshape[:, :, :, :, 4]) # sigmoid of to
    adjusted_c = tf.nn.sigmoid(net_out_reshape[:, :, :, :, 4])
    adjusted_c = tf.reshape(adjusted_c, [-1, H*W, B, 1]) # row major order indexing like placeholder input

    adjusted_prob = tf.nn.softmax(net_out_reshape[:, :, :, :, 5:]) # normalize
    adjusted_prob = tf.reshape(adjusted_prob, [-1, H*W, B, C])

    adjusted_net_out = tf.concat([adjusted_coords_xy, adjusted_coords_wh, adjusted_c, adjusted_prob], 3) # xywhc, prob

    wh = tf.square(coords[:,:,:,2:4]) * np.reshape([W, H], [1, 1, 1, 2])
    area_pred = tf.multiply(wh[:,:,:,0], wh[:,:,:,1])
    centers = coords[:,:,:,0:2]
    floor = centers - (wh * .5) # predicted top left
    ceil  = centers + (wh * .5) # predicted bottom right

    # calculate the intersection areas
    intersect_upleft   = tf.maximum(floor, _upleft)
    intersect_botright = tf.minimum(ceil , _botright)
    intersect_wh = intersect_botright - intersect_upleft
    intersect_wh = tf.maximum(intersect_wh, 0.0)
    intersect = tf.multiply(intersect_wh[:,:,:,0], intersect_wh[:,:,:,1]) # area of intersect
    self.check_op.append(tf.check_numerics(intersect, 'intersect'))

    # calculate the best IOU, set 0.0 confidence for worse boxes
    iou = tf.truediv(intersect, _areas + area_pred - intersect) # intersect / all area
    best_box = tf.equal(iou, tf.reduce_max(iou, [2], True)) # reduce in B dim # best_box is array of True/False. True if that index has max in the B dim
    best_box = tf.to_float(best_box) # True to 1.0, False to 0.0
    confs = tf.multiply(best_box, _confs) # * probability an object at that index?
    self.check_op.append(tf.check_numerics(confs, 'confs'))

    # take care of the weight terms
    conid = snoob * (1. - confs) + sconf * confs # scale with noobject and object parameters. # not sure which part of paper
    weight_coo = tf.concat(4 * [tf.expand_dims(confs, -1)], 3) # x 4?
    cooid = scoor * weight_coo # * coordinate scale
    weight_pro = tf.concat(C * [tf.expand_dims(confs, -1)], 3) # x classes? 
    proid = sprob * weight_pro # * class scale

    self.fetch += [_probs, confs, conid, cooid, proid]
    true = tf.concat([_coord, tf.expand_dims(confs, 3), _probs ], 3) # ground truth, not scaled # why confs not _confs
    wght = tf.concat([cooid, tf.expand_dims(conid, 3), proid ], 3) # predictions, scaled

    #### DEBUGGING
    # adjusted_net_out_min = tf.reduce_min(adjusted_net_out[:,:,:,2:4])
    # self.print_op.append(tf.Print(adjusted_net_out_min, [adjusted_net_out_min], message='adjusted_net_out min', summarize=100))
    # adjusted_net_out_argmin = tf.where(tf.equal(adjusted_net_out[:,:,:,2:4], adjusted_net_out_min))
    # self.print_op.append(tf.Print(adjusted_net_out_argmin, [adjusted_net_out_argmin, tf.shape(adjusted_net_out_argmin)], message='adjusted_net_out argmin', summarize=100))
    
    # adjusted_net_out_max = tf.reduce_max(adjusted_net_out[:,:,:,2:4])
    # self.print_op.append(tf.Print(adjusted_net_out_max, [adjusted_net_out_max], message='adjusted_net_out max', summarize=100))
    # adjusted_net_out_argmax = tf.where(tf.equal(adjusted_net_out[:,:,:,2:4], adjusted_net_out_max))
    # self.print_op.append(tf.Print(adjusted_net_out_argmax, [adjusted_net_out_argmax, tf.shape(adjusted_net_out_argmax)], message='adjusted_net_out argmax', summarize=100))

    ####

    print('Building {} loss'.format(m['model']))
    loss = tf.square(adjusted_net_out - true) # loss function in yolo v1. xywh P(object) C
    loss = tf.multiply(loss, wght) # apply the scales
    
    ### DEBUG
    # loss_min = tf.reduce_min(loss[:,:,:,4])
    # self.print_op.append(tf.Print(loss_min, [loss_min], message='loss min', summarize=100))
    # loss_argmin = tf.where(tf.equal(loss[:,:,:,4], loss_min))
    # self.print_op.append(tf.Print(loss_argmin, [loss_argmin, tf.shape(loss_argmin)], message='loss argmin', summarize=100))

    # loss_max = tf.reduce_max(loss[:,:,:,4])
    # self.print_op.append(tf.Print(loss_max, [loss_max], message='loss max', summarize=100))
    # loss_argmax = tf.where(tf.equal(loss[:,:,:,4], loss_max))
    # self.print_op.append(tf.Print(loss_argmax, [loss_argmax, tf.shape(loss_argmax)], message='loss argmax', summarize=100))
    
    # nans = tf.is_nan(loss)
    # points = tf.where(nans)
    # self.print_op.append(tf.Print(points, [points, tf.shape(points)], message="points that are nan in the loss: ", summarize=100))
    ###

    loss = tf.reshape(loss, [-1, H*W*B*(4 + 1 + C)])
    loss = tf.reduce_sum(loss, 1)
    self.loss = tf.reduce_mean(loss) 
    self.check_op.append(tf.check_numerics(self.loss, 'loss'))
    # tf.summary.scalar('{} loss'.format(m['model']), self.loss)