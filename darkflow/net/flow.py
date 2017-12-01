import os
import time
import numpy as np
import tensorflow as tf
import pickle
from multiprocessing.pool import ThreadPool

train_stats = (
    'Training statistics: \n'
    '\tLearning rate : {}\n'
    '\tBatch size    : {}\n'
    '\tEpoch number  : {}\n'
    '\tBackup every  : {}'
)
pool = ThreadPool()

def _save_ckpt(self, step, loss_profile):
    file = '{}-{}{}'
    model = self.meta['name']

    profile = file.format(model, step, '.profile')
    profile = os.path.join(self.FLAGS.backup, profile)
    with open(profile, 'wb') as profile_ckpt: 
        pickle.dump(loss_profile, profile_ckpt)

    ckpt = file.format(model, step, '')
    ckpt = os.path.join(self.FLAGS.backup, ckpt)
    self.say('Checkpoint at step {}'.format(step))
    self.saver.save(self.sess, ckpt)


def train(self):
    loss_ph = self.framework.placeholders
    loss_mva = None; profile = list()
    batches = self.framework.shuffle()

    
    if self.FLAGS.validation:
        val_loss_mva = None;
        val_batches = self.framework.shuffle(training=False)
    
    loss_op = self.framework.loss
    # preprocess_op = self.framework.img_out

    # prev_epoch = None
    for i, (x_batch, datum) in enumerate(batches):

        start = time.time()

        if not i: self.say(train_stats.format(
            self.FLAGS.lr, self.FLAGS.batch,
            self.FLAGS.epoch, self.FLAGS.save
        ))

        step_now = self.FLAGS.load + i + 1
        # test = time.time()
        # # x_batch = self.sess.run(x_batch)
        # batch = list()
        # for img in x_batch:
        #     img_process = self.sess.run(preprocess_op, {self.framework.img_placeholder: img})
        #     batch += [np.expand_dims(img_process, 0)]

        # x_batch = np.concatenate(batch, 0)
        # print('recolor and resize a batch', time.time() - test)

        # TRAINING
        feed_dict = {
            loss_ph[key]: datum[key] 
                for key in loss_ph }
        feed_dict[self.inp] = x_batch
        feed_dict.update(self.feed)

        # fetches = [self.train_op, loss_op, self.summary_op, self.points_print, self.framework.print_op, self.framework.check_op] 
        # fetches = [self.train_op, loss_op, self.framework.print_op, self.framework.check_op] 
        fetches = [self.train_op, loss_op, self.framework.check_op] 
        
        fetched = self.sess.run(fetches, feed_dict)
        loss = fetched[1]
        # summary = fetched[2]
        print("feed dict and sess run", time.time() - start)
        
        if loss_mva is None: loss_mva = loss
        loss_mva = .9 * loss_mva + .1 * loss

        summary = tf.Summary(value=[
            tf.Summary.Value(tag='{}_loss'.format(self.meta['model']), simple_value=loss_mva), 
        ])

        self.writer.add_summary(summary, step_now)

        form = 'Epoch {} - step {} - loss {} - moving ave loss {}'
        self.say(form.format(self.meta['curr_epoch'], step_now, loss, loss_mva))
        
        profile += [(loss, loss_mva)]
        # ckpt = (i+1) % (self.FLAGS.save // self.FLAGS.batch)
        # ckpt = (i+1) % (self.FLAGS.save)
        ckpt = (step_now) % (self.FLAGS.save)
        args = [step_now, profile]
        if not ckpt: _save_ckpt(self, *args)

        # VALIDATION
        # if prev_epoch is None or prev_epoch != self.meta['curr_epoch']:
        # if self.FLAGS.validation and self.meta['curr_epoch'] != 0 and i % (self.FLAGS.val_step * self.meta['batch_per_epoch']) == 0:
        if self.FLAGS.validation and i % (self.FLAGS.val_step * self.meta['batch_per_epoch']) == 0:
            val_loss = 0.0

            for j, (x_batch, datum) in enumerate(val_batches):
        
                feed_dict = {
                    loss_ph[key]: datum[key] 
                        for key in loss_ph }
                feed_dict[self.inp] = x_batch
                # feed_dict.update(self.feed)

                fetches = [loss_op] 
                fetched = self.sess.run(fetches, feed_dict)
                loss = fetched[0]
                val_loss += loss
                
                form = '{} Validation - epoch {} - step {} - batch loss {}'
                self.say(form.format(j, self.meta['curr_epoch'], step_now, loss))

                if j+1 == self.meta['val_batch_per_epoch']:
                    break

            val_loss /= float(self.meta['val_batch_per_epoch'])
                   
            if val_loss_mva is None: val_loss_mva = val_loss
            val_loss_mva = .9 * val_loss_mva + .1 * val_loss

            summary = tf.Summary(value=[
                tf.Summary.Value(tag='{}_loss'.format(self.meta['model']), simple_value=val_loss), 
            ])

            self.val_writer.add_summary(summary, step_now)  

            form = 'Validation - epoch {} - step {} - loss {} - ave loss {}'
            self.say(form.format(self.meta['curr_epoch'], step_now, val_loss, val_loss_mva))

    if ckpt: _save_ckpt(self, *args)

def return_predict(self, im):
    assert isinstance(im, np.ndarray), \
				'Image is not a np.ndarray'
    h, w, _ = im.shape
    im = self.framework.resize_input(im)
    this_inp = np.expand_dims(im, 0)
    feed_dict = {self.inp : this_inp}

    out = self.sess.run(self.out, feed_dict)[0]
    boxes = self.framework.findboxes(out)
    threshold = self.FLAGS.threshold
    boxesInfo = list()
    for box in boxes:
        tmpBox = self.framework.process_box(box, h, w, threshold)
        if tmpBox is None:
            continue
        boxesInfo.append({
            "label": tmpBox[4],
            "confidence": tmpBox[6],
            "topleft": {
                "x": tmpBox[0],
                "y": tmpBox[2]},
            "bottomright": {
                "x": tmpBox[1],
                "y": tmpBox[3]}
        })
    return boxesInfo

import math

def predict(self):
    inp_path = self.FLAGS.imgdir
    all_inps = os.listdir(inp_path)
    all_inps = [i for i in all_inps if self.framework.is_inp(i)]
    if not all_inps:
        msg = 'Failed to find any images in {} .'
        exit('Error: {}'.format(msg.format(inp_path)))

    batch = min(self.FLAGS.batch, len(all_inps))

    # predict in batches
    n_batch = int(math.ceil(len(all_inps) / batch))
    for j in range(n_batch):
        from_idx = j * batch
        to_idx = min(from_idx + batch, len(all_inps))

        # collect images input in the batch
        inp_feed = list(); new_all = list()
        this_batch = all_inps[from_idx:to_idx]
        for inp in this_batch:
            new_all += [inp]
            this_inp = os.path.join(inp_path, inp)
            this_inp = self.framework.preprocess(this_inp)
            expanded = np.expand_dims(this_inp, 0)
            inp_feed.append(expanded)
        this_batch = new_all

        # Feed to the net
        feed_dict = {self.inp : np.concatenate(inp_feed, 0)}    
        self.say('Forwarding {} inputs ...'.format(len(inp_feed)))
        start = time.time()
        out = self.sess.run(self.out, feed_dict)
        stop = time.time(); last = stop - start
        self.say('Total time = {}s / {} inps = {} ips'.format(
            last, len(inp_feed), len(inp_feed) / last))

        # Post processing
        self.say('Post processing {} inputs ...'.format(len(inp_feed)))
        start = time.time()
        pool.map(lambda p: (lambda i, prediction:
            self.framework.postprocess(
               prediction, os.path.join(inp_path, this_batch[i])))(*p),
            enumerate(out))
        stop = time.time(); last = stop - start

        # Timing
        self.say('Total time = {}s / {} inps = {} ips'.format(
            last, len(inp_feed), len(inp_feed) / last))
