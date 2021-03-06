import numpy as np
import tensorflow as tf
import cv2

def imcv2_recolor(im, a = .1):
	t = [np.random.uniform()]
	t += [np.random.uniform()]
	t += [np.random.uniform()]
	t = np.array(t) * 2. - 1.

	# random amplify each channel
	im = im * (1 + t * a)
	mx = 255. * (1 + a)
	up = np.random.uniform() * 2 - 1
# 	im = np.power(im/mx, 1. + up * .5)
	im = cv2.pow(im/mx, 1. + up * .5)
	return np.array(im * 255., np.uint8)

def imtf_recolor(im, a = .1):
	t = tf.Variable(tf.random_uniform([3], -1.0, 1.0))

	# random amplify each channel
	amp = tf.add(1., tf.scalar_mul(a, t))
	im_tf = tf.multiply(amp, tf.convert_to_tensor(im, dtype=tf.float32))
	mx = tf.multiply(tf.add(1., a), 255.)
	up = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
	im_tf = tf.pow(tf.divide(im, mx), tf.add(1., tf.scalar_mul(.5, up)))

	# with tf.Session() as sess:
	# 	sess.run(tf.variables_initializer([t, up]))
		
	# 	# print('im', im)
		
	# 	# print('im_tf', sess.run(im_tf))

	# 	im = sess.run(im_tf)

	# return np.array(im * 255., np.uint8)
	return tf.multiply(255., im_tf)

def imcv2_affine_trans(im):
	# Scale and translate
	try:
		h, w, c = im.shape
	except Exception as e:
		print(e)
		raise
	scale = np.random.uniform() / 10. + 1.
	max_offx = (scale-1.) * w
	max_offy = (scale-1.) * h
	offx = int(np.random.uniform() * max_offx)
	offy = int(np.random.uniform() * max_offy)
	
	im = cv2.resize(im, (0,0), fx = scale, fy = scale)
	im = im[offy : (offy + h), offx : (offx + w)]
	flip = np.random.binomial(1, .5)
	if flip: im = cv2.flip(im, 1)
	return im, [w, h, c], [scale, [offx, offy], flip]
