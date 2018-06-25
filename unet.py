# Varun Ravi
# unet.py

import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import ipdb
import os
import tensorflow as tf


# HYPER-PARAMETERS
IMG_HEIGHT=572
IMG_WIDTH=572
NUM_CHANNELS=1

IMG_HEIGHT_y=388
IMG_WIDTH_y=388
NUM_CHANNELS_y=2


def pywalker(path):
	
	list_files = []

	for root, dirs, files in os.walk(path):
 		for file_ in files:
 			if file_ != '.DS_Store':
 				list_files.append(os.path.join(root, file_))

	return list_files


def conv2d_transpose(x):
	x_shape = tf.shape(x)
	output_shape = tf.stack([x_shape[0], x_shape[1]*2, x_shape[2]*2, x_shape[3]//2])
	
	return tf.nn.conv2d_transpose(x, (3,3, 1, 1), output_shape, strides=[1, 1, 1, 1], padding='VALID')


def down_block(inputs, filters, kernel_size=(3,3), padding='same', strides=(2,2), activation=tf.nn.relu, max_pool_size=(2,2), max_strides=2):

	inputs = tf.layers.max_pooling2d(inputs, max_pool_size, strides, padding)
	inputs = tf.layers.conv2d(inputs, filters, kernel_size, padding=padding, activation=activation)
	inputs = tf.layers.conv2d(inputs, filters, kernel_size, padding=padding, activation=activation)

	return inputs

def up_block(inputs, parallel_down_conv, filters, kernel_size=(3,3), padding='same', activation=tf.nn.relu):

	inputs = tf.layers.conv2d_transpose(inputs, filters, kernel_size, padding=padding)
	inputs = tf.image.resize_nearest_neighbor(inputs, (inputs.get_shape().as_list()[1]*2,inputs.get_shape().as_list()[2]*2))

	inputs = tf.concat([parallel_down_conv[:,:inputs.get_shape().as_list()[1],:inputs.get_shape().as_list()[2],:], inputs], axis=-1)
	
	inputs = tf.layers.conv2d(inputs, 512, (3,3), activation=activation)
	inputs = tf.layers.conv2d(inputs, 512, (3,3), activation=activation)

	return inputs

def train(layers, X_train, y_train):

	x = tf.placeholder(tf.float32, shape=[None, IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS], name='x')
	y = tf.placeholder(tf.float32, shape=[None, IMG_HEIGHT_y, IMG_WIDTH_y, NUM_CHANNELS_y], name='y')

	loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=final, labels=y_train))
	optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-4)
	train_op = optimizer.minimize(loss_op)

	init = tf.global_variables_initializer()
	
	with tf.Session() as sess:
		
		sess.run(init)
		
		_, training_loss = sess.run([train_op, loss_op], feed_dict={x: X, y: y_train})
		print ("Step %d/%d, Training Loss = %d" % (1, 1, training_loss))
	

# -------------------------------------------------------------------
# Function:  unet
# Purpose:   Creates a unet model given the input shape 
# In args:   input_shape
# Out arg: model
def unet(X_train_shape, Y_train_shape):

	down_conv1 = tf.layers.conv2d(X_train_shape, 64, (3,3), activation=tf.nn.relu, padding='same')
	down_conv1 = tf.layers.conv2d(down_conv1, 64, (3,3), activation=tf.nn.relu, padding='same')
	
	down_conv2 = down_block(down_conv1, 128)
	down_conv3 = down_block(down_conv2, 256)
	down_conv4 = down_block(down_conv3, 512)
	down_conv5 = down_block(down_conv4, 1024)
	
	up_conv1 = up_block(down_conv5, down_conv4, 512)
	up_conv2 = up_block(up_conv1, down_conv3, 256)
	up_conv3 = up_block(up_conv2, down_conv2, 128)
	up_conv4 = up_block(up_conv3, down_conv1, 64)
	
	final_layer = tf.layers.conv2d(up_conv4, 2, 1, activation=tf.nn.softmax)

	return final_layer

if __name__ == '__main__':

	X_image = tf.constant(np.zeros((1, 572, 572, 1)), tf.float32)
	y_mask = np.zeros((1, 388, 388, 2))	
	
	final = unet(X_image, y_mask)

	#train(final, X_image, y_mask)


	

	
