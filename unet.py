# Varun Ravi
# unet.py


# from keras.layers.convolutional import Conv2D
# from keras.layers.core import Activation
# from keras.models import Model
from keras.layers import Cropping2D
# from keras.optimizers import SGD
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import ipdb
import os
#from keras.layers import Input
import tensorflow as tf
#ipdb.set_trace()

IMG_HEIGHT=572
IMG_WIDTH=572
NUM_CHANNELS=1

IMG_HEIGHT_y=388
IMG_WIDTH_y=388
NUM_CHANNELS_y=2


def conv2d_transpose(tensor):
	tensor_shape = tf.shape(tensor)
	output_shape = tf.stack([tensor_shape[0], tensor_shape[1]*2, tensor_shape[2]*2, tensor_shape[3]//2])
	
	return tf.nn.conv2d_transpose(tensor, (3,3, 1, 1), output_shape, strides=[1, 1, 1, 1], padding='VALID')

# -------------------------------------------------------------------
# Function:  pywalker
# Purpose:   Takes in a path and iterates through it to find all the
#			 files within. stores the path to each file as a string
#			 inside a list. 
# In args:   path
# Out arg: list_files. 
def pywalker(path):
	
	list_files = []

	for root, dirs, files in os.walk(path):
 		for file_ in files:
 			if file_ != '.DS_Store':
 				list_files.append(os.path.join(root, file_))

	return list_files


# -------------------------------------------------------------------
# Function:  crop_img
# Purpose:   Crops a tensor given the start and end of each 
#			 height and width. 
# In args:   hi_1 = start of cropping of height
#		     hi_2 = end of cropping of height
#			 wi_1 = start of cropping of width
#			 wi_2 = end of cropping of width
#			 tensor = The tensor to perform the operation on
# Out arg: tensor. 
def crop_img(hi_2, wi_2, tensor, wi_1=0, hi_1=0):

	tensor = Cropping2D(cropping=((hi_1, hi_2), (wi_1, wi_2)))(tensor)
	return tensor

# -------------------------------------------------------------------
# Function:  sub_height
# Purpose:   Subtracts the height of two tensors 
# In args:   tensor_1, tensor_2
# Out arg: hi
def sub_height(tensor_1, tensor_2):

	hi = tensor_2.get_shape().as_list()[1] - tensor_1.get_shape().as_list()[1]
	return hi

# -------------------------------------------------------------------
# Function:  sub_width
# Purpose:   Subtracts the width of two tensors 
# In args:   tensor_1, tensor_2
# Out arg: wi
def sub_width(tensor_1, tensor_2):
	
	wi = tensor_2.get_shape().as_list()[2] - tensor_1.get_shape().as_list()[2]
	return wi

# -------------------------------------------------------------------
# Function:  unet
# Purpose:   Creates a unet model given the input shape 
# In args:   input_shape
# Out arg: model
def unet(X_train, Y_train, padding = "valid", activation=tf.nn.relu, EPOCHS=5, BATCH_SIZE=1):
	#ipdb.set_trace()	
	## features
	x = tf.placeholder(tf.float32, shape=[None, IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS], name='x')

	## labels
	y = tf.placeholder(tf.float32, shape=[None, IMG_HEIGHT_y, IMG_WIDTH_y, NUM_CHANNELS_y], name='y')

	down_conv1 = tf.layers.conv2d(x, 64, (3,3), activation=activation)
	down_conv1 = tf.layers.conv2d(down_conv1, 64, (3,3), activation=activation)

	down_conv2 = tf.layers.max_pooling2d(inputs=down_conv1, pool_size=(2,2), strides=2)
	down_conv2 = tf.layers.conv2d(down_conv2, 128, (3,3), activation=activation)
	down_conv2 = tf.layers.conv2d(down_conv2, 128, (3,3), activation=activation)
	
	down_conv3 = tf.layers.max_pooling2d(inputs=down_conv2, pool_size=(2,2), strides=2)
	down_conv3 = tf.layers.conv2d(down_conv3, 256, (3,3), activation=activation)
	down_conv3 = tf.layers.conv2d(down_conv3, 256, (3,3), activation=activation)
	
	down_conv4 = tf.layers.max_pooling2d(inputs=down_conv3, pool_size=(2,2), strides=2)
	down_conv4 = tf.layers.conv2d(down_conv4, 512, (3,3), activation=activation)
	down_conv4 = tf.layers.conv2d(down_conv4, 512, (3,3), activation=activation)
	
	down_conv5 = tf.layers.max_pooling2d(inputs=down_conv4, pool_size=(2,2), strides=2)
	down_conv5 = tf.layers.conv2d(down_conv5, 1024, (3,3), activation=activation)
	down_conv5 = tf.layers.conv2d(down_conv5, 1024, (3,3), activation=activation)

	up_conv1 = tf.layers.conv2d_transpose(down_conv5, 512, (3,3), strides=(1, 1), padding='same')
	up_conv1 = tf.image.resize_nearest_neighbor(down_conv5, (down_conv5.get_shape().as_list()[1]*2,down_conv5.get_shape().as_list()[2]*2))
	down_conv4 = crop_img(sub_height(up_conv1, down_conv4), sub_width(up_conv1, down_conv4), down_conv4)
	up_conv1 = tf.concat([down_conv4, up_conv1], axis=-1)
	up_conv1 = tf.layers.conv2d(up_conv1, 512, (3,3), activation=activation)
	up_conv1 = tf.layers.conv2d(up_conv1, 512, (3,3), activation=activation)
	
	up_conv2 = tf.layers.conv2d_transpose(up_conv1, 256, (3,3))
	up_conv2 = tf.image.resize_nearest_neighbor(up_conv1, (up_conv1.get_shape().as_list()[1]*2,up_conv1.get_shape().as_list()[2]*2))
	down_conv3 = crop_img(sub_height(up_conv2, down_conv3), sub_width(up_conv2, down_conv3), down_conv3)
	up_conv2 = tf.concat([down_conv3, up_conv2], axis=-1)
	up_conv2 = tf.layers.conv2d(up_conv2, 256, (3,3), activation=activation)
	up_conv2 = tf.layers.conv2d(up_conv2, 256, (3,3), activation=activation)
	
	up_conv3 = tf.layers.conv2d_transpose(up_conv2, 128, (3,3))
	up_conv3 = tf.image.resize_nearest_neighbor(up_conv2, (up_conv2.get_shape().as_list()[1]*2,up_conv2.get_shape().as_list()[2]*2))
	down_conv2 = crop_img(sub_height(up_conv3, down_conv2), sub_width(up_conv3, down_conv2), down_conv2)
	up_conv3 = tf.concat([down_conv2, up_conv3], axis=-1)
	up_conv3 = tf.layers.conv2d(up_conv3, 128, (3,3), activation=activation)
	up_conv3 = tf.layers.conv2d(up_conv3, 128, (3,3), activation=activation)

	up_conv4 = tf.layers.conv2d_transpose(up_conv3, 64, (3,3))
	up_conv4 = tf.image.resize_nearest_neighbor(up_conv3, (up_conv3.get_shape().as_list()[1]*2,up_conv3.get_shape().as_list()[2]*2))
	down_conv1 = crop_img(sub_height(up_conv4, down_conv1), sub_width(up_conv4, down_conv1), down_conv1)
	up_conv4 = tf.concat([down_conv1, up_conv4], axis=-1)
	up_conv4 = tf.layers.conv2d(up_conv4, 64, (3,3), activation=activation)
	up_conv4 = tf.layers.conv2d(up_conv4, 64, (3,3), activation=activation)
	
	final = tf.layers.conv2d(up_conv4, 2, 1, activation=tf.nn.softmax)
	
	loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=final, labels=y))
	optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-4)
	train_op = optimizer.minimize(loss_op)

	init = tf.global_variables_initializer()
	
	with tf.Session() as sess:
		sess.run(init)
		
		_, training_loss = sess.run([train_op, loss_op], feed_dict={x: X_image, y: y_mask})
		print ("Step %d/%d, Training Loss = %d" % (1, 1, training_loss))


if __name__ == '__main__':

	X_image = np.zeros((1, 572, 572, 1))
	y_mask = np.zeros((1, 388, 388, 2))
	
	model = unet(X_image, y_mask, (10,572,572,1))
	

	
	

	#model.summary()

	#X_train = np.append([mpimg.imread(list_img[0])], [mpimg.imread(list_img[1])], axis=0)
	#model.fit(X_train, Y_train, 32)

