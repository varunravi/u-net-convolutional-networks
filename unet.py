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
def unet(input_shape, padding = "valid", activation=tf.nn.relu):
	
	Y_train = np.zeros([301088, 1, 1, 2])

	inputs = tf.constant(0., shape=input_shape)

	down_conv1 = tf.layers.conv2d(inputs=inputs, filters=64, kernel_size=(3,3), padding=padding, activation=activation)
	down_conv1 = tf.layers.conv2d(inputs=down_conv1, filters=64, kernel_size=(3,3), padding=padding, activation=activation)

	down_conv2 = tf.layers.max_pooling2d(inputs=down_conv1, pool_size=(2,2), strides=2)
	down_conv2 = tf.layers.conv2d(down_conv2, 128, (3,3), padding = padding, activation=activation)
	down_conv2 = tf.layers.conv2d(down_conv2, 128, (3,3), padding = padding, activation=activation)
	
	
	
	down_conv3 = tf.layers.max_pooling2d(inputs=down_conv2, pool_size=(2,2), strides=2)
	down_conv3 = tf.layers.conv2d(down_conv3, 256, (3,3), padding = padding, activation=activation)
	down_conv3 = tf.layers.conv2d(down_conv3, 256, (3,3), padding = padding, activation=activation)
	
	
	
	down_conv4 = tf.layers.max_pooling2d(inputs=down_conv3, pool_size=(2,2), strides=2)
	down_conv4 = tf.layers.conv2d(down_conv4, 512, (3,3), padding = padding, activation=activation)
	down_conv4 = tf.layers.conv2d(down_conv4, 512, (3,3), padding = padding, activation=activation)
	
	

	down_conv5 = tf.layers.max_pooling2d(inputs=down_conv4, pool_size=(2,2), strides=2)
	down_conv5 = tf.layers.conv2d(down_conv5, 1024, (3,3), padding = padding, activation=activation)
	down_conv5 = tf.layers.conv2d(down_conv5, 1024, (3,3), padding = padding, activation=activation)

	up_conv1 = tf.image.resize_nearest_neighbor(down_conv5, (down_conv5.get_shape().as_list()[1]*2,down_conv5.get_shape().as_list()[2]*2))
	down_conv4 = crop_img(sub_height(up_conv1, down_conv4), sub_width(up_conv1, down_conv4), down_conv4)
	up_conv1 = tf.concat([down_conv4, up_conv1], axis=-1)
	up_conv1 = tf.layers.conv2d(up_conv1, 512, (3,3), padding = padding, activation=activation)
	up_conv1 = tf.layers.conv2d(up_conv1, 512, (3,3), padding = padding, activation=activation)
	

	up_conv2 = tf.image.resize_nearest_neighbor(up_conv1, (up_conv1.get_shape().as_list()[1]*2,up_conv1.get_shape().as_list()[2]*2))
	down_conv3 = crop_img(sub_height(up_conv2, down_conv3), sub_width(up_conv2, down_conv3), down_conv3)
	up_conv2 = tf.concat([down_conv3, up_conv2], axis=-1)
	up_conv2 = tf.layers.conv2d(up_conv2, 256, (3,3), padding=padding, activation=activation)
	up_conv2 = tf.layers.conv2d(up_conv2, 256, (3,3), padding=padding, activation=activation)
	
	
	#convolution transpose
	up_conv3 = tf.image.resize_nearest_neighbor(up_conv2, (up_conv2.get_shape().as_list()[1]*2,up_conv2.get_shape().as_list()[2]*2))
	down_conv2 = crop_img(sub_height(up_conv3, down_conv2), sub_width(up_conv3, down_conv2), down_conv2)
	up_conv3 = tf.concat([down_conv2, up_conv3], axis=-1)
	up_conv3 = tf.layers.conv2d(up_conv3, 128, (3,3), padding = padding, activation=activation)
	up_conv3 = tf.layers.conv2d(up_conv3, 128, (3,3), padding = padding, activation=activation)

	

	up_conv4 = tf.image.resize_nearest_neighbor(up_conv3, (up_conv3.get_shape().as_list()[1]*2,up_conv3.get_shape().as_list()[2]*2))
	down_conv1 = crop_img(sub_height(up_conv4, down_conv1), sub_width(up_conv4, down_conv1), down_conv1)
	up_conv4 = tf.concat([down_conv1, up_conv4], axis=-1)
	up_conv4 = tf.layers.conv2d(up_conv4, 64, (3,3), padding = padding, activation=activation)
	up_conv4 = tf.layers.conv2d(up_conv4, 64, (3,3), padding = padding, activation=activation)
	final = tf.layers.conv2d(up_conv4, 2, 1, padding = padding, activation=tf.nn.softmax)

	loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=final, labels=Y_train))
	optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-4)
	train_op = optimizer.minimize(loss_op)


	# Evaluate model
	correct_pred = tf.equal(tf.argmax(final, 1), tf.argmax(np.zeros([2, 388, 388, 2]), 1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

	# Initialize the variables (i.e. assign their default value)
	init = tf.global_variables_initializer()
	
	ipdb.set_trace()

	model = Model(inputs=inputs, outputs=final)
	model.compile(optimizer=SGD(lr = 1e-4), loss='binary_crossentropy', metrics=['accuracy'])



	return model


if __name__ == '__main__':

	X_train = np.array([])
	Y_train = ['dog', 'cat']
	list_img = pywalker('./misc')

	model = unet((2, 572, 572, 572))
	model.summary()

	X_train = np.append([mpimg.imread(list_img[0])], [mpimg.imread(list_img[1])], axis=0)
	model.fit(X_train, Y_train, 32)

	ipdb.set_trace()


	
	
	
