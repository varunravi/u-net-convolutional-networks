# Varun Ravi
# unet.py

import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import ipdb
import os
import tensorflow as tf

X_SHAPE = [None,32,32,3]
y_SHAPE = [None,100]
EPOCHS = 1
DATA = 'cifar10'
BATCH_SIZE = 8
DATA = 'cifar100'

def get_data(data):

	if data == 'cifar10':
		(train_data, train_labels), (eval_data, eval_labels) = tf.keras.datasets.cifar10.load_data()
		train_labels = one_hot(train_labels, (50000,10))
		eval_labels = one_hot(eval_labels, (10000,10))

	if data == 'cifar100':
		(train_data, train_labels), (eval_data, eval_labels) = tf.keras.datasets.cifar10.load_data()
		train_labels = one_hot(train_labels, (50000,100))
		eval_labels = one_hot(eval_labels, (50000,100))


	return train_data, train_labels, eval_data, eval_labels

def one_hot(labels, shape):
	one_hot= np.zeros(shape)
	pos=0

	for i in labels:
		one_hot[pos][i] = 1
		pos+=1

	return one_hot

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
	
	return tf.nn.conv2d_transpose(x, (3,3, 1, 1), output_shape, strides=[1, 1, 1, 1], padding='SAME')


def down_block(inputs, filters, kernel_size=(3,3), padding='same', strides=(2,2), activation=tf.nn.relu, max_pool_size=(2,2), max_strides=2):

	inputs = tf.layers.max_pooling2d(inputs, max_pool_size, strides, padding)
	inputs = tf.layers.conv2d(inputs, filters, kernel_size, padding=padding, activation=activation)
	inputs = tf.layers.conv2d(inputs, filters, kernel_size, padding=padding, activation=activation)

	return inputs

def up_block(inputs, parallel_down_conv, filters, kernel_size=(3,3), padding='same', activation=tf.nn.relu):

	inputs = tf.layers.conv2d_transpose(inputs, filters, kernel_size, padding=padding)
	inputs = tf.image.resize_nearest_neighbor(inputs, (inputs.get_shape().as_list()[1]*2,inputs.get_shape().as_list()[2]*2))

	inputs = tf.concat([parallel_down_conv[:,:inputs.get_shape().as_list()[1],:inputs.get_shape().as_list()[2],:], inputs], axis=-1)
	
	inputs = tf.layers.conv2d(inputs, 512, (3,3), activation=activation,  padding='same')
	inputs = tf.layers.conv2d(inputs, 512, (3,3), activation=activation,  padding='same')

	return inputs

def model(x,y):

	down_conv1 = tf.layers.conv2d(x, 64, (3,3), activation=tf.nn.relu, padding='same')
	down_conv1 = tf.layers.conv2d(down_conv1, 64, (3,3), activation=tf.nn.relu, padding='same')
	
	down_conv2 = down_block(down_conv1, 128)
	down_conv3 = down_block(down_conv2, 256)
	down_conv4 = down_block(down_conv3, 512)
	down_conv5 = down_block(down_conv4, 1024)

	up_conv1 = up_block(down_conv5, down_conv4, 512)
	up_conv2 = up_block(up_conv1, down_conv3, 256)
	up_conv3 = up_block(up_conv2, down_conv2, 128)
	up_conv4 = up_block(up_conv3, down_conv1, 64)
	#ipdb.set_trace()

	flatten_layer = tf.contrib.layers.flatten(up_conv4)
	final_layer = tf.layers.dense(inputs=flatten_layer, units=100)
	
	return final_layer


if __name__ == '__main__':

	# dimensions
	x = tf.placeholder(tf.float32, shape=X_SHAPE, name='x')
	y = tf.placeholder(tf.float32, shape=y_SHAPE, name='y')

	train_data, train_labels, eval_data, eval_labels = get_data(DATA)

	final = model(x, y)
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=final, labels=y))
	optimize = tf.train.GradientDescentOptimizer(learning_rate=1e-4).minimize(loss)

	# training
	init = tf.global_variables_initializer()	
	saver = tf.train.Saver()
	config = tf.ConfigProto()
	#config.gpu_options.allow_growth = True
	config.log_device_placement=True
	#config.gpu_options.per_process_gpu_memory_fraction = 0.7

	with tf.Session(config=config) as session:
		session.run(init)
		for epoch in range(0, EPOCHS):
			total_steps = int(len(train_data)/BATCH_SIZE)

			total_loss, _ = session.run([loss, optimize], feed_dict={x:train_data[:500],y:train_labels[:500]})
			print("epochs: %d, loss: %f" % (epoch, np.mean(total_loss)))

			# for step in range(200):
			# 	start = step*BATCH_SIZE
			# 	end = start+BATCH_SIZE
			# 	X_batch = train_data[start:end]
			# 	y_batch = train_labels[start:end]
			# 	_, training_loss = session.run([optimize, loss], feed_dict={x: X_batch, y: y_batch})
				
			# 	print ("Steps %d/%d, Training Loss=%f" % (step+1, total_steps, training_loss))

			total_steps = int(len(eval_data)/BATCH_SIZE)
			total_loss = 0
			for step in range(200):
				start = step*BATCH_SIZE
				end = start + BATCH_SIZE
				X_batch = eval_data[start:end]
				yers.conv2d(inputs, 512, (3,3), activation=activation,  padding='same')
	inputs = tf.layers.conv2d(inputs, 512, (3,3), activation=activation,  padding='same')

	return inputs

def model(x,y):

	down_conv1 = tf.layers.conv2d(x, 64, (3,3), activation=tf.nn.relu, padding='same')
	down_conv1 = tf.layers.conv2d(down_conv1, 64, (3,3), activation=tf.nn.relu, padding='same')
	
	down_conv2 = down_block(down_conv1, 128)
	down_conv3 = down_block(down_conv2, 256)
	down_conv4 = down_block(down_conv3, 512)
	down_conv5 = down_block(down_conv4, 1024)

	up_conv1 = up_block(down_conv5, down_conv4, 512)
	up_conv2 = up_block(up_conv1, down_conv3, 256)
	up_conv3 = up_block(up_conv2, down_conv2, 128)
	up_conv4 = up_block(up_conv3, down_conv1, 64)
	#ipdb.set_trace()

	flatten_layer = tf.contrib.layers.flatten(up_conv4)
	final_layer = tf.layers.dense(inputs=flatten_layer, units=100)
	
	return final_layer


if __name__ == '__main__':

	# dimensions
	x = tf.placeholder(tf.float32, shape=X_SHAPE, name='x')
	y = tf.placeholder(tf.float32, shape=y_SHAPE, name='y')

	train_data, train_labels, eval_data, eval_labels = get_data(DATA)

	final = model(x, y)
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=final, labels=y))
	optimize = tf.train.GradientDescentOptimizer(learning_rate=1e-4).minimize(loss)

	# training
	init = tf.global_variables_initializer()	
	saver = tf.train.Saver()
	config = tf.ConfigProto()
	#config.gpu_options.allow_growth = True
	config.log_device_placement=True
	#config.gpu_options.per_process_gpu_memory_fraction = 0.7

	with tf.Session(config=config) as session:
		session.run(init)
		for epoch in range(0, EPOCHS):y_batch = eval_labels[start:end]
				valid_loss = session.run(loss, feed_dict={x: X_batch, y: y_batch})
				total_loss += valid_loss
				print ("Steps %d/%d, Validation Loss=%f" % (step+1, total_steps, valid_loss))
			valid_loss = total_loss / total_steps
			ipdb.set_trace()
			print ("Epoch %d, Training Loss = %d, Validation Loss = %d" % (epoch, training_loss, valid_loss))

		saver.save(session, './dpn_results-tf')	



	

	
