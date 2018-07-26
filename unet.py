import tensorflow as tf
from keras.datasets import cifar10
import numpy as np

EPOCH = 100
BATCH_SIZE = 512

def conv2d(layer, w_name, w_shape, b_name, b_shape):
 
  # with tf.Graph().as_default() as graph:
  w = tf.get_variable(name=w_name, shape=w_shape, dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
  b = tf.get_variable(name=b_name, shape=b_shape, dtype=tf.float32, initializer=tf.zeros_initializer())
  layer = tf.nn.conv2d(layer, w, [1, 1, 1, 1], padding='SAME')
  layer = tf.add(b, layer)

  return layer


def up_conv2d(layer, w_name, w_shape, b_name, b_shape):
 
  w = tf.get_variable(name=w_name, shape=w_shape, dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
  b = tf.get_variable(name=b_name, shape=b_shape, dtype=tf.float32, initializer=tf.zeros_initializer())

  layer_shape = layer.shape

  output_shape=[BATCH_SIZE] + [int(layer_shape[1].value*2),int(layer_shape[2].value*2),int(layer_shape[3].value/2)]
  
  layer = tf.nn.conv2d_transpose(layer, w, output_shape, [1, 2, 2, 1], padding='SAME')
  layer = tf.add(b, layer)

  return layer

def unet_down(name, layer, input_filter, output_filter):
  
	with tf.variable_scope(name):
		layer = conv2d(layer, 'w_0', [3, 3, input_filter, output_filter], 'b_0', [output_filter])
		layer = tf.nn.relu(layer)
		layer = conv2d(layer, 'w_1', [3, 3, output_filter, output_filter], 'b_1', [output_filter])
		layer = tf.nn.relu(layer, name='conv2d_0')

	return layer

def unet_up(name, layer, layer_down, input_filter, output_filter):
	
	with tf.variable_scope(name):
		layer = up_conv2d(layer, 'w_0', [2, 2, output_filter, input_filter], 'b_0', [output_filter])
		layer = tf.concat([layer_down, layer], axis=3)

		layer = conv2d(layer, 'w_1', [3, 3, input_filter, output_filter], 'b_1', [output_filter])
		layer = tf.nn.relu(layer)
		layer = conv2d(layer, 'w_2', [3, 3, output_filter, output_filter], 'b_2', [output_filter])
		layer = tf.nn.relu(layer)

	return layer


def unet(layer, num_layers=3, input_shape=[BATCH_SIZE, 32, 32, 3]):

	list_layers=[]
	num_layers=num_layers-1

	layer = unet_down('conv_1', layer, input_shape[3], np.power(2, 6))
	list_layers.append(layer)
	layer = tf.nn.max_pool(value=layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='maxpool_0')

	for i in range(1, num_layers):
		layer = unet_down('conv_1'+str(i), layer, np.power(2, (i+5)), np.power(2, (i+6)))
		list_layers.append(layer)
		layer = tf.nn.max_pool(value=layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='maxpool_0')

	layer = unet_down('conv_bottom', layer, np.power(2, num_layers+5), np.power(2, num_layers+6))
	list_layers.append(layer)


	for i in range(1, num_layers+1):
		layer = unet_up('upconv_1'+str(i), layer, list_layers[num_layers-i], np.power(2, num_layers+7-i), np.power(2, num_layers+6-i))

	print(layer)


	with tf.variable_scope("final"):
		layer = conv2d(layer, 'w22', [1, 1, 64, 1], 'b22', [1])

		layer = tf.contrib.layers.flatten(inputs=layer)
		w_flat = tf.get_variable(name='w_flat', shape=[layer.shape[1], 10], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
		b_flat = tf.get_variable(name='b_flat', shape=[10], dtype=tf.float32, initializer=tf.zeros_initializer())
		layer = tf.matmul(layer, w_flat)
		final = tf.add( name='cifar10_output', x=b_flat, y=layer)

	return final

if __name__ == '__main__':

	(x_train, y_train), (x_test, y_test) = cifar10.load_data()
	
	tf.reset_default_graph()

	x = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32, 3], name='input')
	y = tf.placeholder(dtype=tf.int32, shape=[None, 1], name = 'y')

	final = unet(x)

	# loss
	global_step = tf.train.get_or_create_global_step() 
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(tf.cast(y, dtype=tf.int32), 10), logits=final))
	optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss, global_step=global_step)

	# prediction
	softmax = tf.argmax(tf.nn.softmax(logits=final), axis=1, output_type=tf.int32)
	re = tf.reshape(softmax, [BATCH_SIZE, 1])
	prediction = tf.reduce_mean(tf.cast(tf.equal(re, y), dtype=tf.float32))

	saver = tf.train.Saver()

	with tf.Session() as sess:
		train_writer = tf.summary.FileWriter('./tmp1', graph=sess.graph) 
		sess.run(tf.global_variables_initializer())
		saver.restore(sess, tf.train.latest_checkpoint('./tmp/'))

		for epoch in range(EPOCH):
			total_steps = int(x_train.shape[0]/BATCH_SIZE)
			for step in range(total_steps):
				x_batch = x_train[step*BATCH_SIZE:step*BATCH_SIZE + BATCH_SIZE].reshape(BATCH_SIZE, 32, 32, 3)
				y_batch = y_train[step*BATCH_SIZE:step*BATCH_SIZE + BATCH_SIZE].reshape(BATCH_SIZE, 1)

				current_global_step, _, current_loss= sess.run([global_step, optimizer, loss], feed_dict={x:x_batch, y:y_batch})

				print("epoch: %d global_step: %d, step: %d loss: %f" % (epoch+1, current_global_step, step, current_loss))

				if step % 10 == 0:
					xtest_batch = x_test[:BATCH_SIZE].reshape(BATCH_SIZE, 32, 32, 3)
					ytest_batch = y_test[:BATCH_SIZE].reshape(BATCH_SIZE, 1)
					softmax_p, accuracy = sess.run([softmax, prediction], feed_dict={x: xtest_batch, y: ytest_batch})

					print("accuracy: %f%%" % (accuracy*100))
	    
			saver.save(sess, save_path='./tmp/model.chkpt', global_step=current_global_step)
			
