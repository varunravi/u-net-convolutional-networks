import tensorflow as tf
from keras.datasets import cifar10
import numpy as np

EPOCH = 5
BATCH_SIZE = 64
TEST_SIZE = 128
is_training = True

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
  if is_training:
  	output_shape=[BATCH_SIZE] + [int(layer_shape[1].value*2),int(layer_shape[2].value*2),int(layer_shape[3].value/2)]
  else:
  	output_shape=[TEST_SIZE] + [int(layer_shape[1].value*2),int(layer_shape[2].value*2),int(layer_shape[3].value/2)]

  layer = tf.nn.conv2d_transpose(layer, w, output_shape, [1, 2, 2, 1], padding='SAME')

  layer = tf.add(b, layer)

  return layer


def unet(x):
	with tf.variable_scope("conv_1"):
		layer = conv2d(x, 'w0', [3, 3, 3, 64], 'b0', [64])
		layer = tf.nn.relu(layer)
		layer = conv2d(layer, 'w1', [3, 3, 64, 64], 'b1', [64])
		layer_1 = tf.nn.relu(layer, name='conv_1')

		layer = tf.nn.max_pool(value=layer_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='maxpool_1')

	with tf.variable_scope("conv_2"):
		layer = conv2d(layer, 'w2', [3, 3, 64, 128], 'b2', [128])
		layer = tf.nn.relu(layer)
		layer = conv2d(layer, 'w3', [3, 3, 128, 128], 'b3', [128])
		layer_2 = tf.nn.relu(layer, name='conv_2')

		layer = tf.nn.max_pool(value=layer_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='maxpool_2')

	with tf.variable_scope("conv_3"):
		layer = conv2d(layer, 'w4', [3, 3, 128, 256], 'b4', [256])
		layer = tf.nn.relu(layer)
		layer = conv2d(layer, 'w5', [3, 3, 256, 256], 'b5', [256])
		layer_3 = tf.nn.relu(layer, name='conv_3')

		layer = tf.nn.max_pool(value=layer_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='maxpool_2')

	with tf.variable_scope("conv_4"):
		layer = conv2d(layer, 'w6', [3, 3, 256, 512], 'b6', [512])
		layer = tf.nn.relu(layer)
		layer = conv2d(layer, 'w7', [3, 3, 512, 512], 'b7', [512])
		layer_4 = tf.nn.relu(layer, name='conv_4')

		layer = tf.nn.max_pool(value=layer_4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='maxpool_3')

	with tf.variable_scope("conv_5"):
		layer = conv2d(layer, 'w8', [3, 3, 512, 1024], 'b8', [1024])
		layer = tf.nn.relu(layer)
		layer = conv2d(layer, 'w9', [3, 3, 1024, 1024], 'b9', [1024])
		layer = tf.nn.relu(layer)

	with tf.variable_scope("upconv_1"):
		layer = up_conv2d(layer, 'w10', [2, 2, 512, 1024], 'b10', [512])
		layer = tf.concat([layer_4, layer], axis=3)

		layer = conv2d(layer, 'w11', [3, 3, 1024, 512], 'b11', [512])
		layer = tf.nn.relu(layer)
		layer = conv2d(layer, 'w12', [3, 3, 512, 512], 'b12', [512])
		layer = tf.nn.relu(layer)

	with tf.variable_scope("upconv_2"):
		layer = up_conv2d(layer, 'w13', [2, 2, 256, 512], 'b13', [256])
		layer = tf.concat([layer_3, layer], axis=3)

		layer = conv2d(layer, 'w14', [3, 3, 512, 256], 'b14', [256])
		layer = tf.nn.relu(layer)
		layer = conv2d(layer, 'w15', [3, 3, 256, 256], 'b15', [256])
		layer = tf.nn.relu(layer)

	with tf.variable_scope("upconv_3"):
		layer = up_conv2d(layer, 'w16', [2, 2, 128, 256], 'b16', [128])
		layer = tf.concat([layer_2, layer], axis=3)

		layer = conv2d(layer, 'w17', [3, 3, 256, 128], 'b17', [128])
		layer = tf.nn.relu(layer)
		layer = conv2d(layer, 'w18', [3, 3, 128, 128], 'b18', [128])
		layer = tf.nn.relu(layer)

	with tf.variable_scope("upconv_4"):
		layer = up_conv2d(layer, 'w19', [2, 2, 64, 128], 'b19', [64])
		layer = tf.concat([layer_1, layer], axis=3)

		layer = conv2d(layer, 'w20', [3, 3, 128, 64], 'b20', [64])
		layer = tf.nn.relu(layer)
		layer = conv2d(layer, 'w21', [3, 3, 64, 64], 'b21', [64])
		layer = tf.nn.relu(layer)

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
		train_writer = tf.summary.FileWriter('./tmp', graph=sess.graph) 
		sess.run(tf.global_variables_initializer())
		#saver.restore(sess, tf.train.latest_checkpoint('~/saved_models/unet/cifar-10/'))

		for epoch in range(EPOCH):
			total_steps = int(x_train.shape[0]/BATCH_SIZE)
			for step in range(total_steps):
				x_batch = x_train[step*BATCH_SIZE:step*BATCH_SIZE + BATCH_SIZE].reshape(BATCH_SIZE, 32, 32, 3)
				y_batch = y_train[step*BATCH_SIZE:step*BATCH_SIZE + BATCH_SIZE].reshape(BATCH_SIZE, 1)

				current_global_step, _, current_loss= sess.run([global_step, optimizer, loss], feed_dict={x:x_batch, y:y_batch})

				print("epoch: %d global_step: %d, step: %d loss: %f" % (epoch+1, current_global_step, step, current_loss))

				if step % 10 == 0:
					is_training = False
					xtest_batch = x_test[:TEST_SIZE].reshape(TEST_SIZE, 32, 32, 3)
					ytest_batch = y_test[:TEST_SIZE].reshape(TEST_SIZE, 1)
					softmax_p, accuracy = sess.run([softmax, prediction], feed_dict={x: xtest_batch, y: ytest_batch})
					is_training = True
					print("accuracy: %f%%" % (accuracy*100))
	    
		# 	saver.save(sess, save_path='~/saved_models/unet/cifar-10/model.chkpt', global_step=current_global_step)
		# tf.train.write_graph(sess.graph, '~/saved_models/unet/cifar-10/', 'final_graph.pb', as_text=False)
		# tf.train.export_meta_graph('~/saved_models/unet/cifar-10/final_graph.meta', graph=graph, clear_devices=True)

