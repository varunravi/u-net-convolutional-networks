

from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation
from keras.models import Model
from keras.layers import MaxPooling2D, UpSampling2D, concatenate
from keras.optimizers import SGD
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import ipdb
import os
from keras.layers import Input
#ipdb.set_trace()

def pywalker(path):
	
	list_files = []

	for root, dirs, files in os.walk(path):
 		for file_ in files:
 			list_files.append(os.path.join(root, file_))

	return list_files

def unet(input_shape):
	
	inputs = Input(shape=input_shape)

	down_conv1 = Conv2D(64, (3,3), padding = 'same', input_shape=input_shape, activation='relu')(inputs)
	down_conv1 = Conv2D(64, (3,3), padding = 'same', activation='relu')(down_conv1)
	
	down_conv2 = MaxPooling2D(pool_size=(2,2), strides=2)(down_conv1)
	down_conv2 = Conv2D(128, (3,3), padding = 'same', activation='relu')(down_conv2)
	down_conv2 = Conv2D(128, (3,3), padding = 'same', activation='relu')(down_conv2)
	
	down_conv3 = MaxPooling2D(pool_size=(2,2), strides=2)(down_conv2)
	down_conv3 = Conv2D(256, (3,3), padding = 'same', activation='relu')(down_conv3)
	down_conv3 = Conv2D(256, (3,3), padding = 'same', activation='relu')(down_conv3)
	
	down_conv4 = MaxPooling2D(pool_size=(2,2), strides=2)(down_conv3)
	down_conv4 = Conv2D(512, (3,3), padding = 'same', activation='relu')(down_conv4)
	down_conv4 = Conv2D(512, (3,3), padding = 'same', activation='relu')(down_conv4)
	
	down_conv5 = MaxPooling2D(pool_size=(2,2), strides=2)(down_conv4)
	down_conv5 = Conv2D(1024, (3,3), padding = 'same', activation='relu')(down_conv5)
	down_conv5 = Conv2D(1024, (3,3), padding = 'same', activation='relu')(down_conv5)
	
	up_conv1 = UpSampling2D(size=(2,2))(down_conv5)
	up_conv1 = concatenate([down_conv4, up_conv1], axis=-1)
	up_conv1 = Conv2D(512, (3,3), padding = 'same', activation='relu')(up_conv1)
	up_conv1 = Conv2D(512, (3,3), padding = 'same', activation='relu')(up_conv1)

	up_conv2 = UpSampling2D(size=(2,2))(up_conv1)
	up_conv2 = concatenate([down_conv3, up_conv2], axis=-1)
	up_conv2 = Conv2D(256, (3,3), padding = 'same', activation='relu')(up_conv2)
	up_conv2 = Conv2D(256, (3,3), padding = 'same', activation='relu')(up_conv2)

	up_conv3 = UpSampling2D(size=(2,2))(up_conv2)
	up_conv3 = concatenate([down_conv2, up_conv3], axis=-1)
	up_conv3 = Conv2D(128, (3,3), padding = 'same', activation='relu')(up_conv3)
	up_conv3 = Conv2D(128, (3,3), padding = 'same', activation='relu')(up_conv3)	

	up_conv4 = UpSampling2D(size=(2,2))(up_conv3)
	up_conv4 = concatenate([down_conv1, up_conv4], axis=-1)
	up_conv4 = Conv2D(64, (3,3), padding = 'same', activation='relu')(up_conv4)
	up_conv4 = Conv2D(64, (3,3), padding = 'same', activation='relu')(up_conv4)

	final = Conv2D(1, 1, padding = 'same', activation='softmax')(up_conv4)

	model = Model(inputs=inputs, outputs=final)
	model.compile(optimizer=SGD(lr = 1e-4), loss='binary_crossentropy', metrics=['accuracy'])

	return model

 
if __name__ == '__main__':

	X_train = []
	Y_train = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
	list_img = pywalker('./letters')

	model = unet((64, 64, 1))
	model.summary()
	
	for i in list_img:
		X_train.append(mpimg.imread(i))
	ipdb.set_trace()

	X_train = np.array(X_train)

	f = 4
	

	
	
	
