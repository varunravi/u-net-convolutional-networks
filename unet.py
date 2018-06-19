# Varun Ravi
# unet.py


from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation
from keras.models import Model
from keras.layers import MaxPooling2D, UpSampling2D, concatenate, Cropping2D
from keras.optimizers import SGD
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import ipdb
import os
from keras.layers import Input
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
def unet(input_shape):
	
	inputs = Input(shape=input_shape)

	down_conv1 = Conv2D(64, (3,3), padding = 'valid', input_shape=input_shape, activation='relu')(inputs)
	down_conv1 = Conv2D(64, (3,3), padding = 'valid', activation='relu')(down_conv1)

	down_conv2 = MaxPooling2D(pool_size=(2,2), strides=2)(down_conv1)
	down_conv2 = Conv2D(128, (3,3), padding = 'valid', activation='relu')(down_conv2)
	down_conv2 = Conv2D(128, (3,3), padding = 'valid', activation='relu')(down_conv2)
	
	down_conv3 = MaxPooling2D(pool_size=(2,2), strides=2)(down_conv2)
	down_conv3 = Conv2D(256, (3,3), padding = 'valid', activation='relu')(down_conv3)
	down_conv3 = Conv2D(256, (3,3), padding = 'valid', activation='relu')(down_conv3)
	
	down_conv4 = MaxPooling2D(pool_size=(2,2), strides=2)(down_conv3)
	down_conv4 = Conv2D(512, (3,3), padding = 'valid', activation='relu')(down_conv4)
	down_conv4 = Conv2D(512, (3,3), padding = 'valid', activation='relu')(down_conv4)
	
	down_conv5 = MaxPooling2D(pool_size=(2,2), strides=2)(down_conv4)
	down_conv5 = Conv2D(1024, (3,3), padding = 'valid', activation='relu')(down_conv5)
	down_conv5 = Conv2D(1024, (3,3), padding = 'valid', activation='relu')(down_conv5)

	up_conv1 = UpSampling2D(size=(2,2))(down_conv5)
	down_conv4 = crop_img(sub_height(up_conv1, down_conv4), sub_width(up_conv1, down_conv4), down_conv4)
	up_conv1 = concatenate([down_conv4, up_conv1], axis=-1)
	up_conv1 = Conv2D(512, (3,3), padding = 'valid', activation='relu')(up_conv1)
	up_conv1 = Conv2D(512, (3,3), padding = 'valid', activation='relu')(up_conv1)

	up_conv2 = UpSampling2D(size=(2,2))(up_conv1)
	down_conv3 = crop_img(sub_height(up_conv2, down_conv3), sub_width(up_conv2, down_conv3), down_conv3)
	up_conv2 = concatenate([down_conv3, up_conv2], axis=-1)
	up_conv2 = Conv2D(256, (3,3), padding = 'valid', activation='relu')(up_conv2)
	up_conv2 = Conv2D(256, (3,3), padding = 'valid', activation='relu')(up_conv2)

	up_conv3 = UpSampling2D(size=(2,2))(up_conv2)
	down_conv2 = crop_img(sub_height(up_conv3, down_conv2), sub_width(up_conv3, down_conv2), down_conv2)
	up_conv3 = concatenate([down_conv2, up_conv3], axis=-1)
	up_conv3 = Conv2D(128, (3,3), padding = 'valid', activation='relu')(up_conv3)
	up_conv3 = Conv2D(128, (3,3), padding = 'valid', activation='relu')(up_conv3)	

	up_conv4 = UpSampling2D(size=(2,2))(up_conv3)
	down_conv1 = crop_img(sub_height(up_conv4, down_conv1), sub_width(up_conv4, down_conv1), down_conv1)
	up_conv4 = concatenate([down_conv1, up_conv4], axis=-1)
	up_conv4 = Conv2D(64, (3,3), padding = 'valid', activation='relu')(up_conv4)
	up_conv4 = Conv2D(64, (3,3), padding = 'valid', activation='relu')(up_conv4)

	final = Conv2D(2, 1, padding = 'valid', activation='softmax')(up_conv4)

	model = Model(inputs=inputs, outputs=final)
	model.compile(optimizer=SGD(lr = 1e-4), loss='binary_crossentropy', metrics=['accuracy'])

	return model


if __name__ == '__main__':

	X_train = []
	Y_train = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
	list_img = pywalker('./letters')

	model = unet((572, 572, 1))
	model.summary()
	
	for i in list_img:
		X_train.append(mpimg.imread(i))
	
	X_train = np.array(X_train)

	ipdb.set_trace()

	
	
	
