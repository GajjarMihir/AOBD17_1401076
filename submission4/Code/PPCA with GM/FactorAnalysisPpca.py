import numpy
import numpy as np
import matplotlib.pyplot as plt
from ppca import PPCA
import matplotlib.image as mpimg
import skimage
from skimage import color
import cv2
from pandas import read_csv
import matplotlib
from numpy import *

seed = 7
np.random.seed(seed)


from tensorflow.examples.tutorials.mnist import input_data
dataset = input_data.read_data_sets("christmas-trees/")

dataset = dataframe.values
X = dataset[1:,1]

E = []

for i in range(len(X)):
	Y = np.fromstring(X[i], dtype=int, sep = ' ')
	Y = np.reshape(Y,(48, 48))
	E.append(Y)

X_inp = np.array(E)

X_train = X_inp.astype('float32')
print X_inp

inp_img = X_train[0,:,:]
ppca = PPCA(inp_img)
ppca.fit(d=20, verbose=False)
component_mat = ppca.transform()
E_y = component_mat.reshape(1,component_mat.shape[0],component_mat.shape[1])

for i in range(1,len(X_train)):
	print i
	inp_img =   X_train[i,:,:]
	ppca = PPCA(inp_img)
	try:
		ppca.fit(d=20, verbose=False)
		component_mat = ppca.transform()
		shape = component_mat.shape
		component_mat = component_mat.reshape(1,component_mat.shape[0],component_mat.shape[1])
		if shape[1] == 20:
			E_y = concatenate((E_y,component_mat))
	except numpy.linalg.linalg.LinAlgError:
		print "Numpy Error"
X_ppca = np.array(E_y)
print X_ppca.shape
np.save('ppca_model_1000.npy', X_ppca)
