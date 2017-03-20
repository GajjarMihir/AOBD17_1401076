import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from numpy.linalg import *
import numpy as np
from r_pca import R_pca
import skimage
from skimage import color
from sklearn.metrics import mean_squared_error
from numpy import *
import time

RGB = mpimg.imread("boat.png")
#print(RGB)
#plt.imshow(RGB)
#plt.show()
#RGB_noise = mpimg.imnoise(RGB,'salt & pepper',0.02)

#print(img)
RGB_noise = skimage.util.random_noise(RGB, mode='s&p', seed=None, clip=True, amount=0.05)
corr_img = color.rgb2gray(RGB_noise)
#print corr_img.shape
# apply Robust PCA
#print corr_img
plt.imshow(corr_img, cmap = plt.cm.Greys_r)
plt.show()
Lambda = 0.0625 # close to the default one, but works better
tic = time.time()
#plt.imshow(X, cmap = plt.cm.Greys_r)
#plt.show()

rpca = R_pca(corr_img)
toc = time.time()
L, S = rpca.fit(max_iter=10000, iter_print=100)


original_image = mpimg.imread("boat.png")
RMSE = sqrt(mean_squared_error(L,original_image))
print (RMSE)
plt.imshow(L, cmap = plt.cm.Greys_r)
plt.show()
# plt.imshow(S, cmap = plt.cm.Greys_r)
# plt.show()
# plt.imshow(L+S, cmap = plt.cm.Greys_r)
# plt.show()

#rpca.plot_fit()
#plt.show()
