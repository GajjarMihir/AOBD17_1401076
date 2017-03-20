import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.metrics import mean_squared_error
from numpy.linalg import *
import numpy as np
from r_pca import R_pca
import skimage
from skimage import color
from numpy import *
import time

gray = mpimg.imread("boat.png")
r = np.random.randint(256,size=(19661,2))
r1,c1 = r.shape

for i in range(0,r1):
    gray[r[i,0],r[i,1]] = np.nan
Lambda = 0.0625 # close to the default one, but works better
tic = time.time()

plt.imshow(gray, cmap = plt.cm.Greys_r)
plt.show()

rpca = R_pca(gray)
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
