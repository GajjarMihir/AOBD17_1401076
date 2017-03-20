import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from numpy.linalg import inv
import numpy as np
import skimage
from skimage import color
from ppca_mv import ppca_mv
from sklearn.metrics import mean_squared_error
from math import sqrt


original_image = mpimg.imread("boat.png")

# x = np.random.choice(256, 200, replace=False)
# y = np.random.choice(256, 200, replace=False)

r = np.random.randint(256,size=(19661,2))
	
r1,c1 = r.shape

for i in range(0,r1):
    original_image[r[i,0],r[i,1]] = np.nan


# plt.imshow(original_image, cmap = plt.cm.Greys_r)
# plt.show()

hidden = np.isnan(original_image)
missing = np.count_nonzero(hidden)

[a,b,c,d,reconstructed_image]=ppca_mv(original_image,20,1)

plt.imshow(reconstructed_image, cmap = plt.cm.Greys_r)
plt.show()
original_image = mpimg.imread("boat.png")

RMSE = sqrt(mean_squared_error(reconstructed_image,original_image))
print (RMSE)
