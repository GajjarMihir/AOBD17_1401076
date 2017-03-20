import numpy as np
from numpy import *
from numpy.linalg import *
from scipy.linalg import *
from numpy.matlib import *
from math import *
import matplotlib.pyplot as plt
from skimage import color
import skimage

def rpca(X, Lambda=None, mu=None, tol=10**(-6), max_iter=1000):
    # - X is a data matrix (of the size N x M) to be decomposed
    #   X can also contain NaN's for unobserved values
    # - Lambda - regularization parameter, default = 1/sqrt(max(N,M))
    # - mu - the augmented lagrangian parameter, default = 10*Lambda
    # - tol - reconstruction error tolerance, default = 1e-6
    # - max_iter - maximum number of iterations, default = 1000
  #  plt.imshow(X, cmap = plt.cm.Greys_r)
   # plt.show()

    M, N = X.shape
    #print "tol",tol
    unobserved = np.isnan(X)
    #X[unobserved] = 0
    normX = norm(X) #fro norm

    # initial solution
    L = zeros((M, N))
    S = zeros((M, N))
    Y = zeros((M, N))

    for iteR in range(1,max_iter):
        # ADMM step: update L and S
        #print "mu",1/mu
        L = Do(1/mu, X - S + (1/mu)*Y)
        S = So(Lambda/mu, X - L + (1/mu)*Y)
        #print "L",L.shape
        #print "S",S.shape
        # and augmented lagrangian multiplier
        Z = X - L - S
        #Z[unobserved] = 0 # skip missing values
        Y = Y + mu*Z

        err = norm(Z) / normX #Fro norm
        if iteR == 1 or iteR % 10 == 0 or err < tol:
            #print err
            #print matrix_rank(L)
            #print "S",S.shape
            #print "unobserved",unobserved.shape
            #print "non_zero",count_nonzero(S[~unobserved])
            #print('iter: %d\terr: %f\trank(L): %d\tcard(S): %d\n') %(iteR, err, matrix_rank(L), count_nonzero(S[~unobserved]))
        if err < tol:
            break
    return L,S

def So(tau, XX):
    # shrinkage operator
    #print "signX",sign(X).shape
    #print "amax",amax(absolute(X) - tau, 0).shape
    max_ = maximum(absolute(XX) - tau, zeros(XX.shape))
    r = multiply(max_,sign(XX))
    #print "So r",r.shape
    return r

def Do(tau, XX):
    # shrinkage operator for singular values
    U, S, VT = svd(XX, full_matrices=False)
    #print "S",S
    S = diag(S)

    r = U.dot(So(tau, S)).dot(VT.transpose())
    #print "Do r",r.shape
    return r
