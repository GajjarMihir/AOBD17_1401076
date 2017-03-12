# -*- coding: utf-8 -*-
# Simulate the mean of two normal distributions
import math
import copy
import numpy as np
import matplotlib.pyplot as plt

is_debug = False


# Reference：MachineLearning TomM.Mitchell P.137
# Assign the paras of Gaussian distribution, here assign k equals 2
# attention: these two Gaussian distribution share the same Sigma
# and their average Gaussian distribution value are Mu1, Mu2
def init_data(Sigma1,Sigma2, Mu1, Mu2, k, N):
    global X
    global Mu
    global Expectations
    X = np.zeros((1, N))
    Mu = np.random.random(2)
    Expectations = np.zeros((N, k))
    for i in range(0, N):
        if np.random.random(1) > 0.5:
            X[0, i] = np.random.normal()*Sigma1 + Mu1
        else:
            X[0, i] = np.random.normal()*Sigma2 + Mu2
    if is_debug:
        print("*****************")
        print("Initial the observation data X")
        print(X)


# EM Algorithm Step 1  -- E step, Calculate E[zij]
def e_step(Sigma1,Sigma2, k, N):
    global Expectations
    global Mu
    global X
    for i in range(0, N):
        Demon = 0
        for j in range(0, k):
            Demon += math.exp((-1 / (2*(float(Sigma2**2)))) * (float(X[0, i] - Mu[j])) ** 2)
        for j in range(0, k):
            Numer = math.exp((-1 / (2*(float(Sigma1**2)))) * (float(X[0, i] - Mu[j])) ** 2)
            Expectations[i, j] = Numer / Demon
    if is_debug:
        print("*********************")
        print("Missing values E(Z):")
        print(Expectations)


# EM Algorithm Step 2  -- M step, get Mu whick maximize the E[Zij]
def m_step(k, N):
    global Expectations
    global X
    for j in range(0, k):
        Numer = 0
        Demon = 0
        for i in range(0, N):
            Numer += Expectations[i, j] * X[0, i] 
            Demon += Expectations[i, j]
        Mu[j] = Numer / Demon


# EM Algorithm iterate iter_num times, or it terminates when it meet the accuracy Epsilon
# The algorithm iterations iter_num times, or to the precision Epsilon stops the iteration
def EM(Sigma1, Sigma2, Mu1, Mu2, k, N, iter_num, Epsilon):
    init_data(Sigma1,Sigma2, Mu1, Mu2, k, N)
    print("Initial the observation data X", Mu)
    for i in range(iter_num):
        old_Mu = copy.deepcopy(Mu)
        e_step(Sigma1,Sigma2, k, N)
        m_step(k, N)
        print(i, Mu)
        if sum(abs(Mu - old_Mu)) < Epsilon:
            break
    return X

"""
def main():
    # EM(Sigma1, Sigma2, Mu1, Mu2, k, N, iter_num, Epsilon)
    # Here Sigme = 6, Mu1 = 40, Mu2 = 20, k = 2, N = 1000, iter_num = 1000, Epsilon = 0.0001
    
    X = EM(6, 8, 40, 20, 2, 1000, 1000, 0.0001)
    plt.hist(X[0, :], 50)
    plt.show()

if __name__ == '__main__':
    import sys
    sys.exit(int(main() or 0))
"""