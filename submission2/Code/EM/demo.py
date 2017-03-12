import matplotlib.pyplot as plt
from EM import EM


def main():
    # EM(Sigma, Mu1, Mu2, k, N, iter_num, Epsilon)
    # Here Sigme = 6, Mu1 = 40, Mu2 = 20, k = 2, N = 1000, iter_num = 1000, Epsilon = 0.0001
    X = EM(14.44, 15.85, 63.42, 88.04, 2, 1000, 1000, 0.0001)
    plt.hist(X[0, :], 50)
    plt.show()

if __name__ == '__main__':
    import sys
    sys.exit(int(main() or 0)) 