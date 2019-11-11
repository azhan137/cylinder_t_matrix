import numpy as np


def compute_gauss_quadrature(start, end, divisions):
    n = divisions-1
    n_1 = divisions
    n_2 = divisions+1

    x = np.linspace(-1, 1, n_1)
    y = np.cos((2*(np.linspace(0, n, n_1))+1)*np.pi/(2*n+2))+(0.27/n_1*np.sin(np.pi*x*n/n_2))

    L = np.zeros((n_1,n_2))

    y0 = 2

    while max(np.abs(y-y0)) > np.spacing(1.0):
        L[:, 0] = 1
        Lp = np.zeros((n_1,2))a
        L[:, 1] = y
        Lp[:, 1] = 1

        for k in range(1, n):
            L[:, k+1] = ((2*k)*y*L[:, k]-k*L[:, k-1])/(k+1)

        Lp = n_2*(L[:, n]-y*L[:, n_1])/(1-y**2)

    w = (end-start)/((1-y**2)*Lp**2)*(n_2/n_1)**2
    x = (end-start)*y/2+(start+end)/2

    return w, x

if __name__ == '__main__':
    [x, w] = compute_gauss_quadrature(0, 1, 10)
    print(x)
    print(w)