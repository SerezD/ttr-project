import numpy as np
import matplotlib.pyplot as plt

def pca(X):

    # Centering data...
    m = np.mean(X, axis = 0)
    Xc = X - m

    # Cov matrix (dim Nfeat x Nfeat)
    C = np.cov(Xc, rowvar = False)  

    # eigen of Cov Matrix
    lambdas, U = np.linalg.eigh(C)

    # Ordering eigen values
    best_eig_idxs = np.argsort(lambdas)[::-1]
    best_eig = lambdas[best_eig_idxs] 
    best_U = U[:,best_eig_idxs] 

    # Best N dimension
    y = np.cumsum(best_eig)/np.sum(best_eig)
    N = np.where(y >= 0.95)[0][0]
    if N < 1:
        N = 2
    else:
        N = N + 1
    
    print ("PCA N = ", N)

    # Matrix T (size N):   
    T = best_U[:,:N]

    # Transforming data...
    XT = np.dot(Xc, T)

    return XT, N
