import numpy as np
#import cupy as cp
from modules import *

def Px(Ix2, Iy2, lmbda, lmbda2, N, M, x):
    '''This function allow us to compute the product of P^-1*x without storing P^-1
    where P is the given preconditioner and x is a given vector. This function will be used 
    to construct the preconditioned version of the Minres algorithm.

        Parameters:
            -Ix2: The square of the image derivative with respect to x 
            -Iy2: The square of the image derivative with respect to y
            -lmbda: Tikhonov parameter
            -lmbda2: the weight used for the non local term  
            -N: number of the image rows
            -M: number of the image cols 
            -x: a 2*M*N given vector 

        Returns:
            res: a 2*M*N vector res=P^-1*x
                      '''

    # Number of pixels
    npixels = N*M
    '''x1=np.reshape(x[:pix],(N,M),order='F')
    x2=np.reshape(x[pix:2*pix],(N,M),order='F')'''

    # Extract the term related to u and the term related to v
    x1 = cp.reshape(x[:npixels], (N, M), order='F')
    x2 = cp.reshape(x[npixels:2*npixels], (N, M), order='F')

    # Compute the result related to each term
    x1 = ((Ix2+lmbda2+8*lmbda))*x1
    x2 = ((Iy2+lmbda2+8*lmbda))*x2

    '''x1=fct2(x1,fct1(Ix2,lmbda))
    x2=fct2(x2,fct1(Iy2,lmbda))'''

    # Reshaping and storing the results

    res = cp.vstack((cp.reshape(x1, (npixels, 1), order='F'),
                     cp.reshape(x2, (npixels, 1), order='F')))
    return res
