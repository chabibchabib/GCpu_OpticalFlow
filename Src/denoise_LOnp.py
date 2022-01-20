import numpy as np
from math import floor, ceil
from numpy import matlib
from time import time as t
import numba as nb
'''In This file we will implement the Li and Osher median filter
This Filter will be applied after each warping step.  
'''
#####################################################
@nb.njit
def im2col(mtx, mtx2,  block_size):
    '''
    This function allow us to rearrange the  blocks of two matrices into columns.

    Parameters: 
        mtx : ndarray
            The first image 
        -mtx2 : ndarray
            The second array

        block_size: 
            The size of the blocks 

    Returns:
        result : ndarray
            Contains the blocks of the first image in  columns
        result2 : ndarray
            Contains the blocks of the second image in  columns
    '''

    mtx_shape = mtx.shape
    sx = mtx_shape[0] - block_size[0] + 1
    sy = mtx_shape[1] - block_size[1] + 1
    # If the number of lines # Let A m × n, for the [p q] of the block division, the final matrix of p × q, is the number of columns (m-p + 1) × (n-q + 1).
    result = np.empty((block_size[0] * block_size[1], sx * sy))
    result2 = np.empty((block_size[0] * block_size[1], sx * sy))
    # Moved along the line, so the first holding column (i) does not move down along the row (j)
    for i in nb.prange(sy):
        for j in nb.prange(sx):
            for k in nb.prange(j, j + block_size[0]):
                for l in nb.prange(i, i + block_size[1]):
                    row = k-j
                    col = l-i
                    result[col*block_size[0]+row, i * sx + j] = mtx[k, l]
                    result2[col*block_size[0]+row, i * sx + j] = mtx2[k, l]
    return [result, result2]
######################################################


def denoise_LO(un, vn, median_filter_size, lambda23, niters):
    '''Denoising using the Li & Osher median formula
    Y. Li and Osher "A New Median Formula with Applications to PDE Based

    Parameters:
        un : ndarray
            First array to be filtred 
        vn : ndarray
            Second array to be filtred 
        median_filter_size : int
            The window size used 
        lambda23 : float
            The factor used for Li and Osher formulation
        niters : int 
            Number of iterations  
    '''
    mfsize = median_filter_size
    hfsize = floor(mfsize/2)
    n = (mfsize*mfsize-1)/2

    tmp = (np.arange(-n, n+1, dtype=np.float32))
    tmp = (matlib.repmat(tmp, un.shape[0]*un.shape[1], 1)/lambda23).T
    tmpu = matlib.repmat(np.reshape(
        un, (1, un.shape[0]*un.shape[1]), 'F'), int(2*n+1), 1)+tmp
    tmpv = matlib.repmat(np.reshape(
        vn, (1, un.shape[0]*un.shape[1]), 'F'), int(2*n+1), 1)+tmp
    uo = un
    vo = vn
    for i in range(niters):
        u = np.pad(uo, ((hfsize, hfsize), (hfsize, hfsize)), mode='symmetric')
        v = np.pad(vo, ((hfsize, hfsize), (hfsize, hfsize)), mode='symmetric')
        [u2, v2] = im2col(u, v, (mfsize, mfsize))
        u2 = np.vstack((u2[:floor(mfsize*mfsize/2), :],
                        u2[ceil(mfsize*mfsize/2): u2.shape[0], :]))
        v2 = np.vstack((v2[:floor(mfsize*mfsize/2), :],
                        v2[ceil(mfsize*mfsize/2): v2.shape[0], :]))
        uo = np.reshape(np.median(np.vstack((u2, tmpu)), 0),
                        (un.shape[0], un.shape[1]), 'F')
        vo = np.reshape(np.median(np.vstack((v2, tmpv)), 0),
                        (un.shape[0], un.shape[1]), 'F')

    return [uo, vo]

'''
u = np.round(10*np.random.rand(600, 600))
v = np.round(10*np.random.rand(600, 600))
median_filter_size = 3  # 2
lambda23 = 10.02
niters = 1

# print(u0)
t3 = t()
u0n, v0n = denoise_LO(u, v, median_filter_size, lambda23, niters)
t4 = t()


print('CPU', t4-t3)
#print('norm u',np.linalg.norm(u0.get()-u0n),'norm v',np.linalg.norm(v0.get()-v0n))
'''