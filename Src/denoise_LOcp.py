import numpy as np
import cupy as cp
from math import floor, ceil
from numpy import matlib
from time import time as t
import numba as nb
'''In This file we will implement the Li and Osher median filter
This Filter will be applied after each warping step in order  
'''
#####################################################
@nb.njit
def im2col(mtx, mtx2,  block_size):
    '''
    This function allow us to rearrange the  blocks of two matrices into columns
    Parameters: 
        -mtx: an array
        -mtx2: the second array
        -block_size: the size of the blocks 

    Returns:
        result and result2 twoarrays containing the blocks of the image in their columns
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
    mfsize = median_filter_size
    hfsize = floor(mfsize/2)
    n = (mfsize*mfsize-1)/2

    tmp = (cp.arange(-n, n+1, dtype=np.float32))
    tmp = (cp.tile(tmp, (un.shape[0]*un.shape[1], 1))/lambda23).T
    # tmp=tmp.T
    tmpu = cp.tile(cp.reshape(
        un, (1, un.shape[0]*un.shape[1]), 'F'), (int(2*n+1), 1))+tmp
    tmpv = cp.tile(cp.reshape(
        vn, (1, un.shape[0]*un.shape[1]), 'F'), (int(2*n+1), 1))+tmp
    uo = un
    vo = vn
    for i in range(niters):
        u = cp.pad(uo, ((hfsize, hfsize), (hfsize, hfsize)), mode='symmetric')
        v = cp.pad(vo, ((hfsize, hfsize), (hfsize, hfsize)), mode='symmetric')
        [u2, v2] = im2col(u.get(), v.get(), (mfsize, mfsize))
        u2 = cp.array(u2)
        v2 = cp.array(v2)
        u2 = cp.vstack((u2[:floor(mfsize*mfsize/2), :],
                        u2[ceil(mfsize*mfsize/2): u2.shape[0], :]))
        v2 = cp.vstack((v2[:floor(mfsize*mfsize/2), :],
                        v2[ceil(mfsize*mfsize/2): v2.shape[0], :]))
        uo = cp.reshape(cp.median(cp.vstack((u2, tmpu)), 0),
                        (un.shape[0], un.shape[1]), 'F')
        vo = cp.reshape(cp.median(cp.vstack((v2, tmpv)), 0),
                        (un.shape[0], un.shape[1]), 'F')

    return [uo, vo]

'''
u = cp.round(10*cp.random.rand(600, 600))
v = cp.round(10*cp.random.rand(600, 600))
median_filter_size = 3  # 2
lambda23 = 10.02
niters = 1
t1 = t()
u0, v0 = denoise_LO(u, v, median_filter_size, lambda23, niters)
t2 = t()


print('GPU', t2-t1)
#print('norm u',np.linalg.norm(u0.get()-u0n),'norm v',np.linalg.norm(v0.get()-v0n))
'''