import numpy as np
from math import floor, ceil
from numpy import matlib

'''In This file we will implement the Li and Osher median filter:
Y. Li and Osher "A New Median Formula with Applications to PDE Based

This Filter will be applied after each warping step.

'''


def im2col(mtx, mtx2, block_size):
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
    for i in range(sy):
        for j in range(sx):
            result[:, i * sx + j] = mtx[j:j + block_size[0],
                                        i:i + block_size[1]].ravel(order='F')
            result2[:, i * sx + j] = mtx2[j:j + block_size[0],
                                          i:i + block_size[1]].ravel(order='F')
    return [result, result2]


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
    #tmp = (np.tile(tmp, (un.shape[0]*un.shape[1], 1))/lambda23).T
    # tmp=tmp.T
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
u = np.round(10*np.random.rand(5, 5))
v = np.round(10*np.random.rand(5, 5))
print('u\n')
print(u)
print('v\n')
print(v)

median_filter_size = 3  # 2
lambda23 = 100
niters = 1
u0, v0 = denoise_LO(u, v, median_filter_size, lambda23, niters)
print('u\n',u0)
print('v\n',v0)'''
