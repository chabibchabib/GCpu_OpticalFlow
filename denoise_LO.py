import numpy as np
from scipy.ndimage.filters import convolve as filter2
import math
'''In This file we will implement the Li and Osher median filter 
'''
#####################################################


'''def im2col(mtx, mtx2, block_size):
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
    return [result, result2]'''
######################################################


def denoise_LO(un, vn, median_filter_size, lambda23, niters):
    mfsize = median_filter_size
    hfsize = math.floor(mfsize/2)
    n = (mfsize*mfsize-1)/2

    tmp = (np.arange(-n, n+1, dtype=np.float32))
    tmp = (np.matlib.repmat(tmp, un.shape[0]*un.shape[1], 1)/lambda23).T
    # tmp=tmp.T
    tmpu = np.matlib.repmat(np.reshape(
        un, (1, un.shape[0]*un.shape[1]), 'F'), int(2*n+1), 1)+tmp
    tmpv = np.matlib.repmat(np.reshape(
        vn, (1, un.shape[0]*un.shape[1]), 'F'), int(2*n+1), 1)+tmp
    uo = un
    vo = vn
    for i in range(niters):
        u = np.pad(uo, ((hfsize, hfsize), (hfsize, hfsize)), mode='symmetric')
        v = np.pad(vo, ((hfsize, hfsize), (hfsize, hfsize)), mode='symmetric')
        [u2, v2] = im2col(u, v, (mfsize, mfsize))
        # print(u)
        u2 = np.vstack((u2[:math.floor(mfsize*mfsize/2), :],
                        u2[math.ceil(mfsize*mfsize/2): u2.shape[0], :]))
        v2 = np.vstack((v2[:math.floor(mfsize*mfsize/2), :],
                        v2[math.ceil(mfsize*mfsize/2): v2.shape[0], :]))
        uo = np.reshape(np.median(np.vstack((u2, tmpu)), 0),
                        (un.shape[0], un.shape[1]), 'F')
        vo = np.reshape(np.median(np.vstack((v2, tmpv)), 0),
                        (un.shape[0], un.shape[1]), 'F')

    return [uo, vo]


'''u = np.random.rand(3, 4)
v = np.random.rand(3, 4)

un=np.array([[1 ,2 ,3 ,4 ], [5 ,6 ,7 ,8 ] ,[9 ,10 ,11 ,12 ],[13, 14,15,16]])
un=np.array([[1 ,2 ,3 ], [4 ,5 ,6]])
un=np.array([[0.8147  ,   0.1270   ,  0.6324], [0.9058   ,  0.9134   , 0.0975]])

    
median_filter_size=3 #2
lambda23=100
niters=10
u0=denoise_LO (un, median_filter_size, lambda23, niters)
print(u0)'''
