import numpy as np
import cupy as cp 
from math import floor,ceil
from numpy import matlib
'''In This file we will implement the Li and Osher median filter
This Filter will be applied after each warping step in order  
'''
from time import time as t
#####################################################


def im2colcp(mtx0, mtx20, block_size):
    '''
    This function allow us to rearrange the  blocks of two cupy matrices into columns
    Parameters: 
        -mtx0: an array
        -mtx20: the second array
        -block_size: the size of the blocks 
    
    Returns:
        result and result2 twoarrays containing the blocks of the image in their columns
    '''

    mtx=mtx0.get()
    mtx2=mtx20.get()
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

    return [cp.array(result), cp.array(result2)]
######################################################


def denoise_LOcp(un, vn, median_filter_size, lambda23, niters):
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
        [u2, v2] = im2colcp(u, v, (mfsize, mfsize))
        u2 = cp.vstack((u2[:floor(mfsize*mfsize/2), :],
                        u2[ceil(mfsize*mfsize/2): u2.shape[0], :]))
        v2 = cp.vstack((v2[:floor(mfsize*mfsize/2), :],
                        v2[ceil(mfsize*mfsize/2): v2.shape[0], :]))
        uo = cp.reshape(cp.median(cp.vstack((u2, tmpu)), 0),
                        (un.shape[0], un.shape[1]), 'F')
        vo = cp.reshape(cp.median(cp.vstack((v2, tmpv)), 0),
                        (un.shape[0], un.shape[1]), 'F')

    return [uo, vo]

##########################################################################################################################



def im2col(mtx, mtx2, block_size):
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
    for i in range(sy):
        for j in range(sx):
            result[:, i * sx + j] = mtx[j:j + block_size[0],
                                        i:i + block_size[1]].ravel(order='F')
            result2[:, i * sx + j] = mtx2[j:j + block_size[0],
                                          i:i + block_size[1]].ravel(order='F')
    return [result, result2]
######################################################


def denoise_LO(un, vn, median_filter_size, lambda23, niters):
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




'''u = cp.random.rand(3, 4)
v = cp.random.rand(3, 4)'''

'''un=cp.array([[1 ,2 ,3 ,4 ], [5 ,6 ,7 ,8 ] ,[9 ,10 ,11 ,12 ],[13, 14,15,16]])
un=cp.array([[1 ,2 ,3 ], [4 ,5 ,6]])
un=cp.array([[0.8147  ,   0.1270   ,  0.6324], [0.9058   ,  0.9134   , 0.0975]])'''

u = cp.round(10*cp.random.rand(400, 400))
v = cp.round(10*cp.random.rand(400, 400))
median_filter_size = 3  # 2
lambda23 = 100
niters = 1
t1=t()
u0, v0 = denoise_LOcp(u, v, median_filter_size, lambda23, niters)
t2=t()
#print(u0)
unp=u.get()
vnp=v.get()
t3=t()
u0n, v0n = denoise_LO(unp, vnp, median_filter_size, lambda23, niters)
t4=t()

print('GPU',t2-t1,'CPU',t4-t3)
print('norm u',np.linalg.norm(u0.get()-u0n),'norm v',np.linalg.norm(v0.get()-v0n))