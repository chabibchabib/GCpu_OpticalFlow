'''This python script is used to test which version of the software can be used in your computer 
    if your computer an nvidia GPU and cupy and cucim installed the GPU version of the software will
    be privileged. 
    '''
# Try to import cupy, cupyx.scipy and cucim.skimage functions

try:

    from cupyx.scipy.ndimage import median_filter, gaussian_filter
    from cucim.skimage.transform import resize
    from cucim.skimage.registration import optical_flow_tvl1
    import cucim
    from cupyx.scipy.ndimage.filters import laplace
    from cupyx.scipy.ndimage.filters import convolve as filter2  
    from cupyx.scipy.ndimage import median_filter
    try:
        import cupy as cp
        print("The GPU version of the soft will be used")

    except:

        print("cupyx functions or cucim functions can not be called")

    # If not try  to import numpy, scipy and skimage functions this version will be slow
except:
    from scipy.ndimage import median_filter, gaussian_filter
    from skimage.transform import resize
    from skimage.registration import optical_flow_tvl1
    from scipy.ndimage.filters import laplace
    from scipy.ndimage.filters import convolve as filter2  # , laplace
    try:
        import numpy as cp
        print("The CPU version of the software  will be used.Cannot import GPU modules.\nNB:If you already have  a recent NviDia graphic card, try to install CuPy and cuCim (Rapids)")

    except:
        raise ValueError(
            "Please verify if you already have all requierements\n(Numpy (or cupy) + scipy (or cupyx.scipy)+ skimage(or cucim.skimage) ) ")


# Import matplotlib, opencv and time
try:
    import matplotlib.pyplot as plt
    import cv2
    from time import time
    import sys
# Else dispaly error
except:
    raise ValueError(
        "Check if all requierements are installed(Opencv,Matplotlib.pyplot, time,sys...) ")
