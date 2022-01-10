#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
import sys
import warnings
# Source files path 
sys.path.append('../Src/')
from compute_flow import *

''' In this script we test the function compute_flow and we set parameters 
We created a dico called parameters. 
parameters contains: 
    -pyram_levels:  Number of levels
    -factor: Downsampling factor
    -ordre_inter: Order of interpolation used for resizing
    -size_median_filter: Median filter size
    -max_linear_iter: Maximum number of iterations used for linearization in our case 1 is enough
    -max_iter: Warping steps number
    -lmbda: Tikhonov Parameter
    -lambda2: Li and Osher median parameter for non local term (Encourages the displacements and the auxiliary fields to be the same)
    -lambda3: Li and Osher median parameter (Smooth the auxiliary fields )
By default: 

# Pyram params
pyram_levels = 3
factor = 1/0.5
ordre_inter = 3
size_median_filter = 5


# Algo params
max_linear_iter = 1
max_iter = 10
lmbda = 3*10**4
lambda2 = 0.001
lambda3 = 1
'''




parameters = {"pyram_levels": 3, "factor": 1/0.5, "ordre_inter": 3, "size_median_filter": 5, "max_linear_iter": 1, "max_iter": 10,
              "lmbda": 3.*10**4, "lambda2": 0.001, "lambda3": 1., "Mask": None}


# pyram_levels=ri.compute_auto_pyramd_levels(Im1,spacing) #Computing the number of levels dinamically, in  the finest level we get images of 20 to 30 pixels
if __name__ == "__main__":
    if (len(sys.argv)<4):
        raise ValueError("The image sequence was not found\n Verify that you are using  correct directory path and  images name")
    # Images Loading from sys.argv[1] dir

    im1_path=sys.argv[1]+sys.argv[2]
    im2_path=sys.argv[1]+sys.argv[3]
    Im1= cv2.imread(im1_path, 0)
    Im2= cv2.imread(im2_path, 0)

    t1 = time.time()
    u, v = compute_flow(Im1, Im2,  parameters["pyram_levels"], parameters["factor"], parameters["ordre_inter"],
                        parameters["lmbda"], parameters["size_median_filter"], parameters["max_linear_iter"], parameters["max_iter"], parameters["lambda2"], parameters["lambda3"], parameters["Mask"])
    t2 = time.time()

    # Display time
    print('Elapsed time:', (t2-t1), '(s)  --> ', (t2-t1)/60, '(min)')


    # Saving displacements
    if (('cucim'in sys.modules)):
        # Uniaxial strain GPU version case		
        Exy, Exx = np.gradient(u.get())
        cp.save('u_cucim.npy', u.get())
        cp.save('v_cucim.npy', v.get())
    if (('cucim'not in sys.modules)):
        # Uniaxial strain CPU version 
        Exy, Exx = np.gradient(u)
        cp.save('u_cucim.npy', u)
        cp.save('v_cucim.npy', v)

    # Compute energies
    '''print("Energie Image: %E"%(en.energie_image(Im1,Im2,u,v)))
    print("Energie Grad déplacement: %E"%(en.energie_grad_dep(u,v,lmbda)))  '''
    # Plot
    plt.figure()
    plt.imshow(Exx)
    plt.clim(-0.1, 0.1)
    plt.colorbar()
    '''plt.title("mf.size=%i lmbda=%2ef,Lambda2=%2ef Lambda3=2ef" %
            size_median_filter % lmbda % lambda2 % lambda3)'''
    plt.show(block=False)
    # Saving Strain plt
    plt.savefig('StrainImg')
