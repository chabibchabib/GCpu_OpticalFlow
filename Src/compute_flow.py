import numpy as np
import cv2
import matplotlib.pyplot as plt
from modules import *
import flow_operator as fo
import rescale_img as ri
import energies as en
import time



def compute_image_pyram(Im1, Im2, ratio, N_levels,ordre_inter):
    ''' This function creates  the images used at each level (Pyramid levels)

        Parameters:

            -Im1: Reference image
            -Im2: Second images 
            -ratio: Downsampling factor 
            -N_levels: Number of levels
            -ordre_inter: the desired order of interpolation used for the skimage.resize function must be between 0 and 5
        
        Returns:
            2 lists P1 and P2 containing the images pyramids  

         '''

    # Lists containing the Images
    # For the first image
    P1 = []
    # The second image
    P2 = []

    tmp1 = Im1
    tmp2 = Im2
    P1.append(tmp1)
    P2.append(tmp2)

    print("Level:", '0', "shape Images :", tmp1.shape)

    for lev in range(1, N_levels):

        sz = np.round(np.array(tmp1.shape, dtype=np.float32)*ratio)

        tmp1 = resize(cp.array(tmp1),
                      (sz[0], sz[1]), anti_aliasing=False, mode='symmetric',order=ordre_inter)
        tmp2 = resize(cp.array(tmp2),
                      (sz[0], sz[1]), anti_aliasing=False, mode='symmetric',order=ordre_inter)

        print("Level:", lev, "shape Images :", tmp1.shape)
        P1.append(tmp1)
        P2.append(tmp2)
    return [P1, P2]


def resample_flow_unequal(u, v, sz, ordre_inter):
    '''
    This function reshape the flow fields u and v
        Parameters:

            -u: horizontal flow field
            -v: vertical flow field
            -sz: The new shape 
            -ordre_inter: the desired order of the interpolation used for skimage.resize function must be between 0 and 5

        Returns: 
            new reshaped flow fields u and v 

    '''
    # Old size
    osz = u.shape
    # Computing factors
    ratioU = sz[0]/osz[0]
    ratioV = sz[1]/osz[1]
    # Resize u and v
    u = resize(u, sz, order=ordre_inter)*ratioU
    v = resize(v, sz, order=ordre_inter)*ratioV
    return u, v


def compute_flow(Im1, Im2,  py_lev, factor, ordre_inter, lmbda, size_median_filter,  max_linear_iter, max_iter, lambda2, lambda3, Mask=None):
    '''Compute the flow fields

        Parameters:
            -Im1: The first image 
            -Im2: The second one
            -u: Initial guest (horizontal flow field)
            -v: Initial guest (vertical flow field)
            -py_lev: Number of the pyramid levels
            -ordre_inter: The desired order of interpolation must be between 0 and 5
            hat parameter will be used for skimage.resize function. 
            For more details: https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.resize
            -lmbda: Tikhonov Parameter
            -size_median_filter: size of the median filter 
            -max_linear_iter: number of the linearization steps desired. In our case for the quadratic norm 1 is good 
            -max_iter: number of warping steps at each level 
            -lambda2: weight to encourage (u,v) and (uhat,vhat) the auxialiary fields to be the same 
            -lambda3: wight to smooth the a the auxialiary fields 
            -Mask: Mask must contain only 0 and 1, 0 are the element that will be droped during the calculations.
            Mask must have the same size as the image sequence

        Returns:
        The computed flow fields 

    '''

    # Spatial derivative filter to use
    h = cp.array([[-1, 8, 0, -8, 1]])
    h = h/12

    '''
    # If we want to use the structure-texture decoposition for the images with lightining changes
    Im1,Imm1=ri.decompo_texture(Im1, param1, param2, param3, param4)
    Im2,Imm1=ri.decompo_texture(Im2, param1, param2, param3, param4)'''

    #Mask = cp.ones_like(Mask)*1.0

    #Check the shape of the image sequence 
    if(Im1.shape!=Im2.shape):
        raise ValueError("Images must be the same shape")
    
    v, u = optical_flow_tvl1(cp.array(Im1), cp.array(Im2),attachment=20)
    Im1=np.array(Im1,dtype=np.float32)
    Im2=np.array(Im2,dtype=np.float32)

    #Check order interpolation 
    if(ordre_inter <0 or ordre_inter >5 ):
        raise ValueError("The interpolation order must be an int between 0 and 5\nFor more details: https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.resize ")
    # The pyramid images

    P1, P2 = compute_image_pyram(
        Im1, Im2, 1/factor, py_lev,ordre_inter)
    # P1_gnc,P2_gnc=compute_image_pyram(Im1,Im2,1/gnc_factor,gnc_pyram_levels,math.sqrt(gnc_spacing)/math.sqrt(2))

    uhat = u
    vhat = v

    # Print Pyramid level
    print('\t Pyram levels', py_lev)

    for lev in range(py_lev-1, -1, -1):
        # Adapt lambda to each pyramid level
        lmbda = lmbda/(lev+1)

        '''sz2 = np.round(np.array(Im1.shape, dtype=np.float32)
                       * (1/factor**(lev)))
        sz2 = np.array(sz2, dtype=np.int32)'''

        # Printing the level number and extract the image sequence which will be used at this level

        print("\t \t Level Number", lev)
        Image1 = cp.array(P1[lev])
        Image2 = cp.array(P2[lev])

        # New shape of the flow field
        sz = Image1.shape

        # Reshape the displacements and the auxiliary fields
        u, v = resample_flow_unequal(u, v, sz, ordre_inter)
        uhat, vhat = resample_flow_unequal(uhat, vhat, sz, ordre_inter)

        if (type(Mask) != type(None)):
            # Check if image sequence and mask are
            if(Mask.shape != P1[0].shape):
                raise ValueError("Mask and Images must be the same size")
            # Reshape Msk if exist
            Msk = resize(Mask, sz, anti_aliasing=False,
                         mode='symmetric', order=ordre_inter)

        # Construct the a grid
        N = sz[0]
        M = sz[1]
        yy = np.linspace(0, N-1, N)
        xx = np.linspace(0, M-1, M)
        xx, yy = np.meshgrid(xx, yy)

        #print("x shape:",xx.shape)
        # code.interact(local=locals())

        # Compute derivatives
        Ix = filter2(Image1, h)
        Iy = filter2(Image1, h.T)
        # Cast  Image
        Image1 = cp.array(Image1)
        # Dropping the chosen pixels in Mask
        if (type(Mask) != type(None)):
            Ix = Ix*Msk
            Iy = Iy*Msk
            # Cast Image and dropping the chosen pixels in Mask
            Image1 = Image1*Msk
        else:
            Msk=None
        Image2 = cp.array(Image2)
        # Compute Ix^2, Iy^2 and Ix.*Iy
        Ix2 = Ix*Ix
        Iy2 = Iy*Iy
        Ixy = Ix*Iy

        remplacement = True
        # Compute the new flow fields and the auxiliary one
        u, v, uhat, vhat = fo.compute_flow_base(Image1, Image2, max_iter, max_linear_iter, u, v, lmbda, size_median_filter,
                                                uhat, vhat, lambda2, lambda3, remplacement, Ix, Iy, Ix2, Iy2, Ixy, xx, yy, sz)

    u = uhat
    v = vhat
    return u, v





