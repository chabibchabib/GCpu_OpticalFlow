import numpy as np
import math
import code
from utils import *
import scipy.sparse as sparse
#import denoise_LO as lo
import time
import solveur_precond as sop


###########################################################


def warp_image2(Image, XI, YI):
    ''' We add the flow estimated to the image coordinates, remap them towards the ogriginal image

    Parameters:

        -Image : ndarray
            Image to warp

        -XI : ndarray
            x coordinates for warping

        -YI : ndarray
            y coordinates used for warping

    Returns:

        WImage :  ndarray
            The  warped image using the function remap of opencv

    '''

    # Image=np.array(Image,np.float32)
    XI = np.array(XI, np.float32)
    YI = np.array(YI, np.float32)
    if 'cupy' in sys.modules:

        WImage = cv2.remap(Image.get(), XI, YI, interpolation=cv2.INTER_CUBIC)
    else:
        WImage = cv2.remap(Image, XI, YI, interpolation=cv2.INTER_CUBIC)

    return cp.array(WImage)
############################################


def derivatives(Image1, Image2, u, v, xx, yy, sz):
    '''This function computes the temporal derivative of  image sequence.

        Parameters:

            Image1 : ndrray
                The first image of the sequence.

            Image2 : ndrray
                The second image of the sequence.

            u : ndrray
                The horizontal displacement field

            v : ndrray 
                The vertical displacement

            xx : ndrray
                to define

            yy : ndrray 
                to define

            sz: ndrray
                The size of the image 

        Returns:

            It :  ndrray
                The temporal derivative   
            '''

    N = sz[0]
    M = sz[1]

    # code.interact(local=locals())

    if (('cucim'in sys.modules)):

        xx2 = xx+u.get()
        yy2 = yy+v.get()

        WImage = cv2.remap(Image2.get(), np.array(xx2, np.float32), np.array(
            yy2, np.float32), interpolation=cv2.INTER_CUBIC)

        It = WImage-(Image1).get()  # Temporal deriv

    if (('cucim'not in sys.modules)):

        xx2 = xx+u
        yy2 = yy+v

        WImage = cv2.remap(Image2, np.array(xx2, np.float32), np.array(
            yy2, np.float32), interpolation=cv2.INTER_CUBIC)

        It = WImage-Image1  # Temporal deriv

    It = np.nan_to_num(It)  # Remove Nan values on the derivatives

    # Drop the pixel where the motion is out of the bound
    out_bound = np.where((yy2 > N-1) | (yy2 < 0) | (xx2 > M-1) | (xx2 < 0))

    It[out_bound] = 0

    return cp.array(It)
############################################################


def conv_matrix(F, sz):
    '''conv_matrix is used for the construction of Laplacien Matrix

    Parameters:

        F : ndarray
            Spacial filter to compute spatial derivatives  can be [-1,1] or [[-1],[1]]

        sz : ndarray
            Size of the used image
    
    Returns:

        M: ndarray
            Laplacian matrix
    
    '''
    if(F.shape == (1, 2)):
        I = np.hstack(
            (np.arange(sz[0], (sz[0]*(sz[1]))), np.arange(sz[0], (sz[0]*(sz[1])))))
        J = np.hstack(
            (np.arange(sz[0], (sz[0]*(sz[1]))), np.arange(sz[0], (sz[0]*(sz[1])))))
        K = np.zeros(int(2*(sz[0]*(sz[1]-1))))
        K[:int(sz[0]*(sz[1]-1))] = 1
        J[sz[0]*(sz[1]-1):2*(sz[0]*sz[1])] = J[sz[0]
                                               * (sz[1]-1):2*sz[0]*sz[1]]-sz[0]
        K[sz[0]*(sz[1]-1):2*(sz[0]*sz[1])] = -1

    if(F.shape == (2, 1)):
        lI = []
        for i in range(1, sz[0]*sz[1]):
            if(i % sz[0] != 0):
                lI.append(i)
        I = np.array(lI)
        nnzl = I.shape[0]
        I = np.hstack((I, I))

        J = np.hstack((np.array(lI), np.array(lI)-1))
        K = np.ones((2*nnzl))
        K[nnzl:2*nnzl] = -1
    M = sparse.lil_matrix((sz[0]*sz[1], sz[0]*sz[1]), dtype=np.float32)
    M[I, J] = K

    return M
########################################################


def deriv_charbonnier_over_x(x, sigma, a):
    ''' Evaluate the derivative of Charbonnier function divided by argument.

        Parameters :

            x : ndarray or scalar
                Points where the function will be evaluated

            sigma : float or int
                to define 
            a : float or int 
                to define

        Returns :

            y : ndarray or scalar
                Evaluation of the function on the point X

     '''
    y = 2*a*(sigma**2 + x**2)**(a-1)
    #y = 2 / (a)
    #y = 2
    return y


def deriv_quadra_over_x(x, sigma):
    ''' Evaluate the derivative of quadratic norm divided by argument 
    
        Parameters:

            x : ndarray or scalar
                Points where the function will be evaluated 
            sigma : float or int 
                to define

        Returns:

            y : ndarray or scalar
                Evaluation of the function on the point X
    '''
    y = 2 / (sigma**2)
    return y


def flow_operator(u, v, du, dv, It, Ix, Iy, lmbda, npixels):
    ''' Returns the right hand term for the linear system with the form :math:`A\\times dU = b` using the quadratic norm.  
    The flow equation is linearized around UV.

    Parameters:

        u : ndarray 
            The horizontal displacement

        v : ndarray 
            The vertical displacement  

        du :  
            The horizontal  increment steps

        dv :  ndarray
            The horizontal and vertical increment steps

        It : ndarray 
            Temporal derivative

        Ix : ndarray  
            Spatial derivative with respect to x axis 

        Iy : ndarray
            Spatial derivative with respect to y axis 

        lmbda : float
            The regularization parameter 

        npixels: int
            Number of pixels in the image used

    Returns:
    
        b : ndarray
            The right hand term  
    '''
    # Computing It.Ix and It.*Iy used for the right hand term
    Itx = It*Ix
    Ity = It*Iy
    # partial derivative over x =2 in the case of the quadratique norm
    pp_d = 2
    #N, M = It.shape

    # Calculate the right hand term

    b = lmbda*cp.vstack((cp.reshape(laplace(u+du), (npixels, 1), 'F'),
                         cp.reshape(laplace(v+dv), (npixels, 1), 'F')))

    b = b-cp.vstack((pp_d*cp.reshape(Itx, (npixels, 1), 'F'),
                     pp_d*cp.reshape(Ity, (npixels, 1), 'F')))

    return b
#############################################################


def compute_flow_base(Image1, Image2, max_iter, max_linear_iter, u, v, lmbda, size_median_filter, uhat, vhat, lambda2, lambda3, remplacement,  Ix, Iy, Ix2, Iy2, Ixy, xx, yy, sz,Li_osher):
    '''COMPUTE_FLOW_BASE  function for computing flow field using u,v displacements as an initialization.

    Parameters:
 
        Image1 : ndarray
            The first image of the sequence. 
        Image2 : ndarray
            The second image of the sequence.
        max_iter : int 
            Number of warping iterations 
        max_linear_iter: int
            The maximum number of linearization steps performed per warping.
        u: ndarray
           The horizontal displacement
        v:  ndarray
            The vertical displacement
        lmbda : float
            The regularization parameter 
        size_median_filter :  
            The size of the used median filter or the size of window  used to compute the classical or Li and Osher median filter  

        uhat : ndarray
            Horizontal auxiliar displacement fields 
        vhat : ndarray
            Vertical auxiliar displacement fields 

        lambda2: float 
            Weight for coupling term 
        lambda3: float 
            Weight for non local term 
        remplacement: bool
            If True the flow fileds will be remplaced by the auixilary fields.
        Ix : ndarray
            Spatial derivative with respect to x axis
        Iy : ndarray
            Spatial derivative with respect to y axis
        Ixx :ndarray
            The element-wise square of Ix
        Iyy : ndarray
            The element-wise square of Iy
        Ixy : ndarray
            The element-wise product of Ix and Iy
        xx :  ndarray
            to define
        yy : ndarray
            to define
        sz : ndarray
            to define

    Returns:
    
        u : ndarray
            The new computed  horizontal flow field
        v : ndarray
            The new computed  vertical flow field
        uhat: ndarray
            The new computed auixilary horizontal flow field

        vhat: ndarray
            The new computed auixilary vertical flow field 
        '''
    N, M = Image1.shape
    npixels = N*M

    # charbonnier_over_x=np.vectorize(deriv_charbonnier_over_x)

    # Lambda2 to use
    Lambdas = np.logspace(math.log(1e-4), math.log(lambda2), max_iter)
    lambda2_tmp = Lambdas[0]

    residu = []
    # Warping steps loop
    for i in range(max_iter):
        # Initialization

        du = cp.zeros((u.shape))
        dv = cp.zeros((v.shape))
        # Getting the new temporal derivative

        It = derivatives(Image1, Image2, u, v, xx, yy, sz)

        # Computing residual

        residu.append(cp.linalg.norm(It)/npixels)

        # Loop linearization
        for j in range(max_linear_iter):

            # Right hand term of the problem

            b = flow_operator(u, v, du, dv, It, Ix, Iy, lmbda,
                              npixels)

            tmp0 = cp.reshape(cp.hstack((u-uhat, v-vhat)),
                              (2*N*M, 1), 'F')

            # Taking into account the auxiliary fields  in the right hand term
            b = b - 2.*lambda2_tmp*tmp0

            ''' #This section was used to test the effect of different krylov solvers, calling scipy linalg sparse solvers 
            # x=scipy.sparse.linalg.spsolve(A,b)  #Direct solvers
            # y=scipy.sparse.linalg.gmres(A,b)  #Gmres  solver
            # y=scipy.sparse.linalg.bicg(A,b) #BICg Solver
            # y=scipy.sparse.linalg.lgmres(A,b) # LGMRES
            # diag=1/A.diagonal()
            if(i==0):
                x=None
            '''
            # Solving the linear system to get du and dv the increment step

            x = sop.minres(Ix2.astype(np.float32), Iy2.astype(np.float32), Ixy.astype(
                np.float32), float(lmbda), float(lambda2_tmp), b.astype(np.float32), 300, 10**-5, N, M)

            x[x > 1] = 1
            x[x < -1] = -1

            # Reshape x to get du and dv

            du = cp.reshape(x[0:npixels], (N, M), 'F')
            dv = cp.reshape(x[npixels:2*npixels],
                            (N, M), 'F')

        # Testing if the norm of the steps is  small less than 10**-6

        if((cp.linalg.norm(du)+cp.linalg.norm(dv))/(npixels) < 10**-6):

            break
        # Print the warping step

        print('\t \t \tWarping step', i)

        # Updating fields

        u = u + du
        v = v + dv

        '''
        # Denoising using Li & Osher formula
        uhat=lo.denoise_LO (u, size_median_filter, lambda2_tmp/lambda3, itersLO) # Denoising LO new formula of optimization  
        vhat=lo.denoise_LO (v, size_median_filter, lambda2_tmp/lambda3, itersLO)'''
        #[uhat,vhat]=lo.denoise_LO (u,v, size_median_filter, lambda2_tmp/lambda3, itersLO)

        # Denoising using a normal median filter (This in an approximation of LO median filter)
        if (Li_osher==1):
            [uhat,vhat]=denoise_LO (u,v, size_median_filter, lambda2_tmp/lambda3, 1)
        if(Li_osher==0):

            uhat = median_filter(u, size=size_median_filter)
            vhat = median_filter(v, size=size_median_filter)

        if remplacement == True:
            u = uhat
            v = vhat
        # Update Lambda2
        if i != max_iter-1:
            lambda2_tmp = Lambdas[i+1]

    # Printing the residual in the end of each leavel
    print('Residual:\n', residu)

    return [u, v, uhat, vhat]
