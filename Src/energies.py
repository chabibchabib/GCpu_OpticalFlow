import numpy as np
import cv2
from scipy import signal
######################################################
'''
The function we are minimizing in an alternative way (noted E_A) is described in "Secrets of Optical Flow Estimation and Their Principles" by D.Sun.
It is depending on the displacement fields (u,v) and the auxiliary arrays (uhat,vhat) and It's also  devided:

'''


######################################################


def quadratique(x):
    ''' Evaluate the classical quadratic norm

        Parameters:
            x: a scalar or an ndarray
                Points where the function will be evaluated 

        Returns: 
            y: a scalar or an ndarray
                The image of x 
    '''

    y = x**2
    return y
#######################################################


def energie_image(Im1, Im2, u, v):
    '''
    Compute the energy of the first term  using the quadratic norm (Gray constancy assumption).

    Parameters: 
        Im1 : ndarray 
            The first image of the sequence 
        Im2 : ndarray 
            The second image of the sequence
        u :  ndarray
            The horizontal displacement field 
        v :  ndarray
            The vertical displacement field 

    Returns: 
        res : float 
            Computed energy 
    '''
    N, M = Im1.shape
    y = np.linspace(0, N-1, N)
    x = np.linspace(0, M-1, M)
    x, y = np.meshgrid(x, y)
    x2 = x+u
    y2 = y+v
    x2 = np.array(x2, dtype=np.float32)
    y2 = np.array(y2, dtype=np.float32)
    I = cv2.remap(np.array(Im2, dtype=np.float32), x2, y2, cv2.INTER_LINEAR)
    I = Im1-I
    res = np.sum(quadratique(I))
    return res
######################################################


def energie_grad_dep(u, v, lmbda):
    '''
        Compute the energy of term related to  the gradient of the displacements  using the quadratic norm.

        Parameters: 

            u :  ndarray
                The horizontal displacement field 
            v :  ndarray
                The vertical displacement field
            lmbda : float
                Tikhonov parameter 

    Returns: 
        res : float 
            Computed energy 
    '''
    ul = np.empty_like(u)
    uc = np.empty_like(u)
    vl = np.empty_like(u)
    vc = np.empty_like(u)
    (N, M) = u.shape
    ul[:N-1, :] = u[:N-1, ]-u[1:N, :]
    ul[N-1, :] = u[N-1, :]-u[N-2, :]
    vl[:N-1, :] = v[:N-1, ]-v[1:N, :]
    vl[N-1, :] = v[N-1, :]-v[N-2, :]
    uc[:, :M-1] = u[:, :M-1]-u[:, 1:M]
    uc[:, M-1] = u[:, M-1]-u[:, M-2]
    vc[:, :M-1] = v[:, :M-1]-v[:, 1:M]
    vc[:, M-1] = v[:, M-1]-v[:, M-2]
    res = lmbda*np.sum(ul**2+uc**2+vl**2+vc**2)
    #res = np.sum(ul**2+uc**2+vl**2+vc**2)
    return res


def energie1(Im1, Im2, u, v, lmbda):
    '''
    The energy of the classical methods (Term related to gray value constancy + Image Gradient Term).  

    Parameters: 
        Im1 : ndarray 
            The first image of the sequence 
        Im2 : ndarray 
            The second image of the sequence
        u :  ndarray
            The horizontal displacement field 
        v :  ndarray
            The vertical displacement field 
        lmbda : float
            Tikhonov parameter

    Returns: 
        res : float 
            Computed energy 
    '''
    return (energie_grad_dep(u, v, lmbda)+energie_image(Im1, Im2, u, v))
######################################################


def energie_champs_aux1(u, uhat, v, vhat, lambda2):
    ''' Energy of the  first non local term (The coupling term)

    Parameters : 
        u : ndarray 
            The horizontal displacement field 
        uhat : ndarray 
            The horizontal auxiliary flow field
        v :  ndarray
            The vertical displacement field
        vhat :  ndarray
            The vertical auxiliary flow field
        lambda2 : float
            A scalar weight
    Returns : 
        res : float 
            Computed energy  
    '''
    res = lambda2*np.sum(((u-uhat)**2+(v-vhat)**2))
    #res = np.sum(((u-uhat)**2+(v-vhat)**2))
    return res
######################################################


def energie_champs_aux2(uhat, vhat, lambda3, size_m):
    ''' Energy of the second non local term.

    Parameters: 
            uhat : ndarray 
                The horizontal auxiliary flow field
            vhat :  ndarray
                The vertical auxiliary flow field
            lambda3 : float
                A scalar weight
            size_m : int 
                The size of the median filter used.
    Returns: 
            res : float 
                Computed energy    
    '''
    kernel = np.ones((size_m, size_m))
    un = signal.convolve2d(uhat, kernel, mode='same', boundary='symm')
    vn = signal.convolve2d(vhat, kernel, mode='same', boundary='symm')
    Res = np.abs(un-uhat)+np.abs(vhat-vn)
    res = lambda3*np.sum(Res)
    #res = np.sum(Res)
    return res
############################################################


def energie2(u, uhat, v, vhat, lambda2, lambda3, size_m):
    '''
    Energy related the non local term.

        Parameters: 
            u : ndarray 
                The horizontal displacement field 
            uhat : ndarray 
                The horizontal auxiliary flow field
            v :  ndarray
                The vertical displacement field
            vhat :  ndarray
                The vertical auxiliary flow field
            lambda2 : float
                The first scalar weight
            lambda3 : float
                The second scalar weight
            size_m : int 
                The size of the median filter used.

        Returns: 
            res : float 
                Computed energy   
    '''

    return(energie_champs_aux1(u, uhat, v, vhat, lambda2)+energie_champs_aux2(uhat, vhat, lambda3, size_m))


#########################################################
'''Im1=np.random.rand(4,5)
Im2=np.random.rand(4,5)
u=np.random.rand(4,5)
v=np.random.rand(4,5)
uhat=np.random.rand(4,5)
vhat=np.random.rand(4,5)
lmbda=5
lambda2=2
#I=energie_image(Im1,Im2,u,v)
#print(I)
#print(energie_grad_dep(u,v,lmbda))
print(energie1(Im1,Im2,u,v,lmbda))
print(energie_champs_aux1(u,uhat,v,vhat,lambda2))
print(energie2(u,uhat,v,vhat,lambda2,lambda2))'''
