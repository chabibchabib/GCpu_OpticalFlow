import numpy as np
from utils import *
from scipy.sparse.linalg import LinearOperator
import scipy.sparse as sparse
from precond import Px

##########################################################################


def my_dot2(Ix2, Iy2, Ixy, lmbda, lmbda2, U, N, M):
    ''' The matrix vector product of the unconstructed optical flow matrix 
    A and a vector U

        Parameters:

            Ix2 : ndarray
                Ix^2 square of the spatial derivative with respect to x :math:`(I_x)`

            Iy2 : ndarray 
                Iy^2 square of the spatial derivatives with respect to y :math:`(I_y)`

            Ixy : ndarray
                The product of Ix and Iy 

            lmbda : float
                The parameter of regularization

            lmbda2 : float 
                The term related to the auxiliary fields uhat and vhat

            U : ndarray
                A :math:`2\\times M\\times N`  vector

            N : int
                Number of the image rows

            M : int
                Number of the image cols

        Returns:

            res : ndarray
                The product of A and U
               
    '''
    npix = N*M  # Number of pixels

    # Reshape the vector U to get u and v
    u = cp.reshape(U[:npix], (N, M), order='F')
    v = cp.reshape(U[npix:2*npix], (N, M), order='F')

    # The term related to u is given by: u*Ix**2-derivative_overx*lmbda*laplacien u +Ix*Iy*v
    u1 = u*(Ix2+2*lmbda2)+Ixy*v-2*lmbda*laplace(u)
    # The term related to v is given by: v*Iy**2-derivative_overx*lmbda*laplaceien v  + Ix*Iy*u
    v1 = v*(Iy2+2*lmbda2)+Ixy*u-2*lmbda*laplace(v)

    # Linearize  u1 and v2
    u1 = cp.reshape(u1, (npix, 1), order='F')
    v1 = cp.reshape(v1, (npix, 1), order='F')

    # Allocation
    res = cp.empty((2*npix, 1), dtype=np.float32)

    # Storage
    res[:npix] = u1
    res[npix:2*npix] = v1

    return res.ravel()





def minres(Ix2, Iy2, Ixy, lmbda, lmbda2, b, maxiter, rtol, N, M):
    '''
    
    This is an implementation of `Minres <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.minres.html>`_ code used in scipy.sparse  and the funcion minres of Matlab 
    But it's adapted to solve the pb of the form:

    .. math::
        P^{-1}A x=P^{-1}b

    where A is the optical flow matrix, b the right hand term
    And P is a chosen Preconditionner.

    A has the following form:

    .. math::

        A=
        \\begin{pmatrix}
            I_x^2+\\lambda _2 +2\\bigtriangleup & I_x \\times I_y \n
            I_x\\times I_y & I_y^2+\\lambda _2+2\\bigtriangleup  
        \\end{pmatrix}
    
    The right hand term b must be a vecor containing :math:`2\\times M \\times N` element where (N,M) is the dimension of the images used.
    A is a :math:`2\\times M \\times N` square symmetric matrix.

    In this implementation we will not construct the matrices A and P, we will only use the 
    functions my_dot (and the function :math:`Px` respectively) to show the algorithm how to cumpute the Matrix-vector product of :math:`Ax` (and 
    :math:`P^{-1}x`  respectively).
    
    For more information about the solver: `MINRES <https://web.stanford.edu/group/SOL/software/minres/>`_ 

        Parameters:

            Ix2 : ndarray
                The elementwise square of the  matrix Ix, where Ix is the spatial derivative with respect to x  of the refererence image.

            Iy2 : ndarray
                The elementwise square of the  matrix Iy, where Iy is the spatial derivative with respect to y of the refererence image.

            Ixy : ndarray
                The elementwise product of Ix and Iy

            lmbda : float
                The parameter of regularization

            lmbda2 : float
                The term related to the auxiliary fields uhat and vhat 

            b : ndarray
                The right hand term 

            maxiter : int
                Maximum number of iterations 

            rtol: float
                Relative tolerance  
            N : int
                Number of rows of the image 
            M : int
                Number of cols of the image

        Returns:

            We can also display the variable istop to know the reason why the solver leaved loop   
            x : ndarray
                The solution of the system Ax=b
    '''
    b = b.ravel()
    v0 = cp.zeros_like(b)
    w0 = cp.zeros_like(b)
    w1 = cp.zeros_like(b)
    x = cp.zeros_like(b)

    #q0 = mv(x,lmbda,lmbda2,Ix,Iy,N,M).ravel()
    q0=my_dot2(Ix2, Iy2, Ixy, lmbda, lmbda2, x, N, M)
    v1 = b-q0
    gamma0 = cp.linalg.norm(v1)
    z1 = Px(Ix2, Iy2, lmbda, lmbda2, N, M, v1).ravel()
    gamma1 = cp.sqrt(cp.inner(z1,v1))
    eta = gamma1
    s0 = 0
    s1 = 0
    c0 = -1
    c1 = -1
    it = 0
    norm_res = gamma1
    Anorm=0      
    while(it < maxiter):
        it = it+1
        #Lanczos factorization 

        z1 = z1/gamma1
        #q1 = mv(z1,lmbda,lmbda2,Ix,Iy,N,M).ravel()
        q1=my_dot2(Ix2, Iy2, Ixy, lmbda, lmbda2, z1, N, M)
        delta = cp.inner(q1,z1)
        v2 = q1-(delta/gamma1)*v1
        if it > 1:
            v2 = v2-(gamma1/gamma0)*v0
        v0 = v1
        v1 = v2

        z2 = Px(Ix2, Iy2, lmbda, lmbda2, N, M, v2).ravel()
        gamma2 = cp.sqrt(cp.inner(z2,v2))
        
        Anorm+=gamma2**2+gamma1**2+delta**2

        #Givens rotation
        alpha0 = -c1*delta-c0*s1*gamma1
        alpha1 = cp.sqrt(alpha0**2+gamma2**2)
        alpha2 = s1*delta-c0*c1*gamma1
        alpha3 = s0*gamma1
        #Computing cos and sin 

        c2 = alpha0/alpha1
        s2 = gamma2/alpha1
        w2 = (z1-alpha3*w0-alpha2*w1)/alpha1
        #Update Solution 

        x = x+c2*eta*w2
        eta = s2*eta

        #Next iteration
        w0 = w1
        w1 = w2
        gamma0 = gamma1
        gamma1 = gamma2
        c0 = c1
        c1 = c2
        s0 = s1
        s1 = s2
        z1 = z2
        #print('x[0]', x[0])
        norm_res = norm_res*abs(s2)

        ##print("minres ",it,cp.sqrt(Anorm),x[0])
        # If ||r||/(norm(A)*norm(x))<tol As in the solver of scipy
        if(norm_res < rtol*cp.sqrt(Anorm)*cp.linalg.norm(x)):
            break

    return cp.reshape(x,(2*N*M,1))





'''
#Testing and comparing it with minres solver of scipy sparse 

N=400;  lmbda=2; M=40; maxiter=300; rtol=10**-5
Ix=np.random.rand(N,M)
Iy=np.random.rand(N,M)
u0=np.random.rand(N,M)
v0=np.random.rand(N,M)
b=np.random.rand(2*N*M,1)
Ix2=Ix*Ix
Iy2=Iy*Iy
Ixy=Ix*Iy
b=b.astype(np.float32)
Ix2=Ix2.astype(np.float32)
Iy2=Iy2.astype(np.float32)
Ixy=Ixy.astype(np.float32)'''


'''
####################################################
def mv(U,lmbda,Ix,Iy,N,M):
                #code.interact(local=locals())
    u0=U[:N*M]
    v0=U[N*M:]
    u0=np.reshape(u0,(N,M),order="F")
    v0=np.reshape(v0,(N,M),order="F")
    Ix2=Ix*Ix
    Iy2=Iy*Iy
    u1=Ix2*u0+Ix*Iy*v0-2*lmbda*laplace(u0)
    v1=Iy2*v0+Ix*Iy*u0-2*lmbda*laplace(v0)
    v1=np.reshape(v1,(N*M,1),order='F')
    u1=np.reshape(u1,(N*M,1),order='F')
    return np.vstack((u1,v1))
#U=np.vstack((np.reshape(u0,(N*M,1), order='F'),np.reshape(v0,(u0.shape[0]*u0.shape[1],1), order='F')))
U=b
precond=np.eye(2*N*M,2*N*M)
precond=precond.astype(np.float32)
for i in range(2*N*M):
    precond[i,i]=2*i+1
L = LinearOperator((2*M*N,2*M*N), matvec=lambda U: mv(U,lmbda,Ix,Iy,N,M)) 
###################################################### 
t1= time()
x,exitcode=sparse.linalg.minres(L,b,M=precond)
t2= time()
xm=minres(Ix2,Iy2,Ixy,lmbda,b,maxiter,rtol,N,M,1,precond)
t3= time()          
xm2=minres(Ix2,Iy2,Ixy,lmbda,b,maxiter,rtol,N,M,2,precond)
t4= time()  
print('my x\n',xm[:10])
print('sparse x\n',x[:10],'exit ',exitcode)
print("SPARSE:", (t2-t1),"MY SOLVER: " , (t3-t2),"MY SOLVER DOT: " , (t4-t3))


print('norm sparse:\n',np.linalg.norm(mv(x,lmbda,Ix,Iy,N,M)-b))
print('norm minres:\n',np.linalg.norm(mv(xm,lmbda,Ix,Iy,N,M)-b))
print('norm minres dot :\n',np.linalg.norm(mv(xm2,lmbda,Ix,Iy,N,M)-b))

for i in range(2*N*M):
    precond[i,i]=i+1'''