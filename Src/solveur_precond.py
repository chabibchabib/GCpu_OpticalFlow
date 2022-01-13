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
                Ix^2 square of the spatial derivative with respect to x (Ix)

            Iy2: ndarray 
                Iy^2 square of the spatial derivatives with respect to y (Iy)

            Ixy: ndarray
                The product of Ix and Iy 

            lmbda: float
                The parameter of regularization

            lmbda2: float 
                The term related to the auxiliary fields uhat and vhat

            U: ndarray
                An 2*M*N vector
            N: int
                Number of the image rows. 
            -M: int
                Number of the image cols

        Returns:

            res: ndarray
                The product of A*U   
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
    return res
###############################################################################


def minres(Ix2, Iy2, Ixy, lmbda, lmbda2, b, maxiter, rtol, N, M):
    '''
    This is an implementation of minres code used in scipy.sparse and the funcion minres of Matlab 
    But it's adapted to solve the pb of the form P^-1*A*x=P^-1*b ; where A is the optical flow matrix, b the right hand term
    And P is a chosen Preconditionner.

    A has the following form: A=[Ix^2+lmbda2+2*laplacian Matrix Ixy; Ixy Iy^2+lmbda2+2*laplacian Matrix ].

    The right hand term must be a vecor containing 2*M*N element where (N,M) is the dimension of the images used.
    A is a 2*M*N square symmetric matrix.

    In this implementation we will not construct the matrices A and P, we will only use the 
    functions my_dot (and the function Px respectively) to show the algorithm how to cumpute the Matrix-vector product of Ax (and 
    P^-1*x  respectively).
    
    For more information about the solver: https://web.stanford.edu/group/SOL/software/minres/

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
    # Initialization
    eps = 1e-11
    realmax = 1.7977e+308
    istop = 0
    Anorm = 0
    Acond = 0
    rnorm = 0
    ynorm = 0
    y = b
    r1 = b
    '''if type(precond)=='numpy.ndarray':'''
    # y=mul(precond,b)
    y = Px(Ix2, Iy2, lmbda, lmbda2, N, M, b)

    beta1 = cp.sum(b*y)
    if (beta1 <= 0):
        istop = 9
    beta1 = cp.sqrt(beta1)
    oldb = 0
    beta = beta1
    dbar = 0
    epsln = 0
    qrnorm = beta1
    phibar = beta1
    rhs1 = beta1
    rhs2 = 0
    tnorm2 = 0
    gmax = 0
    gmin = realmax
    cs = -1
    sn = 0
    w = cp.zeros_like(r1)
    w2 = cp.zeros_like(r1)
    x = cp.zeros_like(r1)
    r2 = r1
    itn = 0
    # Main Loop

    while (itn < maxiter):
        itn = itn+1
        s = 1/beta
        v = s*y
        # y=mul(A,v)

        y = my_dot2(Ix2, Iy2, Ixy, lmbda, lmbda2, v, N, M)
        if (itn >= 2):
            y = y-(beta/oldb)*r1
        alpha = cp.sum(v*y)
        y = (-alpha/beta)*r2+y
        r1 = r2
        r2 = y
        # y=mul(precond,r2)
        y = Px(Ix2, Iy2, lmbda, lmbda2, N, M, r2)

        oldb = beta
        beta = cp.sum(r2*y)
        if beta <= 0:
            istop = 9
            break
        beta = cp.sqrt(beta)
        tnorm2 = tnorm2+alpha**2+oldb**2+beta**2
        if (itn == 1):
            if((beta/beta1) < (10*eps)):
                istop = -1
                break

        oldeps = epsln
        delta = cs*dbar+sn*alpha
        gbar = sn*dbar-cs*alpha
        epsln = sn*beta
        dbar = -cs*beta
        root = cp.sqrt(gbar**2+dbar**2)
        Anorm = phibar*root
        # print("Anorm",root)

        gamma = cp.sqrt(gbar**2+beta**2)
        gamma = max(gamma, eps)
        cs = gbar/gamma
        sn = beta/gamma
        phi = cs*phibar
        phibar = sn*phibar

        denom = 1/gamma
        w1 = w2
        w2 = w
        w = (v-oldeps*w1-delta*w2)*denom
        x = x+phi*w

        gmax = max(gmax, gamma)
        gmin = min(gmin, gamma)
        z = rhs1/gamma
        rhs1 = rhs2-delta*z
        rhs2 = -epsln*z

        Anorm = cp.sqrt(tnorm2)
        ynorm = cp.linalg.norm(x)
        epsa = Anorm*eps
        epsx = Anorm*ynorm*eps
        epsr = Anorm*ynorm*rtol
        diag = gbar
        if diag == 0:
            diag = epsa

        qrnorm = phibar
        rnorm = qrnorm
        test1 = rnorm/(Anorm*ynorm)  # ||r|| / (||A|| ||x||)
        test2 = root / Anorm          # ||Ar{k-1}|| / (||A|| ||r_{k-1}||)
        Acond = gmax/gmin
        if istop == 0:

            t1 = 1+test1
            t2 = 1+test2
            if(t2 <= 1):
                istop = 2
            if(t1 <= 1):
                istop = 1
            if (itn > maxiter):

                istop = 6
            if(Acond >= 0.1/eps):
                istop = 4
            if(epsx >= beta1):
                istop = 3
            if(test2 <= rtol):
                istop = 2
        if(test1 <= rtol):
            istop = 1

        if(istop != 0):
            break

    return x


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
