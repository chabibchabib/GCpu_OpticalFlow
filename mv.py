import numpy as np 
import scipy.ndimage
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import minres
#####################################################
#lmbda=5
M=2
N=2
'''Ix=np.round(10*np.random.rand(N,M))
Iy=np.round(10*np.random.rand(N,M))
u0=np.round(10*np.random.rand(N,M))
v0=np.round(10*np.random.rand(N,M))'''
Ix=0.0001*np.array([[ 5.  ,4.],[ 4. ,10.]])
Iy=0.0001*np.array([[2. ,7.],[6. ,5.]])
u0=0.0001*np.array([[2. ,6.],[9., 4.]])
v0=0.0001*np.array([[5., 0.],[3., 2.]])
lmbda0=1
def mv(U,u=u0,v=v0,lmbda=lmbda0,Ix1=Ix,Iy1=Iy):
    '''u=U[:N*M]
    v=U[N*M:]'''
    u=np.reshape(u,(N*M,1),order='F')
    v=np.reshape(v,(N*M,1),order='F')

    Ix1=np.reshape(Ix1,(N*M,1),order='F')
    Ix1[Ix1==0]=0.0001
    Iy1=np.reshape(Iy1,(N*M,1),order='F')
    Iy1[Iy1==0]=0.0001
    Ix1=1/Ix1
    Iy1=1/Iy1
    u1=u+v
    v1=u1
    u=u*Ix1
    u=scipy.ndimage.laplace(u)
    u=u*Ix1
    u1=u1+lmbda*u

    v=v*Iy1
    v=scipy.ndimage.laplace(v)
    v=v*Iy1
    v1=v1+lmbda*v
    return np.array(([u1,v1]))

    #u1=np.reshape(u,(u.shape[0]*u.shape[1],1))


npixels=M*N

#v=np.random.rand(N,M)

'''A=LinearOperator((2*npixels,2*npixels),matvec=mv)
U=np.random.rand(N*M*2)
print(A.matvec(U))'''
'''def mv(U):
    u=U[:N,:]
    v=U[N:,:]
    print('shape of u :) ' ,U.shape)
    #Ix1=np.ravel(Ix)

    #Iy1=np.ravel(Iy)
    Ix1=1/Ix
    Iy1=1/Iy
    u1=u+v
    v1=u1
    u=u*Ix1
    u=scipy.ndimage.laplace(u)
    u=u*Ix1
    u1=u1+lmbda*u

    v=v*Iy1
    v=scipy.ndimage.laplace(v)
    v=v*Iy1
    v1=v1+lmbda*v
    return np.array(([u1,v1]))'''

'''U=np.round(10*np.random.rand(2*N,M))
U=np.ravel(U)'''
#print(U.shape)
x=0.001*np.array( [5, 8, 9, 6, 7, 8,1, 7])
A=LinearOperator((2*npixels,2*npixels),matvec=mv)
#print(A.matvec(U).shape)
b,exitcode=minres(A,x)
print('solution:\n',b)
print('x\n',x)
print("Ix\n",Ix)
print("Iy\n",Iy)
print("u\n",u0)
print("v\n",v0)
