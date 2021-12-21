import numpy as np 
import cv2
from scipy import signal
######################################################
def Charbonnier(x,a,eps):
    ''' Fonction Charbonnier generale'''
    y=(x**2+eps**2)**a
    return y 
######################################################
def quadratique(x):
    y=x**2
    return y 
#######################################################
def energie_image(Im1,Im2,u,v):
    N,M=Im1.shape
    y=np.linspace(0,N-1,N)
    x=np.linspace(0,M-1,M)
    x,y=np.meshgrid(x,y)
    x2=x+u; y2=y+v
    x2=np.array(x2,dtype=np.float32)
    y2=np.array(y2,dtype=np.float32)
    I=cv2.remap(np.array(Im2,dtype=np.float32),x2,y2,cv2.INTER_LINEAR)
    I=Im1-I
    res=np.sum(quadratique(I))
    return res
######################################################
def energie_grad_dep(u,v,lmbda):
    ul=np.empty_like(u);    uc=np.empty_like(u); vl=np.empty_like(u);    vc=np.empty_like(u)
    (N,M)=u.shape
    ul[:N-1,:]=u[:N-1,]-u[1:N,:]; 
    ul[N-1,:]=u[N-1,:]-u[N-2,:]
    vl[:N-1,:]=v[:N-1,]-v[1:N,:]; 
    vl[N-1,:]=v[N-1,:]-v[N-2,:]
    uc[:,:M-1]=u[:,:M-1]-u[:,1:M]; uc[:,M-1]=u[:,M-1]-u[:,M-2]
    vc[:,:M-1]=v[:,:M-1]-v[:,1:M]; vc[:,M-1]=v[:,M-1]-v[:,M-2]
    #res=lmbda*np.sum(ul**2+uc**2+vl**2+vc**2)
    res=np.sum(ul**2+uc**2+vl**2+vc**2)
    return res

def energie1(Im1,Im2,u,v,lmbda):
    return (energie_grad_dep(u,v,lmbda)+energie_image(Im1,Im2,u,v))
######################################################
def energie_champs_aux1(u,uhat,v,vhat,lambda2):
    #res=lambda2*np.sum(((u-uhat)**2+(v-vhat)**2))
    res=np.sum(((u-uhat)**2+(v-vhat)**2))
    return res
######################################################
def energie_champs_aux2(uhat,vhat,lambda3,size_m):
    kernel=np.ones((size_m,size_m))
    un=signal.convolve2d(uhat, kernel,mode='same',boundary='symm')
    vn=signal.convolve2d(vhat, kernel,mode='same',boundary='symm')
    Res=np.abs(un-uhat)+np.abs(vhat-vn)
    #res=lambda3*np.sum(Res)
    res=np.sum(Res)
    return res
############################################################
def energie2(u,uhat,v,vhat,lambda2,lambda3,size_m):
    return( energie_champs_aux1(u,uhat,v,vhat,lambda2)+energie_champs_aux2(uhat,vhat,lambda3,size_m))


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