import numpy as np
from numpy.core.fromnumeric import reshape
from numpy.core.numeric import zeros_like 
import scipy.ndimage
import cv2
from math import ceil,floor
from scipy.ndimage.filters import convolve as filter2
from scipy.sparse import spdiags
from scipy.signal import medfilt
from scipy.ndimage import median_filter
import scipy.sparse as sparse
import math
from scipy.ndimage import correlate
#######################################################################""
def scale_image(Image,vlow,vhigh):
    ilow= np.min(Image)
    ihigh= np.max(Image)
    imo = (Image-ilow)/(ihigh-ilow) * (vhigh-vlow) + vlow
    return  imo

############################################################################
def decompo_texture(im, theta, nIters, alp, isScale):
    IM   = scale_image(im, -1,1)
    #print(IM)
    im= scale_image(im, -1,1)
    '''print('im')

    print(im)'''
    p=np.zeros((im.shape[0],im.shape[1],2),dtype=np.float32)
    #delta = 1.0/(4.0*theta)
    delta=1.0/(8.0*theta)
    I=np.squeeze(IM)
    for iter in range (nIters):

        #Compute divergence        eqn(8)    
        #div_p =filter2(p[:im.shape[0],:im.shape[1],0], np.array([[-1, 1, 0]]))+ filter2(p[:im.shape[0],:im.shape[1],1], np.array( [[-1]  , [1], [0]]))
        div_p =correlate(p[:im.shape[0],:im.shape[1],0], np.array([[-1, 1, 0]]),mode='wrap' )+ correlate(p[:im.shape[0],:im.shape[1],1], np.array( [[-1]  , [1], [0]]), mode='wrap')
        '''print('div')
        print(div_p)'''
        
        I_x = filter2(I+theta*div_p, np.array([[1, -1]]))
         
        I_y = filter2(I+theta*div_p, np.array([ [1],[-1] ]))
        
        # Update dual variable      eqn(9)
        p[:im.shape[0],:im.shape[1],0] = p[:im.shape[0],:im.shape[1],0] + delta*I_x
        p[:im.shape[0],:im.shape[1],1] = p[:im.shape[0],:im.shape[1],1] + delta*I_y
        
        # Reproject to |p| <= 1     eqn(10)    

        reprojection = np.maximum(1.0,  np.sqrt( np.multiply( p[:im.shape[0],:im.shape[1],0],p[:im.shape[0],:im.shape[1],0])+ 
        np.multiply(p[:im.shape[0],:im.shape[1],1],p[:im.shape[0],:im.shape[1],1])    )      )
        #print('repre',reprojection)
        p[:im.shape[0],:im.shape[1],0] = p[:im.shape[0],:im.shape[1],0]/reprojection
        p[:im.shape[0],:im.shape[1],1] = p[:im.shape[0],:im.shape[1],1]/reprojection
        #print(p[:im.shape[0],:im.shape[1],0])
    
    # compute divergence    
    div_p = correlate(p[:im.shape[0],:im.shape[1],0], np.array([[-1, 1, 0]] ),mode='wrap' ) +  correlate(p[:im.shape[0],:im.shape[1],1], np.array( [[-1]  , [1], [0]]),mode='wrap')

    #compute structure component
    IM[:im.shape[0],:im.shape[1]] = I + theta*div_p
    '''print('IM')
    print(IM)'''

    if (isScale):
        '''print('im')
        print(im)
        print('alp*im')
        print(alp*im)'''
        texture   = np.squeeze(scale_image((im - alp*IM), 0, 255))
        structure = np.squeeze(scale_image(IM, 0, 255))
    else:
        texture   = np.squeeze(im - alp*IM)
        structure = np.squeeze(IM)
    
    return [texture, structure]
################################################################################################
def compute_auto_pyramd_levels(Im,spacing):
    N1 = 1 + math.floor( math.log(max(Im.shape[0], Im.shape[1])/16)/math.log(spacing) )
    #smaller size shouldn't be less than 6
    N2 = 1 + math.floor( math.log(min(Im.shape[0], Im.shape[1])/6)/math.log(spacing) )
    pyramid_levels  =  min(N1, N2)
    
    '''if this.old_auto_level
        this.pyramid_levels  =  1 + floor( log(min(size(images, 1),...
            size(images,2))/16) / log(this.pyramid_spacing) );'''
    return pyramid_levels



################################################################################################
'''Image=np.array([[7.6243546 , 8.76405184, 0.93504855, 9.80596673, 8.81201369],
       [5.77520562, 7.52720659, 7.20116081, 0.80966906, 0.56304753],
       [2.86931799, 4.53189126, 9.24839657, 3.15723941, 2.6250291 ],
       [1.06423688, 7.17828962, 8.30585961, 1.51904165, 0.60133615],
       [4.94036009, 8.48463488, 9.34919316, 4.0668727 , 9.53326332]])
theta   = 1/8 
nIters  = 100
alp= 0.95
isScale=True 
[tex,stru]=decompo_texture(Image, theta, nIters, alp, isScale)
print('tex')
print(tex)
print('stru')

print(stru)
'''
