This code is an accelerated open-source software for 2D optical flow estimation and mechanical strain.   
The code is based on `D. Sun <https://cs.brown.edu/people/dqsun/pubs/cvpr_2010_flow.pdf>`_ :footcite:p:`DSun`  method with some improvements.

.. bibliography:: Bib.bib
Requirements
============

There are two versions of the software: a CPU and a GPU version.  
To use the code, you will need Python 3 (3.6 or higher) with the following modules:  

A. For the CPU version:  
-----------------------

- `Numpy <https://numpy.org/>`_ 1.20.3 or newer     
- `Scikit-image <https://scikit-image.org/>`_ 0.16.2  or newer    
- `Scipy <https://scipy.org>`_ 1.6.3 or newer   
- `OpenCV <https://opencv.org/>`_ 4.2.0 or newer   



.. note::
   You may also need `Numba <https://numba.pydata.org/>`_ if Li and Osher filter :footcite:p:`LiOsher` will be used

B. For the GPU version:
-----------------------    
For the GPU version you first need an NVIDIA CUDA GPU with the Compute Capability 3.0 or larger.     
Beside the previous packages some additional ones will be needed to run the code on GPU

- `Cupy <https://cupy.dev/>`_      
- `cuCIM API <https://docs.rapids.ai/api/cucim/stable/api.html>`_ 

.. note::
   The GPU version was tested using `Cupy <https://cupy.dev/>`_ 9.2.0 and `cuCIM API <https://docs.rapids.ai/api/cucim/stable/api.html>`_ 21.10.01  



Method
=======

A. Energy:
----------

Energy to minimize:

.. math::


   \begin{equation*}
   \begin{aligned}
   E(u,v,\hat{u},\hat{v}) =& \textcolor{black}{\sum_{i,j}{\Bigg\{ \rho_D( I_1(i,j)-I_2(i+u_{ij},j+v_{ij}) ) + \lambda[\rho_S(u_{ij}-u_{i+1,j})}  }\\
   &+\textcolor{black}{\rho_S(u_{ij}-u_{i,j+1})+\rho_S(v_{ij}-v_{i+1,j})+\rho_S(v_{ij}-v_{i,j+1})]\Bigg\} }  \\
   &‌+ \textcolor{blue}{\lambda_2(\mid\mid u-\hat{u} \mid\mid^2+\mid\mid v-\hat{v} \mid\mid^2)}\\
   &+\textcolor{OliveGreen}{\sum_{i,j}{\sum_{(i',j')\in N_{i,j}}{\lambda_3(\mid{\hat{u}_{ij}-\hat{u} _{i'j'}\mid}+\mid{\hat{v}_{ij}-\hat{v}_{i'j'}}\mid)}}}
   \end{aligned}
   \end{equation*}

Where:

   - :math:`\textcolor{black}{\text{Term in braces is the same as classical methods}}`

   - :math:`\textcolor{blue}{\text{Encourages $(\hat{u}, \hat{v})$ and $(u,v)$ to be the same}}`

   - :math:`\textcolor{OliveGreen}{\text{Smoothes $\hat{u}, \hat{v}$}}`

.. list-table:: Parameters and their definition
   :widths: 25  50
   :header-rows: 1

   * - Parameter
     - Definition
   * - :math:`\hat{u}`
     - The auxiliary horizontal flow fields
   * - :math:`\hat{v}`
     - The auxiliary vertical flow fields
   * - :math:`N_{ij}` 
     - The neighbouring pixels of the pixel (i,j)
   * - :math:`\lambda _2 \text{ and }\lambda _3`
     - Scalar weights
   * - :math:`\rho_D \text{ and }\rho_D`
     - Penalties

B. Minimization
---------------
The previous objective will be optimized  by alternately minimizing:


.. math::


   \begin{equation*}
   \begin{aligned}
   E_O(u,v) =&\sum_{i,j}\rho _D( I_1(i,j)-I_2(i+u_{ij},j+v_{i,j}) ) \\
   &+\lambda[\rho_S(u_{ij}-u_{i+1,j})+\rho_S(u_{ij}-u_{i,j+1})+\rho_S(v_{ij}-v_{i+1,j}) \\
   & +\rho_S(v_{ij}-v_{i,j+1}) ]+\lambda_2(\mid\mid u-\hat{u}\mid\mid^2+\mid\mid v-\hat{v} \mid\mid^2)  \\
   \end{aligned}
   \end{equation*}

And

.. math::


   \begin{equation*}
   \begin{aligned}
   E_M(\hat{u},\hat{v})=&  \lambda_2(\mid\mid  u-\hat{u}\mid\mid ^2+\mid\mid v-\hat{v}\mid\mid ^2)\\
   &+\sum_{i,j}\sum_{ (i',j') \in N_{i,j} } \lambda_3(\mid{\hat{u}_{ij}-\hat{u}_{i'j'}}\mid+\mid \hat{v}_{ij}-\hat{v}_{i'j'}\mid) \text{           }      \\  
   \end{aligned}
   \end{equation*}

:math:`E_O` will be minimized first during the alterning optimization with :math:`\hat{u},\hat{v}` fixed. Then with fixed :math:`u,v` we minimize :math:`E_M`

:math:`E_M` will be optimized by applying Li and Osher's median filter or approximately optimized by applying a simple median filter. 

C. Summary of the method
------------------------

   - Build a pyramid of the images.
   - Using the following derivation kernel: :math:`h=\frac{1}{12}[-1, 8, 0 ,-8 ,1]`
   - Cancelling the derivatives at pixels where the movement is out of the edges
   - Compute the steps :math:`du,dv` by solving a linear system using Preconditionned Minres. 
   - Update :math:`u` and :math:`v` 
   - Compute :math:`\hat{u},\hat{v}` by  Li and Osher or a simple median filter at each iteration

