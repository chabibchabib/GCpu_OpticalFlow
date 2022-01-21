============
Introduction
============

This code is an accelerated open-source software for 2D optical flow estimation and mechanical strain.   
The code is based on `D. Sun <https://cs.brown.edu/people/dqsun/pubs/cvpr_2010_flow.pdf>`_  method with some improvements.

Requirements
------------

There are two versions of the software: a CPU and a GPU version.  
To use the code, you  will need Python 3 (3.6 or higher) with the following modules:  

A. For the CPU version:  
-----------------------

- `Numpy <https://numpy.org/>`_ 1.20.3 or newer     
- `Scikit-image <https://scikit-image.org/>`_ 0.16.2  or newer    
- `Scipy <https://scipy.org>`_ 1.6.3 or newer   
- `OpenCV <https://opencv.org/>`_ 4.2.0 or newer   



.. note::
   You may also need `Numba <https://numba.pydata.org/>`_ if Li and Osher filter will be used

B. For the CPU version:
-----------------------    
For the GPU version you will first need an  NVIDIA CUDA GPU with the Compute Capability 3.0 or larger.     
Beside the previous packages some additional ones with some and APIs will be needed to run the code on GPU

- `Cupy <https://cupy.dev/>`_      
- `cuCIM API <https://docs.rapids.ai/api/cucim/stable/api.html>`_ 

.. note::
   The GPU version was tested using `Cupy <https://cupy.dev/>`_ 9.2.0 and `cuCIM API <https://docs.rapids.ai/api/cucim/stable/api.html>`_ 21.10.01  


