# GCpu_OpticalFlow
This code is an accelerated open-source software for 2D optical flow estimation and mechanical strain.   
The code is based on D. Sun method with some improvements. 
## Requirements
There are two versions of the software: a CPU and a GPU version.
You will need Python 3 (3.6 or higher) with the following modules:  
**For the CPU version:**  
- [Numpy](https://numpy.org/) 1.20.3 or newer  
- [Scikit-image](https://scikit-image.org/) 0.16.2  or newer  
- [Scipy](https://scipy.org/) 1.6.3 or newer    
**For the CPU version:**

For the GPU version you will first need need an  NVIDIA CUDA GPU with the Compute Capability 3.0 or larger   
Beside [Numpy] (https://numpy.org/) some additionnal packages and APIs will be needed  
- [Cupy](https://cupy.dev/)    
- [cuCIM API](https://docs.rapids.ai/api/cucim/stable/api.html)  

