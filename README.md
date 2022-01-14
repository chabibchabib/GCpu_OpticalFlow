# GCpu_OpticalFlow
This code is an accelerated open-source software for 2D optical flow estimation and mechanical strain.   
The code is based on [D. Sun](https://cs.brown.edu/people/dqsun/pubs/cvpr_2010_flow.pdf) method with some improvements. 
## Requirements
There are two versions of the software: a CPU and a GPU version.
You will need Python 3 (3.6 or higher) with the following modules:  
**For the CPU version:**  
- [Numpy](https://numpy.org/) 1.20.3 or newer  
- [Scikit-image](https://scikit-image.org/) 0.16.2  or newer  
- [Scipy](https://scipy.org/) 1.6.3 or newer 
- [OpenCV](https://opencv.org/) 4.2.0 or newer      
   
**For the CPU version:**  
For the GPU version you will first need an  NVIDIA CUDA GPU with the Compute Capability 3.0 or larger.   
Beside the previous packages some additional ones with some and APIs will be needed to run the code on GPU 
- [Cupy](https://cupy.dev/)    
- [cuCIM API](https://docs.rapids.ai/api/cucim/stable/api.html)
The GPU version was tested using [Cupy](https://cupy.dev/) 9.2.0 and [cuCIM API](https://docs.rapids.ai/api/cucim/stable/api.html) 21.10.01  

## Documentation
**In progress**
## Test
Some image sequences were given in the folder **Images**  in order to test the code.  
**Inputs:**  
The software will need at least 3 inputs:
   - The image sequence path 
   - The name of the first image of the sequence 
   - The name of the second image of the sequence 

The other parameters were set by default, but you can modify them also using the keywords as how it was described in the documentation.   
**Outputs:**  
Two **npy** files will be generated in the end of running:
   - **u_cucim.npy** the horizontal flow field
   - **v_cucim.npy** the vertical flow field

To calculate the strain field, you can use the [function gradient of numpy](https://numpy.org/doc/stable/reference/generated/numpy.gradient.html) or of [cupy](https://docs.cupy.dev/en/stable/reference/generated/cupy.gradient.html) if you are using the graphic card.




