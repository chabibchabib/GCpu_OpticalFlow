=========
Structure
=========

A. Src
------
Is a folder containaing the main python software sources. 

Utils : 
    Utils imports the required functions and modules to run the code. The GPU version of the software will be privileged. If Utils find some missing functions or modules the CPU version will be launched. 

Precond :
    Is used to compute the matrix-vector product  of the inverse of the chosen preconditioner and a given vector. 
	This function will be used to create a matrix-free solver.

Solveur_precon :   
    It’s a matrix-free implementation of Minres solver  used in scipy.sparse and the function Minres of Matlab. This function is adapted to solve the following problem   	                                 P^-1*A*x=P^-1*b.
    Where A is the optical flow matrix, b the right hand term and P is the chosen preconditioner already defined in Precond file.
    If we are working with the  N x M images, then the problem solved have 2xNxM as size.  
    The file contains also a function that computes the  matrix-product of matrix of the problem (A) and  a  given vector.

Flow_operator :
    The rôle  of  flow_operator module is the interpolation of the images using the current computed flow fields, the construction of the right hand term of the linear system and the definition of  the base function for computing flow field that will be used at each level of the pyramid.

Compute_flow :   
    Computes the optical flow fields using a pyramidal approach and the base function already defined in flow_operator.

Energies :
    For computing the different energies described in the paper of D.Sun[.
Denoise_LO :
    Li and Osher median filter implementation.

B. Test
-------
To test the method we created a script able to read and modify the parameters of the algorithm using the sys.argv arguments.  
To run the script you can use ``./mainscript``, then you have to add at least three arguments :

1. Path of the images folder 
2. Name of the first image  of the sequence
3. Name of the second image of the sequence

**Syntax:**

.. code-block:: shell-session

    ./mainscript /Give/The/Path/Folder  Im1.extension Im2.extension 

To change the value of a desired parameter, you must add it after the executable using it's key :

.. list-table:: Variables and their keys
   :widths: 25 25 50
   :header-rows: 1

   * - Parameter
     - Keyword
     - Initial default value
   * - Number of levels in the pyramid
     - pyram_levels
     - 3
   * - The downsampling factor
     - factor
     - 2
   * - The order of interpolation [Must be between 0 and 5] 
     - ordre_inter
     - 3
   * - The size of the window used for Li and Osher or the simple median filter 
     - size_median_filter
     - 5
   * - Number of warping steps
     - max_iter
     - 10
   * - Maximum number of linearization performed per warping
     - max_linear_iter
     - 1
   * - Tikhonov Parameter
     - lmbda
     - 3*10^4
   * - Weight for coupling term
     - lambda2
     - 0.001
   * - Weight for non local term
     - lambda3
     - 1
   * - The name of the image Mask
     - Mask
     - None
   * - Path for the mask image
     - Path_Mask
     - ---

**Syntax :**

.. code-block:: shell-session

 ./mainscript /Give/The/Path/Folder  Im1.extension Im2.extension  keyword=new_value 	Mask=Mask.extension Path_Mask=/Path/Folder/OfMask 


C. Images
---------
In order to test the software, three image sequences and some masks with different shapes were given in this folder.

