Solveur precond
===============
It's a matrix-free implementation of Minres solver used in scipy.sparse and the function Minres of Matlab. This function is adapted to solve the following problem.    	                                 
:math:`P^{-1}Ax=P^{-1}b`.

Where :math:`A` is the optical flow matrix, :math:`b` the right hand term and :math:`P` is the chosen preconditioner already defined in Precond file.

If we are handling :math:`N\times M` images, then the problem solved has :math:`2\times N\times M` as size.    
The file contains also a function that computes the  matrix-vector product of matrix of the problem  and  a  given vector.  

Note: The Laplace operator of certain image is computed using a convolution instead of computing it with a matricial product with the Laplace matrix.


.. automodule:: solveur_precond  
   :members:
   :undoc-members:
   :show-inheritance:
