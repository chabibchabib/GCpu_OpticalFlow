Solveur precond
===============
It’s a matrix-free implementation of Minres solver  used in scipy.sparse and the function Minres of Matlab. This function is adapted to solve the following problem.    	                                 P^-1*A*x=P^-1*b.
Where A is the optical flow matrix, b the right hand term and P is the chosen preconditioner already defined in Precond file.  
If we are working with the  N x M images, then the problem solved have 2xNxM as size.    
The file contains also a function that computes the  matrix-product of matrix of the problem (A) and  a  given vector.  

.. automodule:: solveur_precond  
   :members:
   :undoc-members:
   :show-inheritance:
