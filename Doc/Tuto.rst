Tutorial
========


In this tutorial, we will quickly learn the basics of writing your own scripts for the purpose of using the code.

Before starting, you will will need to verify that all the :ref:`Requirements` for the desired version are already installed. 
First, it is mandatory to create an ``.py`` file. The main function that computes the displacement was implemented in ``compute_flow`` module and has the same name.

Therefore, we have to specify the path of  :ref:`Compute flow`. To do this, it's mandatory to use ``sys``:  

.. code-block:: python

    import sys
    sys.path.append('Your/Path/Cucim/Src/')
    from compute_flow import *    

You can use ``imread`` function of  `OpenCV <https://opencv.org/>`_ to read the image sequence:

.. code-block:: python

    import cv2
    Im1 = cv2.imread('im1_path/Im1.extension', 0)
    Im2 = cv2.imread('im2_path/Im2.extension', 0)

.. note:: 

    ``imread`` can be used also to read the mask image.


``compute_flow`` function takes as input many **parameters** as how it was described in :ref:`Compute flow`, it returns :math:`u`, :math:`v` the horizontal and vertical optical flow field respectively.
Before computing the displacements many variable must be adjusted.
 
.. code-block:: python

    u, v = compute_flow(Im1, Im2,pyram_levels, factor, ordre_inter,lmbda, 
            size_median_filter, max_linear_iter,max_iter, lambda2, lambda3,Mask, LO_filter)

