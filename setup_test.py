import setuptools


setuptools.setup(name='Optical flow GPU/CPU',
      description='Optical flow algorithm ',
      author='Ahmed CHABIB',
      author_email='ahmed.chabib@univ-lille.fr',
      url='https://github.com/chabibchabib',
      packages=['compute_flow', 'flow_operator'],
      classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
      )
