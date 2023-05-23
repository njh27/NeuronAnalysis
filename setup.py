from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np


ext  =  [Extension( "fit_learning_rates", sources=['NeuronAnalysis/fit_learning_rates.pyx'] )]
setup(name='NeuronAnalysis',
      version='1.0',
      description='Basic timeseries organization of neural data.',
      author='Nathan Hall',
      author_email='nathan.hall@duke.edu',
      url='https://',
      packages=['NeuronAnalysis'],
      ext_modules = cythonize(ext, annotate=False, language_level=3),
      include_dirs=[np.get_include()],
     )
