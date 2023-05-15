from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

ext  =  [Extension( "fit_learning_rates", sources=['fit_learning_rates.pyx'] )]
# You must navigate into this directory, not from a path, in the desired Anaconda ENVIRONMENT
# compile as: $ python setup.py build_ext --inplace
# compile as: > python.exe setup.py build_ext --inplace
setup(
    name='Fit Learning Rates',
    ext_modules = cythonize(ext, annotate=False, language_level=3),
    include_dirs = [np.get_include()]
)
