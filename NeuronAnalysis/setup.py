from setuptools import setup
from Cython.Build import cythonize

setup(
    name='Fit Learning Rates',
    ext_modules=cythonize("fit_learning_rates.pyx"),
)
