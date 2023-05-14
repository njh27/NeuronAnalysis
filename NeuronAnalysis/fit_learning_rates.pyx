import numpy as np
cimport numpy as np
from libc.math cimport exp

cdef double gaussian(double x, double mu, double sigma, double scale):
    return scale * exp(-( ((x - mu) ** 2) / (2*(sigma**2))) )

def gaussian_activation(np.ndarray[double, ndim=1] x, np.ndarray[double, ndim=1] fixed_means, np.ndarray[double, ndim=1] fixed_sigmas):
    cdef int num_gaussians = fixed_means.shape[0]
    cdef np.ndarray[double, ndim=2] x_transform = np.zeros((x.size, num_gaussians))
    cdef int k
    cdef int i

    for k in range(num_gaussians):
        for i in range(x.size):
            x_transform[i, k] = gaussian(x[i], fixed_means[k], fixed_sigmas[k], scale=1.0)

    return np.asarray(x_transform)
