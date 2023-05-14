# eye_input_to_PC_gauss_relu.pyx
import numpy as np
cimport numpy as np
cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[double, ndim=1] f_pf_LTD(np.ndarray[double, ndim=1] pf_CS_LTD, np.ndarray[double, ndim=2] state_input_pf, double W_pf=None, double W_min_pf=0.0):
    # Sum of pf_CS_LTD weighted by activation for each input unit
    cdef np.ndarray[double, ndim=1] pf_LTD = np.dot(pf_CS_LTD, state_input_pf)
    # Set state modification scaling according to current weight
    if W_pf is not None:
        W_min_pf = np.full(W_pf.shape, W_min_pf)
        pf_LTD *= (W_min_pf - W_pf).squeeze() # Will all be negative values
    else:
        pf_LTD *= -1.0
    return pf_LTD

@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[double, ndim=1] f_pf_CS_LTD(np.ndarray[double, ndim=1] CS_trial_bin, int tau_1, int tau_2, double scale=1.0, int delay=0):
    # Just CS point plasticity
    cdef np.ndarray[double, ndim=1] pf_CS_LTD = boxcar_convolve(CS_trial_bin, tau_1, tau_2)
    # Rescale to binary scale
    pf_CS_LTD[pf_CS_LTD > 0] = scale
    # Shift pf_CS_LTD LTD envelope according to delay_LTD
    delay = int(delay)
    if delay == 0:
        # No delay so we are done
        return pf_CS_LTD
    elif delay < 0:
        pf_CS_LTD[-delay:] = pf_CS_LTD[0:delay]
        pf_CS_LTD[0:-delay] = 0.0
    else:
        # Implies delay > 0
        pf_CS_LTD[0:-delay] = pf_CS_LTD[delay:]
        pf_CS_LTD[-delay:] = 0.0
    return pf_CS_LTD

@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[double, ndim=1] boxcar_convolve(np.ndarray[double, ndim=1] spike_train, int box_pre, int box_post, double scale=1.0):
    cdef int center_ind = max(abs(box_pre), abs(box_post)) + 1 # plus 1 for cushion on ends
    cdef np.ndarray[double, ndim=1] kernel = np.zeros(2*center_ind + 1)
    kernel[center_ind-box_pre:center_ind+box_post+1] = scale
    cdef np.ndarray[double, ndim=1] filtered_sig = np.convolve(spike_train, kernel, mode='same')
    return filtered_sig

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double gaussian(double x, double mu, double sigma, double scale):
    return scale * np.exp(-( ((x - mu) ** 2) / (2*(sigma**2))) )

@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[double, ndim=2] gaussian_activation(np.ndarray[double, ndim=1] x, np.ndarray[double, ndim=1] fixed_means, np.ndarray[double, ndim=1] fixed_sigmas):
    cdef int num_gaussians = len(fixed_means)
    cdef np.ndarray[double, ndim=2] x_transform = np.zeros((x.size, num_gaussians))
    for k in range(num_gaussians):
        x_transform[:, k] = gaussian(x, fixed_means[k], fixed_sigmas[k], scale=1.0)
    return x_transform

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double negative_relu(double x, double c=0.):
    """ Basic relu function but returns negative result. """
    return -1 * max(0., x-c)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double reflected_negative_relu(double x, double c=0.):
    """ Basic relu function but returns negative result, reflected about y axis. """
    return min(0., x-c)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[double, ndim=2] eye_input_to_PC_gauss_relu(np.ndarray[double, ndim=2] eye_data, np.ndarray[double, ndim=1] gauss_means, np.ndarray[double, ndim=1] gauss_stds, np.ndarray[int, ndim=1] n_gaussians_per_dim=None):
    cdef int n_eye_dims = 4
    cdef int n_eye_lags = 2
    cdef int n_total_eye_dims = n_eye_dims * n_eye_lags
    cdef int n_features, first_relu_ind, dim_start = 0, dim_stop = 0
    cdef np.ndarray[double, ndim=2] eye_transform
    # ... the rest of the function implementation ...
