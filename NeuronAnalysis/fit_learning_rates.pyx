import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport exp, sqrt



# Python wrapper function for calling objective function
def py_learning_function(params, x, y, W_0_pf, W_0_mli, b,
                         n_trials, n_obs_pt,
                         n_gaussians_per_dim, gauss_means, gauss_stds,
                         n_gaussians, W_min_pf, FR_MAX, tau_rise_CS,
                         tau_decay_CS, tau_rise_CS_LTP, tau_decay_CS_LTP):
    residuals = learning_function(params, x, y, W_0_pf, W_0_mli, b,
                             n_trials, n_obs_pt,
                             n_gaussians_per_dim, gauss_means, gauss_stds,
                             n_gaussians, W_min_pf, FR_MAX, tau_rise_CS,
                             tau_decay_CS, tau_rise_CS_LTP, tau_decay_CS_LTP)
    return residuals

cdef double d_max(double x, double y) nogil:
    return x if (x > y) else y

cdef double d_min(double x, double y) nogil:
    return x if (x < y) else y

cdef Py_ssize_t py_max(Py_ssize_t x, Py_ssize_t y) nogil:
    return x if (x > y) else y

cdef Py_ssize_t py_min(Py_ssize_t x, Py_ssize_t y) nogil:
    return x if (x < y) else y

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void f_pf_LTP(np.ndarray[double, ndim=1] pf_LTP,
                   np.ndarray[double, ndim=1] pf_LTP_funs,
                   double[:, :] state_input_pf,
                   np.ndarray[double, ndim=1] W_pf, double W_max_pf):
    cdef Py_ssize_t wi
    # Convert LTP functions to parallel fiber input space
    np.dot(pf_LTP_funs, state_input_pf, out=pf_LTP)
    for wi in range(W_pf.shape[0]):
        pf_LTP[wi] *= (W_max_pf - W_pf[wi])
    return

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void f_pf_FR_LTP(double[::1] pf_LTP_funs, double[::1] PC_FR,
                      double PC_FR_weight_LTP, double FR_MAX) nogil:
    cdef Py_ssize_t t
    for t in range(pf_LTP_funs.shape[0]):
        pf_LTP_funs[t] += ((PC_FR[t] / FR_MAX) * PC_FR_weight_LTP)
    return

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void f_pf_static_LTP(double[::1] pf_LTP_funs, double static_weight_LTP) nogil:
    cdef Py_ssize_t t
    for t in range(pf_LTP_funs.shape[0]):
        pf_LTP_funs[t] += static_weight_LTP
    return

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void box_windows(double[::1] window_sig, double[::1] spike_train,
                      int box_pre, int box_post, double scale=1.0) nogil:
    cdef Py_ssize_t t, w_t, w_start, w_stop
    # Reset window_sig to 0
    for w_t in range(window_sig.shape[0]):
        window_sig[w_t] = 0.0

    for t in range(spike_train.shape[0]):
        if spike_train[t] > 0.0:
            w_start = py_max(0, t-box_pre)
            w_stop = py_min(window_sig.shape[0], t+box_post+1)
            for w_t in range(w_start, w_stop):
                window_sig[w_t] = scale
    return

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void f_pf_CS_LTP(double[::1] pf_LTP_funs,
                      double[::1] CS_trial_bin,
                      int tau_1, int tau_2, double scale=1.0):
    box_windows(pf_LTP_funs, CS_trial_bin, tau_1, tau_2, scale)
    return

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void f_pf_LTD(np.ndarray[double, ndim=1] pf_LTD,
                   np.ndarray[double, ndim=1] pf_CS_LTD,
                   double[:, :] state_input_pf,
                   np.ndarray[double, ndim=1] W_pf, double W_min_pf):
    cdef Py_ssize_t wi
    # Sum of pf_CS_LTD weighted by activation for each input unit
    pf_LTD = np.dot(pf_CS_LTD, state_input_pf, out=pf_LTD)
    # Set state modification scaling according to current weight
    # Will all be negative values
    for wi in range(W_pf.shape[0]):
        pf_LTD[wi] *= (W_min_pf - W_pf[wi])
    return

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void f_pf_CS_LTD(double[::1] pf_CS_LTD,
                      double[::1] CS_trial_bin,
                      int tau_1, int tau_2, double scale=1.0):
    # Just CS window plasticity
    box_windows(pf_CS_LTD, CS_trial_bin, tau_1, tau_2, scale)
    return

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void gaussian_activation(double[:] x,
                              double[:] fixed_means, double[:] fixed_sigmas,
                              double[:, :] x_transform) nogil:
    cdef Py_ssize_t k, t
    for k in range(fixed_means.shape[0]):
        for t in range(x.shape[0]):
            x_transform[t, k] = exp(-( ((x[t] - fixed_means[k]) ** 2) / (2*(fixed_sigmas[k]**2))) )
    return

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double negative_relu(double x, double c=0.) nogil:
    """ Basic relu function but returns negative result. """
    return -1 * d_max(0., x-c)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double reflected_negative_relu(double x, double c=0.) nogil:
    """ Basic relu function but returns negative result, reflected about y axis. """
    return d_min(0., x-c)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void eye_input_to_PC_gauss_relu(double[:, :] eye_data,
                                     np.ndarray[double, ndim=1] gauss_means,
                                     np.ndarray[double, ndim=1] gauss_stds,
                                     np.ndarray[double, ndim=2] eye_transform,
                                     np.ndarray[int, ndim=1] n_gaussians_per_dim):
    cdef int n_eye_dims = 4
    cdef int first_relu_ind
    cdef int dim_start = 0
    cdef int dim_stop = 0
    cdef int k, l

    if gauss_means.shape[0] != gauss_stds.shape[0]:
        raise ValueError("Must input the same number of means and standard deviations but got {0} means and {1} standard deviations.".format(gauss_means.shape[0], len(gauss_stds)))

    first_relu_ind = gauss_means.shape[0]
    for k in range(0, n_eye_dims):
        dim_stop += n_gaussians_per_dim[k]
        # First do Gaussian activation on first 4 eye dims
        gaussian_activation(eye_data[:, k], gauss_means[dim_start:dim_stop],
                              gauss_stds[dim_start:dim_stop],
                              eye_transform[:, dim_start:dim_stop])
        dim_start = dim_stop
        # Then relu activation on second 4 eye dims
        for l in range(0, eye_data.shape[0]):
            eye_transform[l, (first_relu_ind + 2 * k)] = negative_relu(eye_data[l, n_eye_dims + k], c=0.0)
            eye_transform[l, (first_relu_ind + (2 * k + 1))] = reflected_negative_relu(eye_data[l, n_eye_dims + k], c=0.0)
    return

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void update_W_pf(np.ndarray[double, ndim=1] W_pf,
                      np.ndarray[double, ndim=1] pf_LTP,
                      np.ndarray[double, ndim=1] pf_LTD, W_max_pf, W_min_pf):
    cdef int wi
    for wi in range(0, W_pf.shape[0]):
        W_pf[wi] += (pf_LTP[wi] + pf_LTD[wi])
        if W_pf[wi] > W_max_pf:
            W_pf[wi] = W_max_pf
        if W_pf[wi] < W_min_pf:
            W_pf[wi] = W_min_pf
    return

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double learning_function(np.ndarray[double, ndim=1] params,
                                                  np.ndarray[double, ndim=2] x,
                                                  np.ndarray[double, ndim=1] y,
                                                  np.ndarray[double, ndim=1] W_0_pf,
                                                  np.ndarray[double, ndim=1] W_0_mli,
                                                  double b, int n_trials, int n_obs_pt,
                                                  np.ndarray[int, ndim=1] n_gaussians_per_dim,
                                                  np.ndarray[double, ndim=1] gauss_means,
                                                  np.ndarray[double, ndim=1] gauss_stds,
                                                  int n_gaussians, double W_min_pf,
                                                  double FR_MAX, int tau_rise_CS,
                                                  int tau_decay_CS, int tau_rise_CS_LTP,
                                                  int tau_decay_CS_LTP):

    cdef double[:, :] state = x[:, 0:-1]
    cdef double[::1] CS = np.ascontiguousarray(x[:, -1])
    cdef np.ndarray[double, ndim=1] W_pf = np.copy(W_0_pf)

    cdef double residuals = 0.0
    cdef double[:] y_hat_trial
    cdef double[:, :] state_trial
    cdef double[::1] y_obs_trial
    cdef double[::1] CS_trial_bin
    cdef double[:, :] state_input_pf
    cdef np.ndarray[double, ndim=1] W_full = np.zeros((n_gaussians + 8, ))
    cdef np.ndarray[double, ndim=2] state_input = np.zeros((n_obs_pt, n_gaussians + 8))
    cdef np.ndarray[double, ndim=1] pf_CS_LTD = np.zeros((n_obs_pt, ))
    cdef np.ndarray[double, ndim=1] pf_LTD = np.zeros((n_gaussians, ))
    cdef np.ndarray[double, ndim=1] pf_LTP_funs = np.zeros((n_obs_pt, ))
    cdef np.ndarray[double, ndim=1] pf_LTP = np.zeros((n_gaussians, ))
    cdef int trial, wi, t_i

    # REMINDER of param definitions
    cdef double alpha = params[0] / 1e4
    cdef double beta = params[1] / 1e4
    cdef double gamma = params[2] / 1e4
    cdef double epsilon = params[3] / 1e4
    cdef double W_max_pf = params[4]

    # Ensure W_pf values are within range and store in output W_full
    for wi in range(0, W_pf.shape[0]):
        if W_pf[wi] > W_max_pf:
            W_pf[wi] = W_max_pf
        if W_pf[wi] < W_min_pf:
            W_pf[wi] = W_min_pf
        W_full[wi] = W_pf[wi]
    for wi in range(0, 8):
        W_full[n_gaussians + wi] = W_0_mli[wi]

    for trial in range(n_trials):
        state_trial = state[trial*n_obs_pt:(trial + 1)*n_obs_pt, :]
        y_obs_trial = y[trial*n_obs_pt:(trial + 1)*n_obs_pt]

        # Convert state to input layer activations
        # Modifies "state_input" IN PLACE
        eye_input_to_PC_gauss_relu(state_trial, gauss_means, gauss_stds, state_input, n_gaussians_per_dim)
        y_hat_trial = np.maximum(0, np.dot(state_input, W_full) + b)
        for t_i in range(0, n_obs_pt):
          residuals += sqrt((y_obs_trial[t_i] - y_hat_trial[t_i]) ** 2)

        state_input_pf = state_input[:, 0:n_gaussians]
        CS_trial_bin = CS[trial*n_obs_pt:(trial + 1)*n_obs_pt]

        # Call to box_windows inside here resets pf_CS_LTD to zeros!
        f_pf_CS_LTD(pf_CS_LTD[:], CS_trial_bin, tau_rise_CS, tau_decay_CS, epsilon)
        f_pf_LTD(pf_LTD, pf_CS_LTD[:], state_input_pf, W_pf, W_min_pf)

        # Call to box_windows inside here resets pf_LTP_funs to zeros!
        f_pf_CS_LTP(pf_LTP_funs, CS_trial_bin, tau_rise_CS_LTP, tau_decay_CS_LTP, alpha)
        f_pf_FR_LTP(pf_LTP_funs, y_obs_trial, beta, FR_MAX)
        f_pf_static_LTP(pf_LTP_funs, gamma)
        f_pf_LTP(pf_LTP, pf_LTP_funs, state_input_pf, W_pf, W_max_pf)

        # Updates weights of W_pf in place
        update_W_pf(W_pf, pf_LTP, pf_LTD, W_max_pf, W_min_pf)
        # Put updated weights into W_full for next iteration
        for wi in range(0, n_gaussians):
            W_full[wi] = W_pf[wi]

    return residuals
