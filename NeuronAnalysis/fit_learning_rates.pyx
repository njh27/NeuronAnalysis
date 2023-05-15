# eye_input_to_PC_gauss_relu.pyx
import numpy as np
cimport numpy as np
cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[double, ndim=1] f_pf_LTP(np.ndarray[double, ndim=1] pf_LTP_funs, np.ndarray[double, ndim=2] state_input_pf,
                                         np.ndarray[double, ndim=1] W_pf=None, double W_max_pf=None):
    # Check for W_pf and W_max_pf
    if W_pf is not None:
        if ( (W_max_pf is None) or (W_max_pf <= 0) ):
            raise ValueError("If updating weights by inputting values for W_pf, a W_max_pf > 0 must also be specified.")
        W_max_pf = np.full(W_pf.shape, W_max_pf)
    else:
        W_max_pf = np.ones(pf_LTP_funs.shape[0], dtype=np.float64)

    # Convert LTP functions to parallel fiber input space
    cdef np.ndarray[double, ndim=1] pf_LTP = np.dot(pf_LTP_funs, state_input_pf) * (W_max_pf - W_pf).squeeze()

    return pf_LTP

@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[double, ndim=1] f_pf_FR_LTP(np.ndarray[double, ndim=1] PC_FR, double PC_FR_weight_LTP):
    cdef np.ndarray[double, ndim=1] pf_FR_LTP = PC_FR * PC_FR_weight_LTP
    return pf_FR_LTP

@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[double, ndim=1] f_pf_static_LTP(np.ndarray[double, ndim=1] pf_CS_LTD, double static_weight_LTP):
    cdef np.ndarray[double, ndim=1] pf_static_LTP = np.zeros_like(pf_CS_LTD)
    pf_static_LTP[pf_CS_LTD == 0.0] = static_weight_LTP
    return pf_static_LTP

@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[double, ndim=1] f_pf_CS_LTP(np.ndarray[double, ndim=1] CS_trial_bin, int tau_1, int tau_2, double scale=1.0):
    # Inverts the CS function
    # pf_CS_LTP = np.mod(CS_trial_bin + 1, 2)
    cdef np.ndarray[double, ndim=1] pf_CS_LTP = boxcar_convolve(CS_trial_bin, tau_1, tau_2)
    pf_CS_LTP[pf_CS_LTP > 0] = scale
    return pf_CS_LTP

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
cdef np.ndarray[double, ndim=2] eye_input_to_PC_gauss_relu(np.ndarray[double, ndim=2] eye_data,
                                                            np.ndarray[double, ndim=1] gauss_means,
                                                            np.ndarray[double, ndim=1] gauss_stds,
                                                            np.ndarray[int, ndim=1] n_gaussians_per_dim=None):
    cdef int n_eye_dims = 4
    cdef int n_eye_lags = 2
    cdef int n_total_eye_dims = n_eye_dims * n_eye_lags
    cdef int n_features
    cdef int first_relu_ind
    cdef int dim_start = 0
    cdef int dim_stop = 0
    cdef np.ndarray[double, ndim=2] eye_transform

    if len(gauss_means) != len(gauss_stds):
        raise ValueError("Must input the same number of means and standard deviations but got {0} means and {1} standard deviations.".format(len(gauss_means), len(gauss_stds)))

    n_features = len(gauss_means) + 8 # Total input featur to PC is gaussians + relus
    first_relu_ind = len(gauss_means)

    # Transform data into "input" n_gaussians dimensional format
    eye_transform = np.zeros((eye_data.shape[0], n_features))

    for k in range(0, n_eye_dims):
        dim_stop += n_gaussians_per_dim[k]
        # First do Gaussian activation on first 4 eye dims
        eye_transform[:, dim_start:dim_stop] = gaussian_activation(eye_data[:, k], gauss_means[dim_start:dim_stop], gauss_stds[dim_start:dim_stop])
        dim_start = dim_stop
        # Then relu activation on second 4 eye dims
        eye_transform[:, (first_relu_ind + 2 * k)] = negative_relu(eye_data[:, n_eye_dims + k], c=0.0)
        eye_transform[:, (first_relu_ind + (2 * k + 1))] = reflected_negative_relu(eye_data[:, n_eye_dims + k], c=0.0)

    return eye_transform


@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[double, ndim=1] learning_function(np.ndarray[double, ndim=1] params,
                                                  np.ndarray[double, ndim=2] x,
                                                  np.ndarray[double, ndim=1] y,
                                                  np.ndarray[double, ndim=1] W_0_pf,
                                                  np.ndarray[double, ndim=1] W_0_mli,
                                                  double b, double bin_width,
                                                  int n_trials, int n_obs_pt,
                                                  np.ndarray[int, ndim=1] is_missing_data,
                                                  np.ndarray[int, ndim=1] n_gaussians_per_dim,
                                                  np.ndarray[double, ndim=1] gauss_means,
                                                  np.ndarray[double, ndim=1] gauss_stds,
                                                  int n_gaussians, double W_min_pf,
                                                  double FR_MAX, double tau_rise_CS,
                                                  double tau_decay_CS, double tau_rise_CS_LTP,
                                                  double tau_decay_CS_LTP):

    """
    """
    cdef np.ndarray[double, ndim=2] state = x[:, 0:-1]
    cdef np.ndarray[double, ndim=1] CS = x[:, -1]
    cdef np.ndarray[double, ndim=1] y_hat = np.zeros(x.shape[0])
    cdef double alpha = params[0] / 1e4
    cdef double beta = params[1] / 1e4
    cdef double gamma = params[2] / 1e4
    cdef double epsilon = params[3] / 1e4
    cdef double W_max_pf = params[4]
    cdef np.ndarray[double, ndim=1] W_pf = np.copy(W_0_pf)
    cdef np.ndarray[double, ndim=1] W_mli = np.copy(W_0_mli)
    cdef np.ndarray[double, ndim=1] W_full = np.vstack((W_pf, W_mli))

    # Ensure W_pf values are within range and store in output W_full
    W_pf[(W_pf > W_max_pf).squeeze()] = W_max_pf
    W_pf[(W_pf < W_min_pf).squeeze()] = W_min_pf
    W_full[0:n_gaussians] = W_pf
    cdef np.ndarray[double, ndim=1] residuals = np.empty(n_trials, dtype=np.float64)
    cdef np.ndarray[double, ndim=1] y_hat = np.empty(n_trials, dtype=np.float64)
    cdef np.ndarray[double, ndim=1] y_hat_trial
    cdef np.ndarray[double, ndim=1] state_trial
    cdef np.ndarray[double, ndim=1] y_obs_trial
    cdef np.ndarray[double, ndim=1] is_missing_data_trial
    cdef np.ndarray[double, ndim=1] CS_trial_bin
    cdef np.ndarray[double, ndim=1] state_input
    cdef np.ndarray[double, ndim=1] state_input_pf
    cdef np.ndarray[double, ndim=1] pf_CS_LTD
    cdef np.ndarray[double, ndim=1] pf_LTD
    cdef np.ndarray[double, ndim=1] pf_LTP_funs
    cdef np.ndarray[double, ndim=1] pf_LTP
    cdef int trial

    for trial in range(n_trials):
        state_trial = state[trial*n_obs_pt:(trial + 1)*n_obs_pt, :]
        y_obs_trial = y[trial*n_obs_pt:(trial + 1)*n_obs_pt]
        is_missing_data_trial = is_missing_data[trial*n_obs_pt:(trial + 1)*n_obs_pt]

        # Convert state to input layer activations
        state_input = eye_input_to_PC_gauss_relu(state_trial, gauss_means, gauss_stds, n_gaussians_per_dim)
        state_input[is_missing_data_trial, :] = 0.0
        y_hat_trial = np.maximum(0, np.dot(state_input, W_full) + b)
        y_hat[trial*n_obs_pt:(trial + 1)*n_obs_pt] = y_hat_trial

        state_input_pf = state_input[:, 0:n_gaussians]
        y_obs_trial /= FR_MAX
        CS_trial_bin = CS[trial*n_obs_pt:(trial + 1)*n_obs_pt]

        pf_CS_LTD = f_pf_CS_LTD(CS_trial_bin, tau_rise_CS, tau_decay_CS, epsilon, 0.0)
        pf_LTD = f_pf_LTD(pf_CS_LTD, state_input_pf, W_pf=W_pf, W_min_pf=W_min_pf)

        pf_LTP_funs = f_pf_CS_LTP(CS_trial_bin, tau_rise_CS_LTP, tau_decay_CS_LTP, alpha)
        pf_LTP_funs += f_pf_FR_LTP(y_obs_trial, beta)
        pf_LTP_funs += f_pf_static_LTP(pf_CS_LTD, gamma)
        pf_LTP_funs[pf_CS_LTD > 0.0] = 0.0
        pf_LTP = f_pf_LTP(pf_LTP_funs, state_input_pf, W_pf=W_pf, W_max_pf=W_max_pf)
        W_pf += ( pf_LTP[:, None] + pf_LTD[:, None] )

        W_pf[(W_pf > W_max_pf).squeeze()] = W_max_pf
        W_pf[(W_pf < W_min_pf).squeeze()] = W_min_pf
        W_full[0:n_gaussians] = W_pf

    residuals = (y - y_hat) ** 2
    return residuals
