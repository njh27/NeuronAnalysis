import numpy as np
from scipy.optimize import least_squares
from NeuronAnalysis.general import box_windows
import NeuronAnalysis.activation_functions as af
from NeuronAnalysis.fit_NN_model import bin_data
from NeuronAnalysis.fit_learning_rates import py_learning_function



""" SOME FUNCTIONS FOR GETTING DATA TO PREDICT FIRING BASED ON PLASTIC WEIGHTS """
""" ********************************************************************** """
def comp_learning_response(NN_FIT, X_trial, W_trial, return_comp=False):
    """
    """
    if X_trial.shape[2] != 8:
        raise ValueError("Gaussian basis kinematics model is fit for 8 data dimensions but input data dimension is {0}.".format(X.shape[1]))

    pos_means = NN_FIT.fit_results['gauss_basis_kinematics']['pos_means']
    vel_means = NN_FIT.fit_results['gauss_basis_kinematics']['vel_means']
    n_gaussians_per_dim = [len(pos_means), len(pos_means),
                           len(vel_means), len(vel_means)]
    gauss_means = np.hstack([pos_means,
                             pos_means,
                             vel_means,
                             vel_means])
    pos_stds = NN_FIT.fit_results['gauss_basis_kinematics']['pos_stds']
    vel_stds = NN_FIT.fit_results['gauss_basis_kinematics']['vel_stds']
    gauss_stds = np.hstack([pos_stds,
                            pos_stds,
                            vel_stds,
                            vel_stds])

    n_gaussians = len(gauss_means)
    y_hat = np.zeros((X_trial.shape[0], X_trial.shape[1]))
    W = np.copy(NN_FIT.fit_results['gauss_basis_kinematics']['coeffs'])
    b = NN_FIT.fit_results['gauss_basis_kinematics']['bias']
    pf_in = np.zeros((X_trial.shape[0], X_trial.shape[1]))
    mli_in = np.zeros((X_trial.shape[0], X_trial.shape[1]))
    for t_ind in range(0, X_trial.shape[0]):
        # Transform X_data for this trial into input space
        X_input = af.eye_input_to_PC_gauss_relu(X_trial[t_ind, :, :],
                                        gauss_means, gauss_stds,
                                        n_gaussians_per_dim=n_gaussians_per_dim)
        # Each trial update the weights for W
        W[:, 0] = W_trial[t_ind, :]
        y_hat[t_ind, :] = (np.dot(X_input, W) + b).squeeze()
        pf_in[t_ind, :] = (np.dot(X_input[:, 0:n_gaussians], W[0:n_gaussians, 0]) + b).squeeze()
        mli_in[t_ind, :] = (np.dot(X_input[:, n_gaussians:], W[n_gaussians:, 0]) + b).squeeze()
        if NN_FIT.activation_out == "relu":
            y_hat[t_ind, :] = np.maximum(0., y_hat[t_ind, :])
            pf_in[t_ind, :] = np.maximum(0., pf_in[t_ind, :])
            mli_in[t_ind, :] = np.maximum(0., mli_in[t_ind, :])

    if return_comp:
        return y_hat, pf_in, mli_in
    else:
        return y_hat


def predict_learning_response_by_trial(NN_FIT, blocks, trial_sets, weights_by_trial,
                                        return_comp=False, test_data_only=False,
                                        verbose=False):
    """
    """
    X, init_shape, t_inds = NN_FIT.get_gauss_basis_kinematics_predict_data_trial(
                            blocks, trial_sets, return_shape=True,
                            test_data_only=test_data_only, return_inds=True,
                            verbose=verbose)
    X_trial = X.reshape(init_shape)
    # Get weights in a single matrix to pass through here
    W_trial = np.zeros((X_trial.shape[0], weights_by_trial[t_inds[0]].shape[0]))
    # Go through t_nums IN ORDER
    for t_i, t in enumerate(t_inds):
        try:
            W_trial[t_i, :] = weights_by_trial[t].squeeze()
        except KeyError:
            raise ValueError("weights by trial does not contain weights for requested trial number {0}.".format(t))
    if return_comp:
        y_hat, pf_in, mli_in = comp_learning_response(NN_FIT, X_trial, W_trial,
                                        return_comp=return_comp)
        return y_hat, pf_in, mli_in
    else:
        y_hat = comp_learning_response(NN_FIT, X_trial, W_trial)
        return y_hat


""" SOME HELPERS FOR GETTING THE EYE DATA TO FIT FOR PLASTIC WEIGHTS """
""" *********************************************************************** """
def get_eye_data_traces_win(NN_FIT, blocks, trial_sets, time_window, lag=0,
                            return_inds=False):
    """ Gets eye position and velocity in array of trial x time_window
        3rd dimension of array is ordered as pursuit, learning position,
        then pursuit, learning velocity.
        Data are only retrieved for valid neuron trials!
    """
    lag_time_window = time_window + np.int32(lag)
    if lag_time_window[1] <= lag_time_window[0]:
        raise ValueError("time_window[1] must be greater than time_window[0]")

    trial_sets = NN_FIT.neuron.append_valid_trial_set(trial_sets)
    pos_p, pos_l, t_inds = NN_FIT.neuron.session.get_xy_traces("eye position",
                            lag_time_window, blocks, trial_sets,
                            return_inds=True)
    vel_p, vel_l = NN_FIT.neuron.session.get_xy_traces("eye velocity",
                            lag_time_window, blocks, trial_sets,
                            return_inds=False)
    eye_data = np.stack((pos_p, pos_l, vel_p, vel_l), axis=2)
    if return_inds:
        return eye_data, t_inds
    else:
        return eye_data


def get_plasticity_data_trial_win(NN_FIT, blocks, trial_sets, time_window,
                                    return_shape=False, return_inds=False):
    """ Gets behavioral data from blocks and trial sets and formats in a
    way that it can be used to predict firing rate according to the linear
    eye kinematic model using predict_lin_eye_kinematics.
    Data are only retrieved for trials that are valid for the fitted neuron. """
    trial_sets = NN_FIT.neuron.append_valid_trial_set(trial_sets)
    eye_data_pf, t_inds = get_eye_data_traces_win(NN_FIT, blocks, trial_sets,
                            time_window,
                            NN_FIT.fit_results['gauss_basis_kinematics']['pf_lag'],
                            return_inds=True)
    eye_data_mli = get_eye_data_traces_win(NN_FIT, blocks, trial_sets,
                            time_window,
                            NN_FIT.fit_results['gauss_basis_kinematics']['mli_lag'])
    eye_data = np.concatenate((eye_data_pf, eye_data_mli), axis=2)
    initial_shape = eye_data.shape
    eye_data = eye_data.reshape(eye_data.shape[0]*eye_data.shape[1], eye_data.shape[2], order='C')
    if return_shape and return_inds:
        return eye_data, initial_shape, t_inds
    elif return_shape and not return_inds:
        return eye_data, initial_shape
    elif not return_shape and return_inds:
        return eye_data, t_inds
    else:
        return eye_data


""" THESE ARE THE LEARNING RULE PLASTICITY FUNCTIONS """
""" *********************************************************************** """

def f_pf_CS_LTD(CS_trial_bin, tau_1, tau_2, scale=1.0, delay=0):
    """ Computes the parallel fiber LTD as a function of time of the complex
    spike input f_CS with a kernel scaled from tau_1 to tau_2 with peak equal to
    scale and with CSs shifted by an amoutn of time "delay" INDICES (not time!). """
    # Just CS point plasticity
    pf_CS_LTD = box_windows(CS_trial_bin, tau_1, tau_2, scale=scale)
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

def f_pf_LTD(pf_CS_LTD, state_input_pf, W_pf=None, W_min_pf=0.0):
    """ Updates the parallel fiber LTD function of time "pf_CS_LTD" to be scaled
    by PC firing rate if input, then scales by the pf state input and finally
    weights by the current weight function. pf_CS_LTD is MODIFIED IN PLACE!
    The output contains NEGATIVE values in pf_LTD. """
    # Sum of pf_CS_LTD weighted by activation for each input unit
    pf_LTD = np.dot(pf_CS_LTD, state_input_pf)
    # Set state modification scaling according to current weight
    if W_pf is not None:
        W_min_pf = np.full(W_pf.shape, W_min_pf)
        pf_LTD *= (W_min_pf - W_pf).squeeze() # Will all be negative values
    else:
        pf_LTD *= -1.0
    return pf_LTD

def f_pf_CS_LTP(CS_trial_bin, tau_1, tau_2, scale=1.0):
    """ Assumes CS_trial_bin is an array of 0's and 1's where 1's indicate that
    CS LTD is taking place and 0's indicate no CS related LTD. This function
    will then invert the CS train and form windows of LTP where LTD is absent.
    """
    # Inverts the CS function
    # pf_CS_LTP = np.mod(CS_trial_bin + 1, 2)
    pf_CS_LTP = box_windows(CS_trial_bin, tau_1, tau_2, scale=scale)
    return pf_CS_LTP

def f_pf_static_LTP(pf_CS_LTD, static_weight_LTP):
    """ Inverts the input pf_CS_LTD fun so that it is opposite.
    """
    # Inverts the CS function
    pf_static_LTP = np.zeros_like(pf_CS_LTD)
    pf_static_LTP[pf_CS_LTD == 0.0] = static_weight_LTP
    return pf_static_LTP

def f_pf_FR_LTP(PC_FR, PC_FR_weight_LTP):
    """
    """
    # Add a term with firing rate times weight of constant LTP
    pf_FR_LTP = PC_FR * PC_FR_weight_LTP
    return np.zeros_like(pf_FR_LTP)

def f_pf_LTP(pf_LTP_funs, state_input_pf, W_pf=None, W_max_pf=None):
    """ Updates the parallel fiber LTP function of time "pf_CS_LTP" to be scaled
    by PC firing rate if input, then scales by the pf state input and finally
    weights by the current weight function. pf_LTP is MODIFIED IN PLACE!
    W_max_pf should probably be input as FITTED PARAMETER! variable and is
    critical if using weight updates. Same for PC_FR_weight_LTP.
    """
    # Convert LTP functions to parallel fiber input space
    pf_LTP = np.dot(pf_LTP_funs, state_input_pf)
    if W_pf is not None:
        if ( (W_max_pf is None) or (W_max_pf <= 0) ):
            raise ValueError("If updating weights by inputting values for W_pf, a W_max_pf > 0 must also be specified.")
        W_max_pf = np.full(W_pf.shape, W_max_pf)
        pf_LTP *= (W_max_pf - W_pf).squeeze()
    return pf_LTP

def f_mli_CS_LTP(CS_trial_bin, tau_1, tau_2, scale=1.0, delay=0):
    """ Computes the MLI LTP as a function of time of the complex
    spike input f_CS with a kernel scaled from tau_1 to tau_2 with peak equal to
    scale and with CSs shifted by an amoutn of time "delay" INDICES (not time!). """
    # Just CS point plasticity
    mli_CS_LTP = box_windows(CS_trial_bin, tau_1, tau_2, scale=scale)
    # Shift mli_CS_LTP LTP envelope according to delay
    delay = int(delay)
    if delay == 0:
        # No delay so we are done
        return mli_CS_LTP
    elif delay < 0:
        mli_CS_LTP[-delay:] = mli_CS_LTP[0:delay]
        mli_CS_LTP[0:-delay] = 0.0
    else:
        # Implies delay > 0
        mli_CS_LTP[0:-delay] = mli_CS_LTP[delay:]
        mli_CS_LTP[-delay:] = 0.0
    return mli_CS_LTP

def f_mli_LTP(mli_CS_LTP, state_input_mli, W_mli=None, W_max_mli=None):
    """ Updates the MLI LTP function of time "mli_CS_LTP" to be scaled
    by weights by the current weight function. mli_CS_LTP is MODIFIED IN PLACE!"""
    # Sum of mli_CS_LTP weighted by activation for each input unit
    mli_LTP = np.dot(mli_CS_LTP, state_input_mli)
    # Set state modification scaling according to current weight
    if W_mli is not None:
        if ( (W_max_mli is None) or (W_max_mli <= 0) ):
            raise ValueError("If updating weights by inputting values for W_mli, a W_max_mli > 0 must also be specified.")
        W_max_mli = np.full(W_mli.shape, W_max_mli)
        mli_LTP *= (W_max_mli - W_mli).squeeze()
    return mli_LTP

def f_mli_CS_LTD(CS_trial_bin, tau_1, tau_2, scale=1.0):
    """ Assumes CS_trial_bin is an array of 0's and 1's where 1's indicate that
    CS is taking place and 0's indicate no CS related LTD. This function
    will then invert the CS train and form windows of LTD where CS is absent.
    """
    mli_CS_LTD = box_windows(CS_trial_bin, tau_1, tau_2, scale=scale)
    return mli_CS_LTD

def f_mli_FR_LTD(PC_FR, PC_FR_weight_LTD_mli):
    """
    """
    # inv_PC_FR = 1 - PC_FR
    # inv_PC_FR[inv_PC_FR < 0.0] = 0.0
    # mli_FR_LTD = inv_PC_FR * PC_FR_weight_LTD_mli
    # Add a term with firing rate times weight of constant LTD
    mli_FR_LTD = PC_FR * PC_FR_weight_LTD_mli
    return mli_FR_LTD

def f_mli_pf_LTD(state_input_pf, W_pf, PC_FR_weight_LTD_mli):
    """
    """
    mli_FR_LTD = np.dot(state_input_pf, W_pf) * PC_FR_weight_LTD_mli
    return mli_FR_LTD

def f_mli_static_LTD(mli_CS_LTP, static_weight_mli_LTD):
    """ Inverts the input pf_CS_LTD fun so that it is opposite.
    """
    # Inverts the CS function
    mli_static_LTD = np.zeros_like(mli_CS_LTP)
    mli_static_LTD[mli_CS_LTP == 0.0] = static_weight_mli_LTD
    return mli_static_LTD

def f_mli_LTD(mli_LTD_funs, state_input_mli, W_mli=None, W_min_mli=0.0):
    """ Updates the parallel fiber LTP function of time "mli_CS_LTD" to be scaled
    by PC firing rate if input, then scales by the pf state input and finally
    weights by the current weight function. pf_LTP is MODIFIED IN PLACE!
    W_max_pf should probably be input as FITTED PARAMETER! variable and is
    critical if using weight updates. Same for PC_FR_weight_LTP.
    """
    # Convert LTD functions to MLI input space
    mli_LTD = np.dot(mli_LTD_funs, state_input_mli)
    if W_mli is not None:
        W_min_mli = np.full(W_mli.shape, W_min_mli)
        mli_LTD *= (W_min_mli - W_mli).squeeze() # Will all be negative values
    else:
        mli_LTD *= -1.0
    return mli_LTD

""" *********************************************************************** """

# def learning_function(params, x, y, W_0_pf, W_0_mli, b, *args, **kwargs):
#     """ Defines the learning model we are fitting to the data """
#     # Separate behavior state from CS inputs
#     state = x[:, 0:-1]
#     CS = x[:, -1]
#     y_hat = np.zeros(x.shape[0])
#     # Extract other precomputed necessary args
#     bin_width = args[0]
#     n_trials = args[1]
#     n_obs_pt = args[2]
#     is_missing_data = args[3]
#     n_gaussians_per_dim = args[4]
#     gauss_means = args[5]
#     gauss_stds = args[6]
#     n_gaussians = args[7]
#     W_min_pf = 0.0
#     W_min_mli = 0.0
#     FR_MAX = kwargs['FR_MAX']
#     activation_out = kwargs['activation_out']
#
#     # Parse parameters to be fit
#     alpha = params[0]
#     beta = params[1]
#     gamma = params[2]
#     epsilon = params[3]
#     W_max_pf = params[4]
#     pf_scale = 1 #params[5]
#     mli_scale = 1 #params[6]
#     # Set weights to initial fit values
#     W_pf = np.copy(W_0_pf) * pf_scale
#     W_mli = np.copy(W_0_mli) * mli_scale
#     # Ensure W_pf values are within range and store in output W_full
#     W_pf[(W_pf > W_max_pf).squeeze()] = W_max_pf
#     W_pf[(W_pf < W_min_pf).squeeze()] = W_min_pf
#     W_mli[(W_mli < W_min_mli).squeeze()] = W_min_mli
#     if kwargs['UPDATE_MLI_WEIGHTS']:
#         omega = params[5]
#         psi = params[6]
#         chi = params[7]
#         phi = params[8]
#         W_max_mli = params[9]
#         W_mli[(W_mli > W_max_mli).squeeze()] = W_max_mli
#     W_full = np.vstack((W_pf, W_mli))
#
#     for trial in range(0, n_trials):
#         state_trial = state[trial*n_obs_pt:(trial + 1)*n_obs_pt, :] # State for this trial
#         y_obs_trial = y[trial*n_obs_pt:(trial + 1)*n_obs_pt] # Observed FR for this trial
#         is_missing_data_trial = is_missing_data[trial*n_obs_pt:(trial + 1)*n_obs_pt] # Nan state points for this trial
#
#         # Convert state to input layer activations
#         state_input = af.eye_input_to_PC_gauss_relu(state_trial,
#                                         gauss_means, gauss_stds,
#                                         n_gaussians_per_dim=n_gaussians_per_dim)
#         # Set inputs derived from nan points to 0.0 so that the weights
#         # for these states are not affected during nans
#         state_input[is_missing_data_trial, :] = 0.0
#         # Expected rate this trial given updated weights
#         # Use maximum here because of relu activation of output
#         y_hat_trial = (np.dot(state_input, W_full) + b).squeeze()
#         if activation_out == "relu":
#             y_hat_trial = np.maximum(0, y_hat_trial)
#         # Store prediction for current trial
#         y_hat[trial*n_obs_pt:(trial + 1)*n_obs_pt] = y_hat_trial
#
#         # Update weights for next trial based on activations in this trial
#         state_input_pf = state_input[:, 0:n_gaussians]
#         # Rescaled trial firing rate in proportion to max
#         y_obs_trial = y_obs_trial / FR_MAX
#         # Binary CS for this trial
#         CS_trial_bin = CS[trial*n_obs_pt:(trial + 1)*n_obs_pt]
#
#         # Get LTD function for parallel fibers
#         pf_CS_LTD = f_pf_CS_LTD(CS_trial_bin, kwargs['tau_rise_CS'],
#                           kwargs['tau_decay_CS'], epsilon, 0.0)
#         # Convert to LTD input for Purkinje cell
#         pf_LTD = f_pf_LTD(pf_CS_LTD, state_input_pf, W_pf=W_pf, W_min_pf=W_min_pf)
#
#         # Create the LTP function for parallel fibers
#         pf_LTP_funs = f_pf_CS_LTP(CS_trial_bin, kwargs['tau_rise_CS_LTP'],
#                         kwargs['tau_decay_CS_LTP'], alpha)
#         pf_LTP_funs += f_pf_FR_LTP(y_obs_trial, beta)
#         pf_LTP_funs += f_pf_static_LTP(pf_CS_LTD, gamma)
#         pf_LTP_funs[pf_CS_LTD > 0.0] = 0.0
#         # Convert to LTP input for Purkinje cell
#         pf_LTP = f_pf_LTP(pf_LTP_funs, state_input_pf, W_pf=W_pf, W_max_pf=W_max_pf)
#         # Compute delta W_pf as LTP + LTD inputs and update W_pf
#         W_pf += ( pf_LTP[:, None] + pf_LTD[:, None] )
#
#         # Ensure W_pf values are within range and store in output W_full
#         W_pf[(W_pf > W_max_pf).squeeze()] = W_max_pf
#         W_pf[(W_pf < W_min_pf).squeeze()] = W_min_pf
#         W_full[0:n_gaussians] = W_pf
#
#         if kwargs['UPDATE_MLI_WEIGHTS']:
#             # MLI state input is all <= 0, so need to multiply by -1 here
#             state_input_mli = -1.0 * state_input[:, n_gaussians:]
#             # Create the MLI LTP weighting function
#             mli_CS_LTP = f_mli_CS_LTP(CS_trial_bin, kwargs['tau_rise_CS_mli_LTP'],
#                               kwargs['tau_decay_CS_mli_LTP'], omega, 0.0)
#             # Convert to LTP input for Purkinje cell MLI weights
#             mli_LTP = f_mli_LTP(mli_CS_LTP, state_input_mli, W_mli, W_max_mli)
#
#             # Create the LTD function for MLIs
#             # mli_LTD_funs = f_mli_CS_LTD(CS_trial_bin, kwargs['tau_rise_CS_mli_LTD'],
#             #                 kwargs['tau_decay_CS_mli_LTD'], psi)
#             # mli_LTD_funs = f_mli_FR_LTD(y_obs_trial, chi)
#             mli_LTD_funs = f_mli_static_LTD(mli_CS_LTP, phi)
#             mli_LTD_funs[mli_CS_LTP > 0.0] = 0.0
#             # Convert to LTD input for MLI
#             mli_LTD = f_mli_LTD(mli_LTD_funs, state_input_mli, W_mli, W_min_mli)
#             # Ensure W_mli values are within range and store in output W_full
#             W_mli += ( mli_LTP[:, None] + mli_LTD[:, None] )
#             W_mli[(W_mli > W_max_mli).squeeze()] = W_max_mli
#             W_mli[(W_mli < W_min_mli).squeeze()] = W_min_mli
#             W_full[n_gaussians:] = W_mli
#
#     residuals = np.sum((y - y_hat) ** 2)
#     return residuals

def fit_learning_rates(NN_FIT, blocks, trial_sets, learn_t_win=None, bin_width=10, bin_threshold=5):
    """ Need the trials from blocks and trial_sets to be ORDERED! Weights will
    be updated from one trial to the next as if they are ordered and will
    not check if the numbers are correct because it could fail for various
    reasons like aborted trials. """
    ftol=1e-8
    xtol=1e-8
    gtol=1e-8
    max_nfev=200000
    loss='linear'

    if learn_t_win is None:
        learn_t_win = NN_FIT.time_window
    NN_FIT.learn_rates_time_window = learn_t_win
    """ Get all the binned firing rate data """
    firing_rate, all_t_inds = NN_FIT.neuron.get_firing_traces(learn_t_win,
                                        blocks, trial_sets, return_inds=True)
    CS_bin_evts = NN_FIT.neuron.get_CS_dataseries_by_trial(learn_t_win,
                                blocks, trial_sets, nan_sacc=False)

    """ Here we have to do some work to get all the data in the correct format """
    # First get all firing rate data, bin and format
    binned_FR = bin_data(firing_rate, bin_width, bin_threshold)
    binned_FR = binned_FR.reshape(binned_FR.shape[0]*binned_FR.shape[1], order='C')

    # And for CSs
    binned_CS = bin_data(CS_bin_evts, bin_width, bin_threshold)
    # Convert to binary instead of binned average
    binned_CS[binned_CS > 0.0] = 1.0
    binned_CS = binned_CS.reshape(binned_CS.shape[0]*binned_CS.shape[1], order='C')

    """ Get all the binned eye data """
    eye_data, initial_shape = get_plasticity_data_trial_win(NN_FIT,
                                    blocks, trial_sets, learn_t_win,
                                    return_shape=True)
    eye_data = eye_data.reshape(initial_shape)
    # Use bin smoothing on data before fitting
    bin_eye_data = bin_data(eye_data, bin_width, bin_threshold)
    # Observations defined after binning
    n_trials = np.int32(bin_eye_data.shape[0]) # Total number of trials to fit
    n_obs_pt = np.int32(bin_eye_data.shape[1]) # Number of observations per trial
    # Reshape to 2D matrix
    bin_eye_data = bin_eye_data.reshape(
                            bin_eye_data.shape[0]*bin_eye_data.shape[1],
                            bin_eye_data.shape[2], order='C')
    # Make an index of all nans that we can use in objective function to set
    # the unit activations to 0.0
    eye_is_nan = np.any(np.isnan(bin_eye_data), axis=1)
    # Firing rate data is only NaN where data for a trial does not cover NN_FIT.time_window
    # So we need to find this separate from saccades and can set to 0.0 to ignore
    # We will AND this with where eye is NaN because both should be if data are truly missing
    is_missing_data = np.int32(np.isnan(binned_FR) | eye_is_nan)
    binned_FR[is_missing_data] = 0.0

    # Need the means and stds for converting state to input
    pos_means = NN_FIT.fit_results['gauss_basis_kinematics']['pos_means']
    vel_means = NN_FIT.fit_results['gauss_basis_kinematics']['vel_means']
    n_gaussians_per_dim = np.array([len(pos_means), len(pos_means),
                           len(vel_means), len(vel_means)], dtype=np.int32)
    gauss_means = np.hstack([pos_means,
                             pos_means,
                             vel_means,
                             vel_means], dtype=np.float64)
    pos_stds = NN_FIT.fit_results['gauss_basis_kinematics']['pos_stds']
    vel_stds = NN_FIT.fit_results['gauss_basis_kinematics']['vel_stds']
    gauss_stds = np.hstack([pos_stds,
                            pos_stds,
                            vel_stds,
                            vel_stds])
    n_gaussians = np.int32(len(gauss_means))

    # Defining learning function within scope so we have access to "NN_FIT"
    # and specifically the weights. Get here to save space
    W_0_pf = np.float64(NN_FIT.fit_results['gauss_basis_kinematics']['coeffs'][0:n_gaussians].squeeze())
    W_0_mli = np.float64(NN_FIT.fit_results['gauss_basis_kinematics']['coeffs'][n_gaussians:].squeeze())
    b = np.float64(NN_FIT.fit_results['gauss_basis_kinematics']['bias'])

    lf_kwargs = {'tau_rise_CS': int(np.around(50 /bin_width)),
                 'tau_decay_CS': int(np.around(50 /bin_width)),
                 'tau_rise_CS_LTP': int(np.around(-100 /bin_width)),
                 'tau_decay_CS_LTP': int(np.around(200 /bin_width)),
                 'tau_rise_CS_mli_LTP': int(np.around(80 /bin_width)),
                 'tau_decay_CS_mli_LTP': int(np.around(-40 /bin_width)),
                 'tau_rise_CS_mli_LTD': int(np.around(-40 /bin_width)),
                 'tau_decay_CS_mli_LTD': int(np.around(100 /bin_width)),
                 'FR_MAX': 500,
                 'UPDATE_MLI_WEIGHTS': False,
                 'activation_out': NN_FIT.activation_out,
                 }
    # Format of p0, upper, lower, index order for each variable to make this legible
    param_conds = {"alpha": (4.0, 0, np.inf, 0),
                   "beta": (1.0, 0, np.inf, 1),
                   "gamma": (1.0, 0, np.inf, 2),
                   "epsilon": (4.0, 0, np.inf, 3),
                   "W_max_pf": (10*np.amax(W_0_pf), np.amax(W_0_pf), np.inf, 4),
                   # "pf_scale": (1.05, 0.9, 1.15, 5),
                   # "mli_scale": (1.05, 0.9, 1.15, 6),
            }
    if lf_kwargs['UPDATE_MLI_WEIGHTS']:
        raise ValueError("check param nums")
        param_conds.update({"omega": (0.01, 0, np.inf, 5),
                            "psi": (0.01, 0, np.inf, 6),
                            "chi": (0.01, 0, np.inf, 7),
                            "phi": (0.01, 0, np.inf, 8),
                            "W_max_mli": (10*np.amax(W_0_mli), np.amax(W_0_mli), np.inf, 9),
                            })

    # Make sure params are in correct order and saved for input to least_squares
    p0 = [x[1][0] for x in sorted(param_conds.items(), key=lambda item: item[1][3])]
    lower_bounds = [x[1][1] for x in sorted(param_conds.items(), key=lambda item: item[1][3])]
    upper_bounds = [x[1][2] for x in sorted(param_conds.items(), key=lambda item: item[1][3])]
    W_min_pf = np.float64(0.0)
    FR_MAX = np.int32(lf_kwargs['FR_MAX'])
    tau_rise_CS = np.int32(lf_kwargs['tau_rise_CS'])
    tau_decay_CS = np.int32(lf_kwargs['tau_decay_CS'])
    tau_rise_CS_LTP = np.int32(lf_kwargs['tau_rise_CS_LTP'])
    tau_decay_CS_LTP = np.int32(lf_kwargs['tau_decay_CS_LTP'])
    # Finally append CS to inputs and get other args needed for learning function
    fit_inputs = np.hstack([bin_eye_data, binned_CS[:, None]])

    lf_args = (n_trials, n_obs_pt, is_missing_data,
                n_gaussians_per_dim, gauss_means, gauss_stds, n_gaussians,
                W_min_pf, FR_MAX, tau_rise_CS, tau_decay_CS, tau_rise_CS_LTP,
                tau_decay_CS_LTP)
    print(fit_inputs.shape, binned_FR.shape, W_0_pf.shape, W_0_mli.shape, b.shape)
    for a_ind, arg in enumerate((fit_inputs, binned_FR, W_0_pf, W_0_mli, b)):
        if isinstance(arg, np.ndarray):
            print(a_ind, arg.shape, arg.dtype)
        else:
            print(a_ind, type(arg))
    for a_ind, arg in enumerate(lf_args):
        if isinstance(arg, np.ndarray):
            print(a_ind, arg.shape, arg.dtype)
        else:
            print(a_ind, type(arg))
    # Fit the learning rates to the data
    result = least_squares(py_learning_function, p0,
                            args=(fit_inputs, binned_FR, W_0_pf, W_0_mli, b, *lf_args),
                            bounds=(lower_bounds, upper_bounds),
                            ftol=ftol,
                            xtol=xtol,
                            gtol=gtol,
                            max_nfev=max_nfev,
                            loss=loss)
    for key in param_conds.keys():
        param_ind = param_conds[key][3]
        NN_FIT.fit_results['gauss_basis_kinematics'][key] = result.x[param_ind]
    for key in lf_kwargs.keys():
        NN_FIT.fit_results['gauss_basis_kinematics'][key] = lf_kwargs[key]

    return result

def get_learning_weights_by_trial(NN_FIT, blocks, trial_sets, W_0_pf=None,
                                    W_0_mli=None, bin_width=10, bin_threshold=5):
    """ Need the trials from blocks and trial_sets to be ORDERED! """
    """ Get all the binned firing rate data """
    firing_rate, all_t_inds = NN_FIT.neuron.get_firing_traces(
                                        NN_FIT.learn_rates_time_window,
                                        blocks, trial_sets, return_inds=True)
    CS_bin_evts = NN_FIT.neuron.get_CS_dataseries_by_trial(
                                NN_FIT.learn_rates_time_window,
                                blocks, trial_sets, nan_sacc=False)

    """ Here we have to do some work to get all the data in the correct format """
    # First get all firing rate data, bin and format
    binned_FR = bin_data(firing_rate, bin_width, bin_threshold)
    binned_FR = binned_FR.reshape(binned_FR.shape[0]*binned_FR.shape[1], order='C')

    # And for CSs
    binned_CS = bin_data(CS_bin_evts, bin_width, bin_threshold)
    # Convert to binary instead of binned average
    binned_CS[binned_CS > 0.0] = 1.0
    binned_CS = binned_CS.reshape(binned_CS.shape[0]*binned_CS.shape[1], order='C')

    """ Get all the binned eye data """
    eye_data, initial_shape = get_plasticity_data_trial_win(NN_FIT,
                                    blocks, trial_sets,
                                    NN_FIT.learn_rates_time_window,
                                    return_shape=True)
    eye_data = eye_data.reshape(initial_shape)
    # Use bin smoothing on data before fitting
    bin_eye_data = bin_data(eye_data, bin_width, bin_threshold)
    # Observations defined after binning
    n_trials = bin_eye_data.shape[0] # Total number of trials to fit
    n_obs_pt = bin_eye_data.shape[1] # Number of observations per trial
    # Reshape to 2D matrix
    bin_eye_data = bin_eye_data.reshape(
                            bin_eye_data.shape[0]*bin_eye_data.shape[1],
                            bin_eye_data.shape[2], order='C')
    fit_inputs = np.hstack([bin_eye_data, binned_CS[:, None]])
    # Make an index of all nans that we can use in objective function to set
    # the unit activations to 0.0
    eye_is_nan = np.any(np.isnan(bin_eye_data), axis=1)
    # Firing rate data is only NaN where data for a trial does not cover NN_FIT.time_window
    # So we need to find this separate from saccades and can set to 0.0 to ignore
    # We will AND this with where eye is NaN because both should be if data are truly missing
    is_missing_data = np.isnan(binned_FR) | eye_is_nan
    binned_FR[is_missing_data] = 0.0

    # Need the means and stds for converting state to input
    pos_means = NN_FIT.fit_results['gauss_basis_kinematics']['pos_means']
    vel_means = NN_FIT.fit_results['gauss_basis_kinematics']['vel_means']
    n_gaussians_per_dim = [len(pos_means), len(pos_means),
                           len(vel_means), len(vel_means)]
    gauss_means = np.hstack([pos_means,
                             pos_means,
                             vel_means,
                             vel_means])
    pos_stds = NN_FIT.fit_results['gauss_basis_kinematics']['pos_stds']
    vel_stds = NN_FIT.fit_results['gauss_basis_kinematics']['vel_stds']
    gauss_stds = np.hstack([pos_stds,
                            pos_stds,
                            vel_stds,
                            vel_stds])
    n_gaussians = len(gauss_means)

    if W_0_pf is None:
        W_0_pf = NN_FIT.fit_results['gauss_basis_kinematics']['coeffs'][0:n_gaussians]
    if W_0_pf.shape[0] != n_gaussians:
        raise ValueError("Input W_0_pf must have match the fit coefficients shape of {0}.".format(n_gaussians))
    if W_0_mli is None:
        W_0_mli = NN_FIT.fit_results['gauss_basis_kinematics']['coeffs'][n_gaussians:]
    if W_0_mli.shape[0] != 8:
        raise ValueError("Input W_0_mli must have match the MLI coefficients shape of 8.")
    b = NN_FIT.fit_results['gauss_basis_kinematics']['bias']
    W_min_pf = 0.0
    W_min_mli = 0.0

    # Separate behavior state from CS inputs
    state = fit_inputs[:, 0:-1]
    CS = fit_inputs[:, -1]
    # Fixed input params into one dict to match above
    kwargs = {}
    for key in NN_FIT.fit_results['gauss_basis_kinematics'].keys():
        if "tau" in key:
            kwargs[key] = NN_FIT.fit_results['gauss_basis_kinematics'][key]
        elif key in ["UPDATE_MLI_WEIGHTS"]:
            kwargs[key] = NN_FIT.fit_results['gauss_basis_kinematics'][key]
    FR_MAX = NN_FIT.fit_results['gauss_basis_kinematics']['FR_MAX']
    # Fit parameters
    alpha = NN_FIT.fit_results['gauss_basis_kinematics']['alpha']
    beta = NN_FIT.fit_results['gauss_basis_kinematics']['beta']
    gamma = NN_FIT.fit_results['gauss_basis_kinematics']['gamma']
    epsilon = NN_FIT.fit_results['gauss_basis_kinematics']['epsilon']
    W_max_pf = NN_FIT.fit_results['gauss_basis_kinematics']['W_max_pf']
    pf_scale = 1. #NN_FIT.fit_results['gauss_basis_kinematics']['pf_scale']
    mli_scale = 1. #NN_FIT.fit_results['gauss_basis_kinematics']['mli_scale']
    W_pf = np.zeros(W_0_pf.shape) # Place to store updating result and copy to output
    W_pf[:] = pf_scale * W_0_pf # Initialize storage to start values
    W_mli = np.zeros(W_0_mli.shape) # Place to store updating result and copy to output
    W_mli[:] = mli_scale * W_0_mli # Initialize storage to start values
    # Ensure W_pf values are within range and store in output W_full
    W_pf[(W_pf > W_max_pf).squeeze()] = W_max_pf
    W_pf[(W_pf < W_min_pf).squeeze()] = W_min_pf
    W_mli[(W_mli < W_min_mli).squeeze()] = W_min_mli
    if kwargs['UPDATE_MLI_WEIGHTS']:
        omega = NN_FIT.fit_results['gauss_basis_kinematics']['omega']
        psi = NN_FIT.fit_results['gauss_basis_kinematics']['psi']
        chi = NN_FIT.fit_results['gauss_basis_kinematics']['chi']
        phi = NN_FIT.fit_results['gauss_basis_kinematics']['phi']
        W_max_mli = NN_FIT.fit_results['gauss_basis_kinematics']['W_max_mli']
        W_mli[(W_mli > W_max_mli).squeeze()] = W_max_mli
    W_full = np.vstack((W_pf, W_mli))
    weights_by_trial = {t_num: np.zeros(W_full.shape) for t_num in all_t_inds}

    for trial_ind, trial_num in zip(range(0, n_trials), all_t_inds):
        weights_by_trial[trial_num][:] = W_full # Copy W for this trial, befoe updating at end of loop
        state_trial = state[trial_ind*n_obs_pt:(trial_ind + 1)*n_obs_pt, :] # State for this trial
        y_obs_trial = binned_FR[trial_ind*n_obs_pt:(trial_ind + 1)*n_obs_pt] # Observed FR for this trial
        is_missing_data_trial = is_missing_data[trial_ind*n_obs_pt:(trial_ind + 1)*n_obs_pt] # Nan state points for this trial
        # Convert state to input layer activations
        state_input = af.eye_input_to_PC_gauss_relu(state_trial,
                                        gauss_means, gauss_stds,
                                        n_gaussians_per_dim=n_gaussians_per_dim)
        # Set inputs derived from nan points to 0.0 so t hat the weights
        # for these states are not affected during nans
        state_input[is_missing_data_trial, :] = 0.0
        state_input_pf = state_input[:, 0:n_gaussians]

        # Rescaled trial firing rate in proportion to max
        y_obs_trial = y_obs_trial / FR_MAX
        # Binary CS for this trial
        CS_trial_bin = CS[trial_ind*n_obs_pt:(trial_ind + 1)*n_obs_pt]

        # Get LTD function for parallel fibers
        pf_CS_LTD = f_pf_CS_LTD(CS_trial_bin, kwargs['tau_rise_CS'],
                          kwargs['tau_decay_CS'], epsilon, 0.0)
        # Convert to LTD input for Purkinje cell
        pf_LTD = f_pf_LTD(pf_CS_LTD, state_input_pf, W_pf=W_pf, W_min_pf=W_min_pf)

        # Create the LTP function for parallel fibers
        pf_LTP_funs = f_pf_CS_LTP(CS_trial_bin, kwargs['tau_rise_CS_LTP'],
                        kwargs['tau_decay_CS_LTP'], alpha)
        pf_LTP_funs += f_pf_FR_LTP(y_obs_trial, beta)
        pf_LTP_funs += f_pf_static_LTP(pf_CS_LTD, gamma)
        pf_LTP_funs[pf_CS_LTD > 0.0] = 0.0
        # Convert to LTP input for Purkinje cell
        pf_LTP = f_pf_LTP(pf_LTP_funs, state_input_pf, W_pf=W_pf, W_max_pf=W_max_pf)
        # Compute delta W_pf as LTP + LTD inputs and update W_pf
        W_pf += ( pf_LTP[:, None] + pf_LTD[:, None] )

        # Ensure W_pf values are within range and store in output W_full
        W_pf[(W_pf > W_max_pf).squeeze()] = W_max_pf
        W_pf[(W_pf < W_min_pf).squeeze()] = W_min_pf
        W_full[0:n_gaussians] = W_pf

        if kwargs['UPDATE_MLI_WEIGHTS']:
            # MLI state input is all <= 0, so need to multiply by -1 here
            state_input_mli = -1.0 * state_input[:, n_gaussians:]
            # Create the MLI LTP weighting function
            mli_CS_LTP = f_mli_CS_LTP(CS_trial_bin, kwargs['tau_rise_CS_mli_LTP'],
                              kwargs['tau_decay_CS_mli_LTP'], omega, 0.0)
            # Convert to LTP input for Purkinje cell MLI weights
            mli_LTP = f_mli_LTP(mli_CS_LTP, state_input_mli, W_mli, W_max_mli)

            # Create the LTD function for MLIs
            # mli_LTD_funs = f_mli_CS_LTD(CS_trial_bin, kwargs['tau_rise_CS_mli_LTD'],
            #                 kwargs['tau_decay_CS_mli_LTD'], psi)
            # mli_LTD_funs = f_mli_FR_LTD(y_obs_trial, chi)
            mli_LTD_funs = f_mli_static_LTD(mli_CS_LTP, phi)
            mli_LTD_funs[mli_CS_LTP > 0.0] = 0.0
            # Convert to LTD input for MLI
            mli_LTD = f_mli_LTD(mli_LTD_funs, state_input_mli, W_mli, W_min_mli)
            # Ensure W_mli values are within range and store in output W_full
            W_mli += ( mli_LTP[:, None] + mli_LTD[:, None] )
            W_mli[(W_mli > W_max_mli).squeeze()] = W_max_mli
            W_mli[(W_mli < W_min_mli).squeeze()] = W_min_mli
            W_full[n_gaussians:] = W_mli

        # if np.all(np.isnan(W_full)):
        #     print(alpha, beta, psi, omega)
        #     return LTP_Inputs, f_LTP, f_LTP_fixed, y_obs_trial, state_input, PC_FR_weight_LTP

    return weights_by_trial
