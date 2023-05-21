import numpy as np
from scipy.optimize import least_squares, basinhopping, differential_evolution
from NeuronAnalysis.fit_NN_model import bin_data, FitNNModel
from NeuronAnalysis.general import box_windows
from NeuronAnalysis.activation_functions import eye_input_to_PC_gauss_relu, gen_linspace_gaussians

# # Transform the eye data to the state_input layer activations
# state_input = np.zeros((X_trial.shape[0], X_trial.shape[1], W_trial.shape[1]))
# for trial in range(0, X_trial.shape[0]):
#     # This will modify stat_input IN PLACE for each trial
#     eye_input_to_PC_gauss_relu(X_trial[trial, :, :], gauss_means,
#                                 gauss_stds, n_gaussians_per_dim,
#                                 state_input[trial, :, :])
#
# for t_ind in range(0, X_trial.shape[0]):
#     # Each trial update the weights for W
#     W[:, 0] = W_trial[t_ind, :]
#     y_hat[t_ind, :] = (np.dot(state_input[trial, :, :], W) + b).squeeze()
#     pf_in[t_ind, :] = (np.dot(state_input[trial, :, :][:, 0:n_gaussians], W[0:n_gaussians, 0]) + b).squeeze()
#     mli_in[t_ind, :] = (np.dot(state_input[trial, :, :][:, n_gaussians:], W[n_gaussians:, 0]) + b).squeeze()

""" SOME FUNCTIONS FOR GETTING DATA TO PREDICT FIRING BASED ON PLASTIC WEIGHTS """
""" ********************************************************************** """
def comp_learning_response(NN_FIT, X_trial, W_trial, return_comp=False):
    """ Computes the estimated firing rate y_hat for each trial input in the
    array "X_trial" according to the trial by trial weights "W_trial." This can
    be used to estimate firing rates AFTER LEARNING WEIGHTS ARE FITTED and known
    for each trial and input in W_trial.
    """
    if X_trial.shape[2] != 8:
        raise ValueError("Gaussian basis kinematics model is fit for 8 data dimensions but input data dimension is {0}.".format(X.shape[1]))

    gauss_means, gauss_stds, n_gaussians_per_dim, n_gaussians =  NN_FIT.get_all_gauss_params()
    y_hat = np.zeros((X_trial.shape[0], X_trial.shape[1]))
    W = np.copy(NN_FIT.fit_results['gauss_basis_kinematics']['coeffs'])
    b = NN_FIT.fit_results['gauss_basis_kinematics']['bias']
    pf_in = np.zeros((X_trial.shape[0], X_trial.shape[1]))
    mli_in = np.zeros((X_trial.shape[0], X_trial.shape[1]))

    X_input = np.zeros((W_trial.shape[1], ))
    for t_ind in range(0, X_trial.shape[0]):
        # Transform X_data for this trial into input space
        X_input = eye_input_to_PC_gauss_relu(X_trial[t_ind, :, :],
                                        gauss_means, gauss_stds,
                                        n_gaussians_per_dim, X_input)
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
                                        weights_t_inds,
                                        return_comp=False, test_data_only=False,
                                        verbose=False, return_inds=False):
    """ Given the input array of weights in weights_by_trial which is an n trials
    by d input dimensions array of the weights for the NN_FIT model, this
    computes the expected firing rate at each trial requested in blocks and
    trial sets.
    """
    X, init_shape, t_inds = NN_FIT.get_gauss_basis_kinematics_predict_data_trial(
                            blocks, trial_sets, return_shape=True,
                            test_data_only=test_data_only, return_inds=True,
                            verbose=verbose)
    X = X.reshape(init_shape)
    # Get weights in a single matrix to pass through here according to weights_t_inds
    sel_t_inds, inds_all_t_inds, _ = np.intersect1d(weights_t_inds, t_inds, return_indices=True)
    # If the input request is valid, then it must be true that the requested
    # trials are a subset of the trials on which the weights have been calculated
    if not np.all(sel_t_inds == t_inds):
        raise ValueError("Requested trials in blocks and trial sets are not a subset of the trial weights input in weights_t_inds.")
    W_trial = weights_by_trial[inds_all_t_inds, :]
    if return_comp:
        y_hat, pf_in, mli_in = comp_learning_response(NN_FIT, X, W_trial,
                                                        return_comp=return_comp)
        if return_inds:
            return y_hat, pf_in, mli_in, sel_t_inds
        else:
            return y_hat, pf_in, mli_in
    else:
        y_hat = comp_learning_response(NN_FIT, X, W_trial)
        if return_inds:
            return y_hat, sel_t_inds
        else:
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
                                    return_inds=False):
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
    if return_inds:
        return eye_data, t_inds
    else:
        return eye_data


def get_firing_eye_by_trial(NN_FIT, time_window, blocks, trial_sets, return_inds=False):
    """ Get all the firing rate and eye data for the neuron fit in the NN_fit
    obejct Returned as a 3D array of trials x time x data_dim.
    """
    #Get the trial indices and use those to get behavior since neural data
    # can be fewer trials.
    firing_rate, all_t_inds = NN_FIT.neuron.get_firing_traces(time_window,
                                        blocks, trial_sets, return_inds=True)
    CS_bin_evts = NN_FIT.neuron.get_CS_dataseries_by_trial(time_window,
                                blocks, all_t_inds, nan_sacc=False)

    # Now get eye data for the same trials
    eye_data = get_plasticity_data_trial_win(NN_FIT, blocks, all_t_inds, time_window)

    if return_inds:
        return firing_rate, eye_data, CS_bin_evts, all_t_inds
    else:
        return firing_rate, eye_data, CS_bin_evts


""" THESE ARE THE LEARNING RULE PLASTICITY FUNCTIONS """
""" *********************************************************************** """

def f_pf_CS_LTD(CS_trial_bin, tau_1, tau_2, scale=1.0, delay=0, zeta_f_move=None):
    """ Computes the parallel fiber LTD as a function of time of the complex
    spike input f_CS with a kernel scaled from tau_1 to tau_2 with peak equal to
    scale and with CSs shifted by an amoutn of time "delay" INDICES (not time!). """
    # Just CS point plasticity
    if zeta_f_move is None:
        pf_CS_LTD = box_windows(CS_trial_bin, tau_1, tau_2, scale=scale)
    else:
        pf_CS_LTD = box_windows(CS_trial_bin, tau_1, tau_2, scale=1.0)
        add_zeta = (pf_CS_LTD * zeta_f_move)
        pf_CS_LTD *= scale
        pf_CS_LTD += add_zeta
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

def f_pf_move_LTD(pf_CS_LTD, move_m_trial, move_LTD_scale):
    """
    """
    # Add a term with movement magnitude times weight
    # pf_CS_LTD += (pf_CS_LTD * np.sqrt(move_m_trial) * move_LTD_scale)
    return pf_CS_LTD

def f_pf_LTD(pf_CS_LTD, state_input_pf, pf_LTD, W_pf=None, W_min_pf=0.0):
    """ Updates the parallel fiber LTD function of time "pf_CS_LTD" to be scaled
    by PC firing rate if input, then scales by the pf state input and finally
    weights by the current weight function. pf_LTD is MODIFIED IN PLACE!
    The output contains NEGATIVE values in pf_LTD. """
    # Sum of pf_CS_LTD weighted by activation for each input unit
    pf_LTD = np.dot(pf_CS_LTD, state_input_pf, out=pf_LTD)
    # Set state modification scaling according to current weight
    if W_pf is not None:
        pf_LTD *= (W_min_pf - W_pf) # Will all be negative values
    else:
        pf_LTD *= -1.0
    return pf_LTD

def f_pf_CS_LTP(CS_trial_bin, tau_1, tau_2, scale=1.0, zeta_f_move=None):
    """ Assumes CS_trial_bin is an array of 0's and 1's where 1's indicate that
    CS LTD is taking place and 0's indicate no CS related LTD. This function
    will then invert the CS train and form windows of LTP where LTD is absent.
    """
    if zeta_f_move is None:
        pf_CS_LTP = box_windows(CS_trial_bin, tau_1, tau_2, scale=scale)
    else:
        pf_CS_LTP = box_windows(CS_trial_bin, tau_1, tau_2, scale=1.0)
        add_zeta = (pf_CS_LTP * zeta_f_move)
        pf_CS_LTP *= scale
        pf_CS_LTP += add_zeta
    return pf_CS_LTP

def f_pf_static_LTP(pf_LTP_funs, pf_CS_LTD, static_weight_LTP, zeta_f_move=None):
    """
    """
    pf_LTP_funs += static_weight_LTP
    # pf_LTP_funs[pf_CS_LTD > 0.0] = 0.0
    if zeta_f_move is not None:
        pf_LTP_funs += (zeta_f_move)
    return pf_LTP_funs

def f_pf_FR_LTP(pf_LTP_funs, PC_FR, PC_FR_weight_LTP, zeta_f_move=None):
    """
    """
    # Add a term with firing rate times weight of constant LTP
    pf_LTP_funs += (PC_FR * PC_FR_weight_LTP)
    if zeta_f_move is not None:
        pf_LTP_funs += (PC_FR * zeta_f_move)
    return pf_LTP_funs

def f_pf_move_LTP(pf_LTP_funs, move_m_trial, move_LTP_scale):
    """
    """
    # Add a term with movement magnitude times weight
    # pf_LTP_funs *= np.sqrt(move_m_trial * move_LTP_scale + 1)
    return pf_LTP_funs

def f_pf_LTP(pf_LTP_funs, state_input_pf, pf_LTP, W_pf=None, W_max_pf=None):
    """ Updates the parallel fiber LTP function of time "pf_CS_LTP" to be scaled
    by PC firing rate if input, then scales by the pf state input and finally
    weights by the current weight function. pf_LTP is MODIFIED IN PLACE!
    W_max_pf should probably be input as FITTED PARAMETER! variable and is
    critical if using weight updates. Same for PC_FR_weight_LTP.
    """
    # Convert LTP functions to parallel fiber input space
    pf_LTP = np.dot(pf_LTP_funs, state_input_pf, out=pf_LTP)
    if W_pf is not None:
        if ( (W_max_pf is None) or (W_max_pf <= 0) ):
            raise ValueError("If updating weights by inputting values for W_pf, a W_max_pf > 0 must also be specified.")
        pf_LTP *= (W_max_pf - W_pf)
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
        mli_LTP *= (W_max_mli - W_mli)
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
        mli_LTD *= (W_min_mli - W_mli) # Will all be negative values
    else:
        mli_LTD *= -1.0
    return mli_LTD

""" *********************************************************************** """

def run_learning_model(weights_0, input_state, FR, CS, move_magn, int_rate,
                        param_kwargs, func_kwargs, arr_kwargs={},
                        return_residuals=True, return_y_hat=False,
                        return_weights=False):
    """ Iterates over the trial-wise data input_state, FR, and CS starting
    with weights_0 at trial1 (row index 0) and runs the learning algorithm to
    update the weights and firing rate prediction of each trial given the
    learning parameters in learn_params dictionary. Input "input_state" is, the
    post-activation function input to the Purkinje cell being fit. e.g., the
    output of "eye_input_to_PC_gauss_relu", organized in a 3D matrix in the
    same format as the usual eye data.

    Parameters
    ----------
    param_kwargs : dict
        Each key contains a single FITTED parameter specifying the learning model
    func_kwargs : dict
        Each key contains a parameter/constant value required to specify the
        learning model but that are not fitted variables
    arr_kwargs : dict
        Each key contains a numpy array that has been initialized for the
        purposes of pre-allocated memory and speed when calling iteratively.

    Returns
    -------
    residuals : float
        The sum of residual squared error over all trials and time points
    y_hat : np.ndarray
        A 2 dimensional numpy array (num trials x num time points) of the
        model firing rate prediction for each trial and time point.
    weights : np.ndarray
        A 2 dimensional numpy array (num trials x num input dimensions/weights)
        of the value of the input weights for each dimenions for each trial.
    """
    # Check and/or create any pre-allocated arrays
    if not isinstance(arr_kwargs, dict):
        arr_kwargs = {}
    shapes_allocated_arrays = {'fr_obs_trial': (FR.shape[1], ),
                               'y_hat_trial': (FR.shape[1], ),
                               'pf_LTD': (func_kwargs['n_gaussians'], ),
                               'pf_LTP': (func_kwargs['n_gaussians'], ),
                                }
    for key in shapes_allocated_arrays.keys():
        try:
            curr_shape = arr_kwargs[key].shape
            for dim in range(0, len(shapes_allocated_arrays[key])):
                if dim >= len(curr_shape):
                    raise ValueError("Incorrect pre-allocated array shape for {0}. Expected {1} but got {2}.".format(key, curr_shape, shapes_allocated_arrays[key]))
                if curr_shape[dim] != shapes_allocated_arrays[key][dim]:
                    raise ValueError("Incorrect pre-allocated array shape for {0}. Expected {1} but got {2}.".format(key, curr_shape, shapes_allocated_arrays[key]))
        except KeyError:
            # This allocated array not found so allocate here
            arr_kwargs[key] = np.zeros(shapes_allocated_arrays[key])
    # Initialize and set for return the requested items
    if return_y_hat:
        y_hat_by_trial = np.zeros(FR.shape)
    if return_weights:
        weights_by_trial = np.zeros((input_state.shape[0], input_state.shape[2]))

    # Set weights to initial values by copying so we don't alter inputs
    W_full = np.copy(weights_0)
    # Make convenient views of weight matrix for pf and mli separately
    W_pf = W_full[0:func_kwargs['n_gaussians']]
    W_pf *= param_kwargs['pf_scale']
    W_mli = W_full[func_kwargs['n_gaussians']:]
    W_mli *= param_kwargs['mli_scale']
    # Ensure W_pf values are within range
    W_pf[(W_pf > param_kwargs['W_max_pf'])] = param_kwargs['W_max_pf']
    W_pf[(W_pf < func_kwargs['W_min_pf'])] = func_kwargs['W_min_pf']
    W_mli[(W_mli < func_kwargs['W_min_mli'])] = func_kwargs['W_min_mli']

    residuals = 0.0
    for trial in range(0, input_state.shape[0]):
        state_trial = input_state[trial, :, :] # Input state for this trial
        move_m_trial = move_magn[trial, :] # Movement for this trial
        # Copy over so we can manipulate and not overwrite the input
        arr_kwargs['fr_obs_trial'][:] = FR[trial, :]  # Observed FR for this trial

        # Expected rate this trial given updated weights. Updated in-place to
        # preallocated array y_hat_trial
        np.dot(state_trial, W_full, out=arr_kwargs['y_hat_trial'])
        arr_kwargs['y_hat_trial'] += int_rate # Add the bias term
        if func_kwargs['activation_out'] == "relu":
            # Set maximum IN PLACE
            np.maximum(0., arr_kwargs['y_hat_trial'], out=arr_kwargs['y_hat_trial'])
        # Now we can convert any nans to 0.0 so they don't affect residuals
        arr_kwargs['y_hat_trial'][func_kwargs['is_missing_data'][trial, :]] = 0.0
        arr_kwargs['fr_obs_trial'][func_kwargs['is_missing_data'][trial, :]] = 0.0

        # Store requested outputs as needed
        if return_residuals:
            # Add residuals for current trial
            residuals += np.sum((arr_kwargs['fr_obs_trial'] - arr_kwargs['y_hat_trial'])**2)
        if return_y_hat:
            # Store y_hat for this trial
            y_hat_by_trial[trial, :] = arr_kwargs['y_hat_trial'][:]
        if return_weights:
            weights_by_trial[trial, :] = W_full[:]

        # Update weights for next trial based on activations in this trial
        state_input_pf = state_trial[:, 0:func_kwargs['n_gaussians']]
        # Rescaled trial firing rate in proportion to max OVERWRITES y_obs_trial!
        arr_kwargs['fr_obs_trial'] /= func_kwargs['FR_MAX']
        CS_trial_bin = CS[trial, :] # Get CS view for this trial

        # Get LTD function for parallel fibers
        # zeta_f_move = np.sqrt(move_m_trial) * param_kwargs['move_LTD_scale']
        pf_CS_LTD = f_pf_CS_LTD(CS_trial_bin, func_kwargs['tau_rise_CS'],
                          func_kwargs['tau_decay_CS'], param_kwargs['epsilon'],
                          0.0, zeta_f_move=None)
        # Add to pf_CS_LTD in place
        pf_CS_LTD = f_pf_move_LTD(pf_CS_LTD, move_m_trial, param_kwargs['move_LTD_scale'])
        # Convert to LTD input for Purkinje cell
        arr_kwargs['pf_LTD'] = f_pf_LTD(pf_CS_LTD, state_input_pf,
                                        arr_kwargs['pf_LTD'], W_pf=W_pf,
                                        W_min_pf=func_kwargs['W_min_pf'])

        # Create the LTP function for parallel fibers
        zeta_f_move = np.sqrt(move_m_trial) * param_kwargs['move_LTP_scale']
        pf_LTP_funs = f_pf_CS_LTP(CS_trial_bin, func_kwargs['tau_rise_CS_LTP'],
                                    func_kwargs['tau_decay_CS_LTP'],
                                    param_kwargs['alpha'], zeta_f_move=None)
        # These functions add on to pf_LTP_funs in place
        pf_LTP_funs = f_pf_FR_LTP(pf_LTP_funs, arr_kwargs['fr_obs_trial'],
                                    param_kwargs['beta'], zeta_f_move=None)
        pf_LTP_funs = f_pf_static_LTP(pf_LTP_funs, pf_CS_LTD,
                                        param_kwargs['gamma'], zeta_f_move=None)
        # Make LTP not directly compete with LTD
        pf_LTP_funs[pf_CS_LTD > 0.0] = 0.0
        # Convert to LTP input for Purkinje cell
        arr_kwargs['pf_LTP'] = f_pf_LTP(pf_LTP_funs, state_input_pf, arr_kwargs['pf_LTP'], W_pf=W_pf, W_max_pf=param_kwargs['W_max_pf'])

        # Compute delta W_pf as LTP + LTD inputs and update W_pf
        # Since W_pf is view into W_full, W_full is updated IN PLACE here!
        W_pf += ( arr_kwargs['pf_LTP'] + arr_kwargs['pf_LTD'] )
        # Ensure W_pf values are within range and store in output W_full
        W_pf[(W_pf > param_kwargs['W_max_pf'])] = param_kwargs['W_max_pf']
        W_pf[(W_pf < func_kwargs['W_min_pf'])] = func_kwargs['W_min_pf']

    # Initialize and set for return the requested items
    return_items = []
    if return_residuals:
        return_items.append(residuals)
    if return_y_hat:
        return_items.append(y_hat_by_trial)
    if return_weights:
        return_items.append(weights_by_trial)
    if len(return_items) == 0:
        return None
    elif len(return_items) == 1:
        return return_items[0]
    else:
        return tuple(return_items)

def obj_fun(params, state_input, FR, *args):
    """ A wrapper for run_learning_model that can be called as an objective
    function by a scipy optimizeer. It stores the parameters and other needed
    values into dictionaries used by run_learning_model and returns the
    residual squared error to the optimizer. """
    # Unpack all the extra args needed here to pass into learning function
    param_conds = args[0]
    weights_0 = args[1]
    int_rate = args[2]
    binned_CS = args[3]
    move_magn = args[4]
    fr_obs_trial = args[5]
    y_hat_trial = args[6]
    pf_LTD = args[7]
    pf_LTP = args[8]
    func_kwargs = args[9]

    # Build dictionary of params being fit to pass to learning function
    # according to the initialization dictionary param_conds
    param_kwargs = {}
    for p in param_conds.keys():
        param_kwargs[p] = params[param_conds[p][3]]

    # Assign preallocated arrays for learning function to use
    arr_kwargs = {'fr_obs_trial': fr_obs_trial,
                  'y_hat_trial': y_hat_trial,
                  'pf_LTD': pf_LTD,
                  'pf_LTP': pf_LTP,
                 }

    residuals = run_learning_model(weights_0, state_input, FR, binned_CS,
                                    move_magn, int_rate,
                                    param_kwargs, func_kwargs, arr_kwargs={},
                                    return_residuals=True, return_y_hat=False,
                                    return_weights=False)
    return residuals

def init_learn_fit_params(CS_LTD_win, CS_LTP_win, bin_width,
                            W_0_pf=None, W_0_mli=None):
    """
    """
    if W_0_pf is None:
        W_max_pf0 = 15.0
        W_max_pf_min = 1.0
    else:
        W_max_pf0 = 10*np.amax(W_0_pf)
        W_max_pf_min = np.amax(W_0_pf)
    if W_0_mli is None:
        W_0_mli0 = 15.0
        W_max_mli_min = 1.0
    else:
        W_0_mli0 = 10*np.amax(W_0_mli)
        W_max_mli_min = np.amax(W_0_mli)
    lf_kwargs = {'tau_rise_CS': int(np.around(CS_LTD_win[0] /bin_width)),
                 'tau_decay_CS': int(np.around(CS_LTD_win[1] /bin_width)),
                 'tau_rise_CS_LTP': int(np.around(CS_LTP_win[0] /bin_width)),
                 'tau_decay_CS_LTP': int(np.around(CS_LTP_win[1] /bin_width)),
                 'FR_MAX': 500,
                 'UPDATE_MLI_WEIGHTS': False,
                 }
    # Format of p0, upper, lower, index order for each variable to make this legible
    param_conds = {"alpha": (0.01, 0, 10., 0),
                   "beta": (0.001, 0, 1.0, 1),
                   "gamma": (0.001, 0, 1.0, 2),
                   "epsilon": (4000.0, 0, 400000, 3),
                   "W_max_pf": (W_max_pf0, W_max_pf_min, 100., 4),
                   "move_LTD_scale": (0.001, 0.0, 0.1, 5),
                   "move_LTP_scale": (0.001, 0.0, 0.1, 6),
                   "pf_scale": (1.0, 0.7, 1.3, 7),
                   "mli_scale": (1.0, 0.7, 1.3, 8),
            }
    if lf_kwargs['UPDATE_MLI_WEIGHTS']:
        raise ValueError("check param nums")
        param_conds.update({"omega": (1.0, 0, np.inf, 5),
                            "psi": (1.0, 0, np.inf, 6),
                            "chi": (1.0, 0, np.inf, 7),
                            "phi": (1.0, 0, np.inf, 8),
                            "W_max_mli": (W_0_mli0, W_max_mli_min, np.inf, 9),
                            })

    # Make sure params are in correct order and saved for input to least_squares
    p0 = [x[1][0] for x in sorted(param_conds.items(), key=lambda item: item[1][3])]
    lower_bounds = [x[1][1] for x in sorted(param_conds.items(), key=lambda item: item[1][3])]
    upper_bounds = [x[1][2] for x in sorted(param_conds.items(), key=lambda item: item[1][3])]

    return lf_kwargs, param_conds, p0, lower_bounds, upper_bounds

def fit_learning_rates(NN_FIT, blocks, trial_sets, learn_fit_window=None,
                        bin_width=10, bin_threshold=5, CS_LTD_win=[25, 0],
                        CS_LTP_win=[-100, 200]):
    """ Need the trials from blocks and trial_sets to be ORDERED! Weights will
    be updated from one trial to the next as if they are ordered and will
    not check if the numbers are correct because it could fail for various
    reasons like aborted trials. """
    if learn_fit_window is None:
        learn_fit_window = NN_FIT.time_window
    NN_FIT.learn_rates_time_window = learn_fit_window

    # Get firing rate and eye data for trials to be fit
    firing_rate, eye_data, CS_bin_evts = get_firing_eye_by_trial(NN_FIT,
                                            learn_fit_window, blocks, trial_sets)
    # Now we need to bin the data over time
    binned_FR = bin_data(firing_rate, bin_width, bin_threshold).squeeze()
    bin_eye_data = bin_data(eye_data, bin_width, bin_threshold)
    binned_CS = bin_data(CS_bin_evts, bin_width, bin_threshold).squeeze()
    # Convert CS to binary instead of binned average
    binned_CS[binned_CS > 0.0] = 1.0

    # Make an index of all nans that we can use in objective function to set
    # the unit activations to 0.0
    eye_is_nan = np.any(np.isnan(bin_eye_data), axis=2)
    # Firing rate data is only NaN where data for a trial does not cover NN_FIT.time_window
    # So we need to find this separate from saccades and can set to 0.0 to ignore
    # We will OR this with where eye is NaN to guarantee all missing points included
    is_missing_data = np.isnan(binned_FR) | eye_is_nan

    # Need the means and stds for converting state to input
    gauss_params =  NN_FIT.get_all_gauss_params()
    gauss_means, gauss_stds, n_gaussians_per_dim, n_gaussians = gauss_params
    # Get the initial starting values for model fit
    W_0_pf, W_0_mli, weights_0, int_rate = NN_FIT.get_model()

    # Transform the eye data to the state_input layer activations
    state_input = np.zeros((bin_eye_data.shape[0], bin_eye_data.shape[1], weights_0.size))
    for trial in range(0, bin_eye_data.shape[0]):
        # This will modify stat_input IN PLACE for each trial
        eye_input_to_PC_gauss_relu(bin_eye_data[trial, :, :], gauss_means,
                                    gauss_stds, n_gaussians_per_dim,
                                    state_input[trial, :, :])
    # Compute that magnitude of the eye movement vector for scaling learning magnitude
    move_magn = np.linalg.norm(bin_eye_data[:, :, 2:4], axis=2)

    # Convert missing input data to 0's. MUST DO THIS TO INPUT NOT EYE DATA
    # because eye_data = 0 implies activiations in the input state!
    state_input[is_missing_data, :] = 0.0
    move_magn[is_missing_data] = 0.0

    init_params = init_learn_fit_params(CS_LTD_win, CS_LTP_win, bin_width,
                                        W_0_pf, W_0_mli)
    func_kwargs, param_conds, p0, lower_bounds, upper_bounds = init_params
    # Add extra needed args to pass in func_kwargs
    func_kwargs.update({'n_gaussians': n_gaussians,
                        'is_missing_data': is_missing_data,
                        'W_min_pf': 0.0,
                        'W_min_mli': 0.0,
                        'activation_out': NN_FIT.activation_out,
                        })

    # Finally append CS to inputs and get other args needed for learning function
    fr_obs_trial = np.zeros((bin_eye_data.shape[1], ))
    y_hat_trial = np.zeros((bin_eye_data.shape[1], ))
    pf_LTD = np.zeros((n_gaussians))
    pf_LTP = np.zeros((n_gaussians))
    lf_args = (param_conds, weights_0, int_rate, binned_CS, move_magn,
                fr_obs_trial, y_hat_trial, pf_LTD, pf_LTP, func_kwargs)

    ftol=1e-4
    xtol=1e-8
    gtol=1e-8
    max_nfev=2000
    loss='linear'
    # Fit the learning rates to the data
    result = least_squares(obj_fun, p0,
                            args=(state_input, binned_FR, *lf_args),
                            bounds=(lower_bounds, upper_bounds),
                            ftol=ftol,
                            xtol=xtol,
                            gtol=gtol,
                            max_nfev=max_nfev,
                            loss=loss)

    result_copy = np.copy(result.x)
    for key in param_conds.keys():
        param_ind = param_conds[key][3]
        NN_FIT.fit_results['gauss_basis_kinematics'][key] = result_copy[param_ind]
    for key in func_kwargs.keys():
        NN_FIT.fit_results['gauss_basis_kinematics'][key] = func_kwargs[key]


    # # Create a local minimizer
    # bounds = [(lb, ub) for lb, ub in zip(lower_bounds, upper_bounds)]
    # minimizer_kwargs = {"method": "L-BFGS-B",
    #                     "args": (fit_inputs, binned_FR, W_0_pf, W_0_mli, b, *lf_args),
    #                     "bounds": bounds}
    # # Call basinhopping
    # result = basinhopping(learning_function, p0, minimizer_kwargs=minimizer_kwargs, niter=10, disp=True)
    # result.residuals = learning_function(result.x, fit_inputs, binned_FR,
    #                                 W_0_pf, W_0_mli, b, *lf_args, **func_kwargs)


    # # Note that differential_evolution() does not allow method specification
    # # for the minimization step because it has its own mechanism.
    # # We now define the bounds as a list of (min, max) pairs for each element in x
    # bounds = [(lb, ub) for lb, ub in zip(lower_bounds, upper_bounds)]
    #
    # # differential_evolution function takes the objective function and the bounds as main arguments.
    # result = differential_evolution(func=learning_function,
    #                                 bounds=bounds,
    #                                 args=(fit_inputs, binned_FR, W_0_pf, W_0_mli, b, *lf_args),
    #                                 workers=-1,
    #                                 disp=True) # Display status messages

    return result

def pred_run_learn_model(NN_FIT, state_input, FR, *args):
    """ A wrapper for run_learning_model that can be called as an objective
    function by a scipy optimizeer. It stores the parameters and other needed
    values into dictionaries used by run_learning_model and returns the
    residual squared error to the optimizer. """
    # Unpack all the extra args needed here to pass into learning function
    all_t_inds = args[0]
    binned_CS = args[1]
    move_magn = args[2]
    fr_obs_trial = args[3]
    y_hat_trial = args[4]
    pf_LTD = args[5]
    pf_LTP = args[6]
    func_kwargs = args[7]
    param_conds = args[8]

    # Build dictionary of params being fit to pass to learning function
    # according to the initialization dictionary param_conds
    param_kwargs = {}
    for p in param_conds.keys():
        param_kwargs[p] = NN_FIT.fit_results['gauss_basis_kinematics'][p]

    # Assign preallocated arrays for learning function to use
    arr_kwargs = {'fr_obs_trial': fr_obs_trial,
                  'y_hat_trial': y_hat_trial,
                  'pf_LTD': pf_LTD,
                  'pf_LTP': pf_LTP,
                 }
    weights_0 = NN_FIT.fit_results['gauss_basis_kinematics']['coeffs'].squeeze()
    int_rate = NN_FIT.fit_results['gauss_basis_kinematics']['bias'][0]

    weights = run_learning_model(weights_0, state_input, FR, binned_CS,
                                    move_magn, int_rate,
                                    param_kwargs, func_kwargs, arr_kwargs={},
                                    return_residuals=False, return_y_hat=False,
                                    return_weights=True)
    return weights

def get_learned_weights(NN_FIT, blocks, trial_sets,
                        bin_width=10, bin_threshold=5, CS_LTD_win=[25, 0],
                        CS_LTP_win=[-100, 200]):
    """ Need the trials from blocks and trial_sets to be ORDERED! """
    """ Get all the binned firing rate data. Get the trial indices and use those
    to get behavior since neural data can be fewer trials. """
    # Get firing rate and eye data for trials to be fit
    firing_rate, eye_data, CS_bin_evts, all_t_inds = get_firing_eye_by_trial(NN_FIT,
                                                        NN_FIT.learn_rates_time_window,
                                                        blocks, trial_sets,
                                                        return_inds=True)
    # Now we need to bin the data over time
    binned_FR = bin_data(firing_rate, bin_width, bin_threshold).squeeze()
    bin_eye_data = bin_data(eye_data, bin_width, bin_threshold)
    binned_CS = bin_data(CS_bin_evts, bin_width, bin_threshold).squeeze()
    # Convert CS to binary instead of binned average
    binned_CS[binned_CS > 0.0] = 1.0

    # Make an index of all nans that we can use in objective function to set
    # the unit activations to 0.0
    eye_is_nan = np.any(np.isnan(bin_eye_data), axis=2)
    # Firing rate data is only NaN where data for a trial does not cover NN_FIT.time_window
    # So we need to find this separate from saccades and can set to 0.0 to ignore
    # We will OR this with where eye is NaN to guarantee all missing points included
    is_missing_data = np.isnan(binned_FR) | eye_is_nan

    # Need the means and stds for converting state to input
    gauss_params =  NN_FIT.get_all_gauss_params()
    gauss_means, gauss_stds, n_gaussians_per_dim, n_gaussians = gauss_params
    # Get the initial starting values for model fit
    W_0_pf, W_0_mli, weights_0, int_rate = NN_FIT.get_model()

    # Transform the eye data to the state_input layer activations
    state_input = np.zeros((bin_eye_data.shape[0], bin_eye_data.shape[1], weights_0.size))
    for trial in range(0, bin_eye_data.shape[0]):
        # This will modify stat_input IN PLACE for each trial
        eye_input_to_PC_gauss_relu(bin_eye_data[trial, :, :], gauss_means,
                                    gauss_stds, n_gaussians_per_dim,
                                    state_input[trial, :, :])
    # Compute that magnitude of the eye movement vector for scaling learning magnitude
    move_magn = np.linalg.norm(bin_eye_data[:, :, 2:4], axis=2)

    # Convert missing input data to 0's. MUST DO THIS TO INPUT NOT EYE DATA
    # because eye_data = 0 implies activiations in the input state!
    state_input[is_missing_data, :] = 0.0
    move_magn[is_missing_data] = 0.0

    # Doing this is a bit repetitive and not totally necessary but I want to
    # keep these functions matched as much as possible
    init_params = init_learn_fit_params(CS_LTD_win, CS_LTP_win, bin_width,
                                        W_0_pf, W_0_mli)
    func_kwargs, param_conds, p0, lower_bounds, upper_bounds = init_params
    # Add extra needed args to pass in func_kwargs
    func_kwargs.update({'n_gaussians': n_gaussians,
                        'is_missing_data': is_missing_data,
                        'W_min_pf': 0.0,
                        'W_min_mli': 0.0,
                        'activation_out': NN_FIT.activation_out,
                        })

    # Finally append CS to inputs and get other args needed for learning function
    fr_obs_trial = np.zeros((bin_eye_data.shape[1], ))
    y_hat_trial = np.zeros((bin_eye_data.shape[1], ))
    pf_LTD = np.zeros((n_gaussians))
    pf_LTP = np.zeros((n_gaussians))
    lf_args = (all_t_inds, binned_CS, move_magn,
                fr_obs_trial, y_hat_trial, pf_LTD, pf_LTP,
                func_kwargs, param_conds)
    # Run learning model to get the weights at each trial in all_t_inds
    weights = pred_run_learn_model(NN_FIT, state_input, binned_FR, *lf_args)
    return weights, all_t_inds

def fit_basic_NNModel(NN_FIT, intrinsic_rate0, bin_width, bin_threshold):
    """ Basically a helper function for get_intrisic_rate_and_CSwin that sets
    up some simple hard coded gaussian means and stds for doing the iterative
    fitting. The fit info is saved to the input NN_FIT object."""
    pos_n_gaussians = 25 *3
    vel_n_gaussians = 35 *3
    std_gaussians = 2
    pos_std = std_gaussians
    vel_std = std_gaussians
    pos_range = 25
    vel_range = 35
    fit_avg_data=False
    quick_lag_step=8
    train_split =1.
    pos_means, pos_stds = gen_linspace_gaussians(pos_range, pos_n_gaussians, pos_std)
    vel_means, vel_stds = gen_linspace_gaussians(vel_range, vel_n_gaussians, vel_std)

    NN_FIT.fit_gauss_basis_kinematics(pos_means, pos_stds, vel_means, vel_stds,
                                    activation_out="relu",
                                    intrinsic_rate0=intrinsic_rate0,
                                    bin_width=bin_width, bin_threshold=bin_threshold,
                                    fit_avg_data=fit_avg_data,
                                    quick_lag_step=quick_lag_step,
                                    train_split=train_split,
                                    learning_rate=0.02,
                                    epochs=200,
                                    batch_size=1200)
    return

def get_intrisic_rate_and_CSwin(NN_FIT, blocks, trial_sets, learn_fit_window=None,
                                bin_width=10, bin_threshold=5):
    """ Hard code intrinsic rate starting points.
    "None" uses default near median rate."""
    # test_intrinsic_rates = [x for x in np.linspace(0, 100, 5)]
    # test_intrinsic_rates[0] = None
    test_intrinsic_rates = [57.]
    CS_wins = [ [[150, -150], [50, -50]],
                [[100, -100],  [0, 0]],
                [[50, -50],     [-50, 50]],
                [[0, 0],     [-100, 100]],
                ]
    # test_intrinsic_rates = [x for x in np.linspace(0, 60, 1)]
    # test_intrinsic_rates[0] = None
    # CS_wins = [ [[100, -50],  [-50, 150]],
    #             [[50, 0],     [-100, 200]],
    #             ]
    min_resids = np.inf
    best_intrinsic_rate = None
    best_CS_wins = None
    best_result = None
    best_iter = None
    n_iter = 0
    for int_rate in test_intrinsic_rates:
        # Fit NN model with current intrinsic rate starting point
        fit_basic_NNModel(NN_FIT, int_rate, bin_width, bin_threshold)
        print("Intrinsic firing rate: ", NN_FIT.fit_results['gauss_basis_kinematics']['bias'][0])

        for curr_win in CS_wins:
            print("Checking CS_wins [{0}] and [{1}].".format(curr_win[0], curr_win[1]))
            result = fit_learning_rates(NN_FIT, blocks, trial_sets, learn_fit_window,
                                           bin_width=bin_width, bin_threshold=bin_threshold,
                                           CS_LTD_win=curr_win[0],
                                           CS_LTP_win=curr_win[1])
            print("RESULTS for iter", n_iter)
            print("Cost", result.cost)
            print("Residuals", result.residuals)
            for i, p in enumerate(result.x):
                print(i, " : ", p)
            # Save the results if they are best
            if result.residuals < min_resids:
                min_resids = result.residuals
                best_intrinsic_rate = NN_FIT.fit_results['gauss_basis_kinematics']['bias']
                best_CS_wins = curr_win
                best_result = result
                best_iter = n_iter
            n_iter += 1
    print("Picked rate", best_intrinsic_rate, "and windows", best_CS_wins, "from iter", best_iter, "of", n_iter)
    return best_result, best_intrinsic_rate, best_CS_wins
