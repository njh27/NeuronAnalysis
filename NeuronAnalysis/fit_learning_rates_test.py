

def boxcar_convolve(spike_train, box_pre, box_post, scale=1.0):
    """ Converts events in spike train to box shape windows of duration pre-event
    equal to box_pre and duration after event box_post. Negative box indices
    will shift the window away from events. If box indices overlap positive/
    negative in the wrong way this will return all zeros. e.g. if box_pre > 0
    and box_post < 0 and abs(box_post) > box_pre. pre == post == 0 returns
    single point values where spike_train == 1"""
    center_ind = max(abs(box_pre), abs(box_post)) + 1 # plus 1 for cushion on ends
    kernel = np.zeros(2*center_ind + 1)
    kernel[center_ind-box_pre:center_ind+box_post+1] = scale
    filtered_sig = np.convolve(spike_train, kernel, mode='same')
    return filtered_sig


def f_pf_CS_LTD(CS_trial_bin, tau_1, tau_2, scale=1.0, delay=0):
    """ Computes the parallel fiber LTD as a function of time of the complex
    spike input f_CS with a kernel scaled from tau_1 to tau_2 with peak equal to
    scale and with CSs shifted by an amoutn of time "delay" INDICES (not time!). """
    # Just CS point plasticity
    pf_CS_LTD = boxcar_convolve(CS_trial_bin, tau_1, tau_2)
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
    pf_CS_LTP = boxcar_convolve(CS_trial_bin, tau_1, tau_2)
    pf_CS_LTP[pf_CS_LTP > 0] = scale
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
    return pf_FR_LTP

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


def learning_function(params, x, y, W_0_pf, W_0_mli, b, *args, **kwargs):
    """ Defines the learning model we are fitting to the data """
    # Separate behavior state from CS inputs
    state = x[:, 0:-1]
    CS = x[:, -1]
    y_hat = np.zeros(x.shape[0])
    # Extract other precomputed necessary args
    bin_width = args[0]
    n_trials = args[1]
    n_obs_pt = args[2]
    is_missing_data = args[3]
    n_gaussians_per_dim = args[4]
    gauss_means = args[5]
    gauss_stds = args[6]
    n_gaussians = args[7]
    W_min_pf = 0.0
    W_min_mli = 0.0
    FR_MAX = kwargs['FR_MAX']

    # Parse parameters to be fit
    alpha = params[0] / 1e4
    beta = params[1] / 1e4
    gamma = params[2] / 1e4
    epsilon = params[3] / 1e4
    W_max_pf = params[4]
    pf_scale = 1 #params[5]
    mli_scale = 1 #params[6]
    # Set weights to initial fit values
    W_pf = np.copy(W_0_pf) * pf_scale
    W_mli = np.copy(W_0_mli) * mli_scale
    # Ensure W_pf values are within range and store in output W_full
    W_pf[(W_pf > W_max_pf).squeeze()] = W_max_pf
    W_pf[(W_pf < W_min_pf).squeeze()] = W_min_pf
    W_mli[(W_mli < W_min_mli).squeeze()] = W_min_mli
    if kwargs['UPDATE_MLI_WEIGHTS']:
        omega = params[5] / 1e4
        psi = params[6] / 1e4
        chi = params[7] / 1e4
        phi = params[8] / 1e4
        W_max_mli = params[9]
        W_mli[(W_mli > W_max_mli).squeeze()] = W_max_mli
    W_full = np.vstack((W_pf, W_mli))

    for trial in range(0, n_trials):
        state_trial = state[trial*n_obs_pt:(trial + 1)*n_obs_pt, :] # State for this trial
        y_obs_trial = y[trial*n_obs_pt:(trial + 1)*n_obs_pt] # Observed FR for this trial
        is_missing_data_trial = is_missing_data[trial*n_obs_pt:(trial + 1)*n_obs_pt] # Nan state points for this trial

        # Convert state to input layer activations
        state_input = af.eye_input_to_PC_gauss_relu(state_trial,
                                        gauss_means, gauss_stds,
                                        n_gaussians_per_dim=n_gaussians_per_dim)
        # Set inputs derived from nan points to 0.0 so that the weights
        # for these states are not affected during nans
        state_input[is_missing_data_trial, :] = 0.0
        # Expected rate this trial given updated weights
        # Use maximum here because of relu activation of output
        y_hat_trial = np.maximum(0, np.dot(state_input, W_full) + b).squeeze()
        # Store prediction for current trial
        y_hat[trial*n_obs_pt:(trial + 1)*n_obs_pt] = y_hat_trial

        # Update weights for next trial based on activations in this trial
        state_input_pf = state_input[:, 0:n_gaussians]
        # Rescaled trial firing rate in proportion to max
        y_obs_trial = y_obs_trial / FR_MAX
        # Binary CS for this trial
        CS_trial_bin = CS[trial*n_obs_pt:(trial + 1)*n_obs_pt]

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

    # missing_y_hat = np.isnan(y_hat)
    # residuals = (y[~missing_y_hat] - y_hat[~missing_y_hat]) ** 2
    residuals = (y - y_hat) ** 2
    return residuals
