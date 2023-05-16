



def box_windows(spike_train, box_pre, box_post, scale=1.0):
    """Same as convolve but just does loops instead of convolution"""
    window_sig = np.zeros_like(spike_train)
    for t in range(0, spike_train.shape[0]):
        if spike_train[t] > 0:
            w_start = max(0, t-box_pre)
            if w_start >= window_sig.shape[0]:
                continue
            w_stop = min(window_sig.shape[0], t+box_post+1)
            if w_stop < 0:
                continue
            for w_t in range(w_start, w_stop):
                window_sig[w_t] = scale
    return window_sig


def f_pf_CS_LTD(CS_trial_bin, tau_1, tau_2, scale=1.0, delay=0):
    """ Computes the parallel fiber LTD as a function of time of the complex
    spike input f_CS with a kernel scaled from tau_1 to tau_2 with peak equal to
    scale and with CSs shifted by an amoutn of time "delay" INDICES (not time!). """
    # Just CS point plasticity
    pf_CS_LTD = box_windows(CS_trial_bin, tau_1, tau_2)
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
    pf_CS_LTP = box_windows(CS_trial_bin, tau_1, tau_2)
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
    activation_out = kwargs['activation_out']



    # W_min_pf = np.float64(0.0)
    # FR_MAX = np.float64(kwargs['FR_MAX'])
    # tau_rise_CS = np.int32(kwargs['tau_rise_CS'])
    # tau_decay_CS = np.int32(kwargs['tau_decay_CS'])
    # tau_rise_CS_LTP = np.int32(kwargs['tau_rise_CS_LTP'])
    # tau_decay_CS_LTP = np.int32(kwargs['tau_decay_CS_LTP'])
    # lf_args = (n_trials, n_obs_pt,
    #             n_gaussians_per_dim, gauss_means, gauss_stds, n_gaussians,
    #             W_min_pf, FR_MAX, tau_rise_CS, tau_decay_CS, tau_rise_CS_LTP,
    #             tau_decay_CS_LTP)
    # cy_residuals = py_learning_function(params, x, y, W_0_pf, W_0_mli, b, *lf_args)



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
    W_pf[(W_pf > W_max_pf)] = W_max_pf
    W_pf[(W_pf < W_min_pf)] = W_min_pf
    W_mli[(W_mli < W_min_mli)] = W_min_mli
    if kwargs['UPDATE_MLI_WEIGHTS']:
        omega = params[5]
        psi = params[6]
        chi = params[7]
        phi = params[8]
        W_max_mli = params[9]
        W_mli[(W_mli > W_max_mli)] = W_max_mli
    W_full = np.concatenate((W_pf, W_mli))

    iter_residuals = 0.0
    for trial in range(0, n_trials):
        state_trial = state[trial*n_obs_pt:(trial + 1)*n_obs_pt, :] # State for this trial
        y_obs_trial = y[trial*n_obs_pt:(trial + 1)*n_obs_pt] # Observed FR for this trial
        # is_missing_data_trial = is_missing_data[trial*n_obs_pt:(trial + 1)*n_obs_pt] # Nan state points for this trial

        # Convert state to input layer activations
        state_input = af.eye_input_to_PC_gauss_relu(state_trial,
                                        gauss_means, gauss_stds,
                                        n_gaussians_per_dim=n_gaussians_per_dim)
        # Set inputs derived from nan points to 0.0 so that the weights
        # for these states are not affected during nans
        # state_input[is_missing_data_trial, :] = 0.0
        # Expected rate this trial given updated weights
        # Use maximum here because of relu activation of output
        y_hat_trial = (np.dot(state_input, W_full) + b)
        if activation_out == "relu":
            y_hat_trial = np.maximum(0, y_hat_trial)
        # Store prediction for current trial
        y_hat[trial*n_obs_pt:(trial + 1)*n_obs_pt] = y_hat_trial



        # for t_i in range(0, n_obs_pt):
        #     iter_residuals += np.sqrt((y_obs_trial[t_i] - y_hat_trial[t_i]) ** 2)




        # Update weights for next trial based on activations in this trial
        state_input_pf = state_input[:, 0:n_gaussians]
        # Rescaled trial firing rate in proportion to max OVERWRITES y_obs_trial!
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
        # Convert to LTP input for Purkinje cell
        pf_LTP = f_pf_LTP(pf_LTP_funs, state_input_pf, W_pf=W_pf, W_max_pf=W_max_pf)
        # Compute delta W_pf as LTP + LTD inputs and update W_pf
        W_pf += ( pf_LTP + pf_LTD )

        # Ensure W_pf values are within range and store in output W_full
        W_pf[(W_pf > W_max_pf)] = W_max_pf
        W_pf[(W_pf < W_min_pf)] = W_min_pf
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
            W_mli[(W_mli > W_max_mli)] = W_max_mli
            W_mli[(W_mli < W_min_mli)] = W_min_mli
            W_full[n_gaussians:] = W_mli

    # if cy_residuals != iter_residuals:
    #     print("Residual misatch: ", cy_residuals, iter_residuals, np.abs(cy_residuals - iter_residuals))
    #     print("With params: ", params)
    residuals = np.sum(np.sqrt((y - y_hat) ** 2))
    return residuals

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
    n_trials = bin_eye_data.shape[0] # Total number of trials to fit
    n_obs_pt = bin_eye_data.shape[1] # Number of observations per trial
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
    is_missing_data = np.isnan(binned_FR) | eye_is_nan
    bin_eye_data[is_missing_data, :] = 0.0
    binned_FR[is_missing_data] = 0.0
    binned_CS[is_missing_data] = 0.0

    # Need the means and stds for converting state to input
    pos_means = NN_FIT.fit_results['gauss_basis_kinematics']['pos_means']
    vel_means = NN_FIT.fit_results['gauss_basis_kinematics']['vel_means']
    n_gaussians_per_dim = np.array([len(pos_means), len(pos_means),
                           len(vel_means), len(vel_means)], dtype=np.int32)
    gauss_means = np.hstack([pos_means,
                             pos_means,
                             vel_means,
                             vel_means], dtype=np.float64)
    pos_stds = np.float64(NN_FIT.fit_results['gauss_basis_kinematics']['pos_stds'])
    vel_stds = np.float64(NN_FIT.fit_results['gauss_basis_kinematics']['vel_stds'])
    gauss_stds = np.hstack([pos_stds,
                            pos_stds,
                            vel_stds,
                            vel_stds], dtype=np.float64)
    n_gaussians = np.int32(len(gauss_means))

    # Defining learning function within scope so we have access to "NN_FIT"
    # and specifically the weights. Get here to save space
    W_0_pf = np.float64(NN_FIT.fit_results['gauss_basis_kinematics']['coeffs'][0:n_gaussians].squeeze())
    W_0_mli = np.float64(NN_FIT.fit_results['gauss_basis_kinematics']['coeffs'][n_gaussians:].squeeze())
    b = np.float64(NN_FIT.fit_results['gauss_basis_kinematics']['bias'])

    lf_kwargs = {'tau_rise_CS': int(np.around(25 /bin_width)),
                 'tau_decay_CS': int(np.around(0 /bin_width)),
                 'tau_rise_CS_LTP': int(np.around(-100 /bin_width)),
                 'tau_decay_CS_LTP': int(np.around(200 /bin_width)),
                 # 'tau_rise_CS_mli_LTP': int(np.around(80 /bin_width)),
                 # 'tau_decay_CS_mli_LTP': int(np.around(-40 /bin_width)),
                 # 'tau_rise_CS_mli_LTD': int(np.around(-40 /bin_width)),
                 # 'tau_decay_CS_mli_LTD': int(np.around(100 /bin_width)),
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
    rescale_1e4 = ["alpha", "beta", "gamma", "epsilon"]
                   # "omega", "psi", "chi", "phi"]

    # Make sure params are in correct order and saved for input to least_squares
    p0 = [x[1][0] for x in sorted(param_conds.items(), key=lambda item: item[1][3])]
    lower_bounds = [x[1][1] for x in sorted(param_conds.items(), key=lambda item: item[1][3])]
    upper_bounds = [x[1][2] for x in sorted(param_conds.items(), key=lambda item: item[1][3])]

    # Finally append CS to inputs and get other args needed for learning function
    fit_inputs = np.hstack([bin_eye_data, binned_CS[:, None]])
    lf_args = (bin_width, n_trials, n_obs_pt, is_missing_data,
                n_gaussians_per_dim, gauss_means, gauss_stds, n_gaussians)


    # W_min_pf = np.float64(0.0)
    # FR_MAX = np.int32(lf_kwargs['FR_MAX'])
    # tau_rise_CS = np.int32(lf_kwargs['tau_rise_CS'])
    # tau_decay_CS = np.int32(lf_kwargs['tau_decay_CS'])
    # tau_rise_CS_LTP = np.int32(lf_kwargs['tau_rise_CS_LTP'])
    # tau_decay_CS_LTP = np.int32(lf_kwargs['tau_decay_CS_LTP'])
    # lf_args = (n_trials, n_obs_pt, is_missing_data,
    #             n_gaussians_per_dim, gauss_means, gauss_stds, n_gaussians,
    #             W_min_pf, FR_MAX, tau_rise_CS, tau_decay_CS, tau_rise_CS_LTP,
    #             tau_decay_CS_LTP)
    # print("gauss means", gauss_means.shape, gauss_means.dtype)
    # print("gauss stds", gauss_stds.shape, gauss_stds.dtype)
    # print(fit_inputs.shape, binned_FR.shape, W_0_pf.shape, W_0_mli.shape, b.shape)
    # for a_ind, arg in enumerate((fit_inputs, binned_FR, W_0_pf, W_0_mli, b)):
    #     if isinstance(arg, np.ndarray):
    #         print(a_ind, arg.shape, arg.dtype)
    #     else:
    #         print(a_ind, type(arg))
    # for a_ind, arg in enumerate(lf_args):
    #     if isinstance(arg, np.ndarray):
    #         print(a_ind, arg.shape, arg.dtype)
    #     else:
    #         print(a_ind, type(arg))
    # all_args = [fit_inputs, binned_FR, W_0_pf, W_0_mli, b]
    # for arg in lf_args:
    #     all_args.append(arg)
    # all_args.extend([p0, lower_bounds, upper_bounds, ftol, xtol, gtol, max_nfev, loss])



    # import pickle
    # save_name = "/home/nate/temp/test_NN_fit.pickle"
    # with open(save_name, 'wb') as fp:
    #     pickle.dump(all_args, fp, protocol=-1)
    # return



    # Fit the learning rates to the data
    result = least_squares(learning_function, p0,
                            args=(fit_inputs, binned_FR, W_0_pf, W_0_mli, b, *lf_args),
                            kwargs=lf_kwargs,
                            bounds=(lower_bounds, upper_bounds),
                            ftol=ftol,
                            xtol=xtol,
                            gtol=gtol,
                            max_nfev=max_nfev,
                            loss=loss)
    for key in param_conds.keys():
        param_ind = param_conds[key][3]
        NN_FIT.fit_results['gauss_basis_kinematics'][key] = result.x[param_ind]
        if key in rescale_1e4:
            NN_FIT.fit_results['gauss_basis_kinematics'][key] /= 1e4
    for key in lf_kwargs.keys():
        NN_FIT.fit_results['gauss_basis_kinematics'][key] = lf_kwargs[key]

    return result
