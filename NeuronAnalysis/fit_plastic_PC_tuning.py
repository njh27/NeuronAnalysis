import numpy as np
from scipy.optimize import differential_evolution, minimize
from NeuronAnalysis.NN_granule_layer import GC_activations
from NeuronAnalysis.general import box_windows



def high_pass_two_state(data, cutoff_freq1, cutoff_freq2, fs):
    # Design the Butterworth filters
    b1, a1 = butter(2, cutoff_freq1, 'high', fs=fs)
    b2, a2 = butter(2, cutoff_freq2, 'high', fs=fs)

    # Apply the filters to get the two states
    state1 = filtfilt(b1, a1, data)
    state2 = filtfilt(b2, a2, data)

    # Subtract the slow state from the fast state to get the final output
    output = state1 - state2

    return output

def decay_fun(data, tau, threshold=5):
    t_0 = np.argmax(data > threshold)
    t_vals = np.arange(0, data.size)
    decay_kernel = np.exp(-1 * (t_vals - t_0) / tau)
    decay_kernel[0:t_0] = 0.
    filt_data = data * decay_kernel
    return filt_data



def get_sim_data_from_neuron(neuron, time_window, blocks, trial_sets, return_inds=False):
    """ Extracts and formats the data from the neuron in the blocks and trial sets and outputs
    in a format that can be dumped into a SimPCTuning object.
    """
    SS_dataseries_by_trial, all_t_inds = neuron.get_firing_traces(time_window, blocks,
                                                            trial_sets, return_inds=True)
    CS_dataseries_by_trial = neuron.get_CS_dataseries_by_trial(time_window, blocks, 
                                                               all_t_inds, nan_sacc=False)
    pos_p, pos_l = neuron.session.get_xy_traces("eye position", time_window, 
                                                blocks, all_t_inds, return_inds=False)
    vel_p, vel_l = neuron.session.get_xy_traces("eye velocity", time_window, 
                                                blocks, all_t_inds, return_inds=False)
    eye_data = np.stack((pos_p, pos_l, vel_p, vel_l), axis=2)
    if return_inds:
        return SS_dataseries_by_trial, CS_dataseries_by_trial, eye_data, all_t_inds
    else:
        return SS_dataseries_by_trial, CS_dataseries_by_trial, eye_data
    
def get_eye_data_traces(neuron, time_window, blocks, trial_sets, lag=0, return_inds=False):
    """ Gets eye position and velocity in array of trial x self.time_window
        3rd dimension of array is ordered as pursuit, learning position,
        then pursuit, learning velocity.
        Data are only retrieved for valid neuron trials!
    """
    lag_time_window = [time_window[0] + np.int32(lag), time_window[1] + np.int32(lag)]
    trial_sets = neuron.append_valid_trial_set(trial_sets)
    pos_p, pos_l, t_inds = neuron.session.get_xy_traces("eye position",
                            lag_time_window, blocks, trial_sets,
                            return_inds=True)
    vel_p, vel_l = neuron.session.get_xy_traces("eye velocity",
                            lag_time_window, blocks, trial_sets,
                            return_inds=False)
    eye_data = np.stack((pos_p, pos_l, vel_p, vel_l), axis=2)
    if return_inds:
        return eye_data, t_inds
    else:
        return eye_data

def eye_to_gc_activations(eye_data, granule_cells):
    """
    """
    eye_data_reshape = eye_data.shape
    eye_data = eye_data.reshape(eye_data.shape[0]*eye_data.shape[1], eye_data.shape[2], order='C')
    gc_activations = GC_activations(granule_cells, eye_data)
    gc_activations = gc_activations.reshape((eye_data_reshape[0], eye_data_reshape[1], gc_activations.shape[1]))
    return gc_activations
    # gc_activations = np.zeros((eye_data.shape[0], eye_data.shape[1], len(granule_cells)))
    # for trial in range(0, eye_data.shape[0]):
    #     gc_activations[trial, :, :] = GC_activations(granule_cells, eye_data[trial, :, :])
    # return gc_activations



def shuffle_CS(CS_train, min_offset=-10, max_offset=10):
    # Get indices where the array is 1
    indices = np.where(CS_train == 1)[0]
    # Calculate new indices
    new_indices = indices + np.random.randint(min_offset, max_offset+1, size=len(indices))
    # Handle the cases when the new indices are out of the array bounds
    new_indices = np.clip(new_indices, 0, len(CS_train)-1)
    # Create a new CS_train full of zeros
    new_CS_train = np.zeros_like(CS_train)
    # Set the shuffled positions to 1
    new_CS_train[new_indices] = 1

    return new_CS_train



def run_learning_model(weights_0, granule_activations, CS, model_params):
    """
    """
    weights = np.copy(weights_0)
    # Allocate CS_trial
    # CS_trial = np.zeros((CS.shape[1]))
    for trial in range(0, granule_activations.shape[0]):
        CS_trial = np.copy(CS[trial, :])
        CS_trial = box_windows(CS_trial, -50, 50, scale=1.0)[None, :]
        gc_trial = np.copy(granule_activations[trial, :, :])
        n_obs_trial = np.count_nonzero(~np.any(np.isnan(gc_trial), axis=1))
        gc_trial[np.isnan(gc_trial)] = 0.
        LTD = CS_trial @ gc_trial
        LTD /= n_obs_trial
        LTD *= model_params['epsilon']
        LTD *= -1.
        # LTD *= (model_params['w_min'] - weights)
        
        # Then LTP
        CS_trial_inv = CS_trial > 0.
        CS_trial[CS_trial_inv] = 0.
        CS_trial[~CS_trial_inv] = 1.
        LTP = CS_trial @ gc_trial
        LTP /= n_obs_trial
        LTP *= model_params['alpha'] 
        # LTP *= (model_params['w_max'] - weights)

        # Then add them up and adjust weights
        delta_w = LTP + LTD
        weights += delta_w

    return weights

def predict_learning_model(weights, granule_activations, intrinsic_rate, SS, return_residuals=True, return_y_hat=False):
    """
    """
    # Initialize and set for return the requested items
    if return_y_hat:
        y_hat_by_trial = np.zeros(SS.shape)

    # Allocate space for these so we don't have to make new one each iteration
    trial_gc_activation = np.zeros((granule_activations.shape[1], granule_activations.shape[2]))
    trial_SS = np.zeros((granule_activations.shape[1], ))
    nan_gc = np.zeros((granule_activations.shape[1], ), dtype='bool')
    nan_ss = np.zeros((granule_activations.shape[1], ), dtype='bool')
    nan_inds = np.zeros((granule_activations.shape[1], ), dtype='bool')

    # Run through trials and compute y_hat and the residuals with SS
    residuals = 0.0
    for trial in range(0, granule_activations.shape[0]):
        trial_gc_activation[:] = granule_activations[trial, :, :]
        trial_SS[:] = SS[trial, :]

        nan_gc[:] = np.any(np.isnan(trial_gc_activation), axis=1)
        nan_ss[:] = np.isnan(trial_SS)
        nan_inds[:] = (nan_gc | nan_ss)
        trial_n_obs = np.count_nonzero(~nan_inds)
        # trial_gc_activation[nan_inds, :] = 0.

        # Setting to intrinsic rate will make this equal to y_hat_trial and not affect residuals
        # trial_SS[nan_inds] = intrinsic_rate
        y_hat_trial = trial_gc_activation @ weights.T + intrinsic_rate
        y_hat_trial = np.squeeze(y_hat_trial)

        # Store requested outputs as needed
        if return_residuals:
            # Add residuals for current trial
            # print(f"y hat {y_hat_trial.shape}, SS {trial_SS.shape} nan inds {nan_inds.shape}")
            residuals_trial = np.sum(np.sqrt((y_hat_trial[~nan_inds] - trial_SS[~nan_inds]) ** 2)) / trial_n_obs
            residuals += residuals_trial
        if return_y_hat:
            # Store y_hat for this trial
            y_hat_by_trial[trial, :] = y_hat_trial

    # Initialize and set for return the requested items
    return_items = []
    if return_residuals:
        return_items.append(residuals)
    if return_y_hat:
        return_items.append(y_hat_by_trial)
    if len(return_items) == 0:
        return None
    elif len(return_items) == 1:
        return return_items[0]
    else:
        return tuple(return_items)

def obj_learning_model(params, granule_activations, SS, CS, *args):
    """
    """
    # Unpack all the extra args needed here to pass into learning function
    param_conds = args[0]
    t_set_select = args[1]
    short_gc_activations = args[2]
    short_CS = args[3]

    # Dictionary of all possible parameters for learning model set to dummy
    # null values that will have no effect on learning model
    model_params = {"alpha": 0.0,
                    "epsilon": 0.0,
                    "intrinsic_rate": 0.0,
                    # "w_max": 100.,
                    # "w_min": -100.,
                    }
    # Build dictionary of params being fit to pass to learning function
    # according to the initialization dictionary param_conds
    for p in param_conds.keys():
        model_params[p] = params[param_conds[p][3]]
        
    # Compute the weights given the current model parameters
    weights_0 = np.zeros((1, granule_activations.shape[2]))

    if short_gc_activations is None:
        # Use the regular full activations for weights
        weights = run_learning_model(weights_0, granule_activations, CS, model_params)
    else:
        # Use the random shortened activations to compute weights
        # weights = run_learning_model(weights_0, granule_activations, CS, model_params)
        # weights = run_learning_model(weights, short_gc_activations, short_CS, model_params)
        weights = run_learning_model(weights_0, short_gc_activations, short_CS, model_params)

    if len(t_set_select) > 0:
        granule_activations, SS = convert_predict_to_mean(granule_activations, SS, t_set_select)

    residuals = predict_learning_model(weights, granule_activations, model_params['intrinsic_rate'], 
                                       SS, return_residuals=True, return_y_hat=False)
    return residuals


def convert_predict_to_mean(granule_activations, SS, t_set_select):
    """ Gets the means of activations and SS
    """
    ga_out = np.zeros((len(t_set_select), granule_activations.shape[1], granule_activations.shape[2]))
    ss_out = np.zeros((len(t_set_select), SS.shape[1]))
    for t_ind, t_set in enumerate(t_set_select.keys()):
        ga_out[t_ind, :, :] = np.nanmean(granule_activations[t_set_select[t_set], :, :], axis=0)
        ss_out[t_ind, :] = np.nanmean(SS[t_set_select[t_set], :], axis=0)
    return ga_out, ss_out

def init_learn_fit_params():
    """
    """
    # Format of p0, lower, upper,
    param_conds = {"alpha": (1e-3, 0.0, 1e-1),
                   "epsilon": (1e-2, 0.0, 1e-1),
                   "intrinsic_rate": (75., 20., 100.),
                #    "w_max": (1., 0.5, 100),
                #    "w_min": (-1., -100, -0.5),
                    }
    # index order for each variable
    param_ind = 0
    for key in param_conds.keys():
        # Append param_ind to each tuple
        param_conds[key] = (*param_conds[key], param_ind)
        param_ind += 1

    # Make sure params are in correct order and saved for input to objective function
    p0 = [x[1][0] for x in sorted(param_conds.items(), key=lambda item: item[1][3])]
    lower_bounds = [x[1][1] for x in sorted(param_conds.items(), key=lambda item: item[1][3])]
    upper_bounds = [x[1][2] for x in sorted(param_conds.items(), key=lambda item: item[1][3])]

    return param_conds, p0, lower_bounds, upper_bounds


def generate_short_trials(eye_data, granule_cells, CS, N_trials=1000, ind_mean=100, ind_min_max=[200, 700]):
    """
    """
    all_durs = []
    n_durs = 0
    while n_durs < N_trials:
        trial_durs = np.random.exponential(ind_mean, size=N_trials-n_durs)
        trial_durs = np.clip(trial_durs, 0, ind_min_max[1] - ind_min_max[0])
        n_durs += trial_durs.shape[0]
        all_durs.append(np.int64(trial_durs))
    all_durs = np.hstack(all_durs, dtype=np.int64)

    short_eye_data = np.full((N_trials, eye_data.shape[1], eye_data.shape[2]), np.nan)
    short_CS = np.full((N_trials, CS.shape[1]), np.nan)
    trial_select = np.random.randint(0, eye_data.shape[0], size=N_trials)
    for st in range(0, N_trials):
        # Calculate the indices for cutting and new duration
        cut_ind_1 = int(ind_min_max[0] + all_durs[st] // 2)
        cut_ind_2 = int(ind_min_max[1] - all_durs[st] // 2)
        total_dur = cut_ind_1 + (eye_data.shape[1] - cut_ind_2)
        # Select shortened velocity profile
        short_eye_data[st, 0:cut_ind_1, 2] = eye_data[trial_select[st], 0:cut_ind_1, 2]
        short_eye_data[st, cut_ind_1:total_dur, 2] = eye_data[trial_select[st], cut_ind_2:, 2]
        short_eye_data[st, 0:cut_ind_1, 3] = eye_data[trial_select[st], 0:cut_ind_1, 3]
        short_eye_data[st, cut_ind_1:total_dur, 3] = eye_data[trial_select[st], cut_ind_2:, 3]
            
        # Then compute new shortened position from velocity. 
        # 1.25 scaling empirically works well to accomodate velocity filtering
        short_eye_data[st, :, 0] = 1.25*np.cumsum(short_eye_data[st, :, 2]) / 1000
        short_eye_data[st, :, 1] = 1.25*np.cumsum(short_eye_data[st, :, 3]) / 1000
        # Finally grab shortened CS
        short_CS[st, 0:cut_ind_1] = CS[trial_select[st], 0:cut_ind_1]
        short_CS[st, cut_ind_1:total_dur] = CS[trial_select[st], cut_ind_2:]

    gc_activations = eye_to_gc_activations(short_eye_data, granule_cells)
    return gc_activations, short_CS


def generate_truncated_trials(eye_data, granule_cells, CS, N_trials=1000, ind_mean=100, ind_min=200):
    """
    """
    ind_min_max = [ind_min, eye_data.shape[1]]
    all_durs = []
    n_durs = 0
    while n_durs < N_trials:
        trial_durs = np.random.exponential(ind_mean, size=N_trials-n_durs)
        trial_durs = np.clip(trial_durs, 0, ind_min_max[1] - ind_min_max[0])
        n_durs += trial_durs.shape[0]
        all_durs.append(np.int64(trial_durs))
    all_durs = np.hstack(all_durs, dtype=np.int64)

    short_eye_data = np.full((N_trials, eye_data.shape[1], eye_data.shape[2]), np.nan)
    short_CS = np.full((N_trials, CS.shape[1]), np.nan)
    trial_select = np.random.randint(0, eye_data.shape[0], size=N_trials)
    for st in range(0, N_trials):
        # Calculate the indices for cutting and new duration
        cut_ind_1 = int(ind_min_max[0] + all_durs[st])
        # Select shortened eye profile
        short_eye_data[st, 0:cut_ind_1, 0] = eye_data[trial_select[st], 0:cut_ind_1, 0]
        short_eye_data[st, 0:cut_ind_1, 1] = eye_data[trial_select[st], 0:cut_ind_1, 1]
        short_eye_data[st, 0:cut_ind_1, 2] = eye_data[trial_select[st], 0:cut_ind_1, 2]
        short_eye_data[st, 0:cut_ind_1, 3] = eye_data[trial_select[st], 0:cut_ind_1, 3]
        # Finally grab shortened CS
        short_CS[st, 0:cut_ind_1] = CS[trial_select[st], 0:cut_ind_1]

    gc_activations = eye_to_gc_activations(short_eye_data, granule_cells)
    return gc_activations, short_CS


def fit_plastic_tuning_model(eye_data, granule_cells, SS, CS, t_set_select={}):
    """
    """
    use_diff_evolution = True
    fit_short = True
    # Convert eye data to granule cell activations
    if fit_short:
        short_gc_activations, short_CS = generate_short_trials(eye_data, granule_cells, CS, 
                                                            N_trials=1000, ind_mean=200, ind_min_max=[250, 1000])
        # short_gc_activations, short_CS = generate_truncated_trials(eye_data, granule_cells, CS, 
        #                                                            N_trials=1000, ind_mean=100, ind_min=200)
    else:
        short_gc_activations = None
        short_CS = None
    gc_activations = eye_to_gc_activations(eye_data, granule_cells)
    
    param_conds, p0, lower_bounds, upper_bounds = init_learn_fit_params()
    lf_args = (param_conds, t_set_select, short_gc_activations, short_CS)

    if use_diff_evolution:
        # Note that differential_evolution() does not allow method specification
        # for the minimization step because it has its own mechanism.
        # We now define the bounds as a list of (min, max) pairs for each element in x
        bounds = [(lb, ub) for lb, ub in zip(lower_bounds, upper_bounds)]
        # differential_evolution function takes the objective function and the bounds as main arguments.
        result = differential_evolution(func=obj_learning_model,
                                        bounds=bounds,
                                        args=(gc_activations, SS, CS, *lf_args),
                                        workers=8, updating='deferred', popsize=12,
                                        disp=True) # Display status messages
    else:
        p0 = [v[0] for v in param_conds.values()]
        lower = [v[1] for v in param_conds.values()]
        upper = [v[2] for v in param_conds.values()]
        bounds = list(zip(lower, upper))

        # Additional arguments
        additional_args = (gc_activations, SS, CS) + lf_args

        # Call the optimizer
        result = minimize(obj_learning_model, p0, args=additional_args, bounds=bounds, method='L-BFGS-B')

    return result, (gc_activations, SS, CS, lf_args)


class FitPlasticPCTuning(object):
    """ Class that simulates how a PC tuning response would emerge given an input set of
    granule cells along with the behavior and complex spikes that will be used to run
    the simulation. This sim will be run trial-wise over the given behavior and CS data
    by trial.
    """
    def __init__(self, granule_cells, eye_data, CS_data, SS_final=None):
        self.weights = np.zeros((len(granule_cells), )) # One weight for each granule cell
        self.eye_data = eye_data
        self.CS_data = CS_data
