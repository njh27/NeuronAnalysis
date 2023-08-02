import numpy as np
from scipy.optimize import differential_evolution
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
    granule_activations[np.isnan(granule_activations)] = 0.
    weights = np.copy(weights_0)
    for trial in range(0, granule_activations.shape[0]):
        CS_trial = np.copy(CS[trial, :])
        CS_trial = box_windows(CS_trial, -50, 50, scale=1.0)[None, :]
        LTD = CS_trial @ granule_activations[trial, :, :]
        LTD /= CS_trial.size
        LTD *= model_params['epsilon']
        LTD *= (model_params['w_min'] - weights)
        
        # Then LTP
        CS_trial_inv = CS_trial > 0.
        CS_trial[CS_trial_inv] = 0.
        CS_trial[~CS_trial_inv] = 1.
        LTP = CS_trial @ granule_activations[trial, :, :]
        LTP /= CS_trial.size
        LTP *= model_params['alpha'] 
        LTP *= (model_params['w_max'] - weights)

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
        trial_gc_activation[nan_inds, :] = 0.

        # Setting to intrinsic rate will make this equal to y_hat_trial and not affect residuals
        trial_SS[nan_inds] = intrinsic_rate
        y_hat_trial = weights @ trial_gc_activation.T + intrinsic_rate

        if np.any(np.isnan(y_hat_trial)):
            print("NAN IN Y HAT")
        if np.any(np.isnan(trial_SS)):
            print("NAN IN INPUT SS")

        # Store requested outputs as needed
        if return_residuals:
            # Add residuals for current trial
            residuals_trial = np.sum((y_hat_trial - trial_SS) ** 2) / trial_n_obs
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
    model_const = args[1]
    t_set_select = args[2]

    # Dictionary of all possible parameters for learning model set to dummy
    # null values that will have no effect on learning model
    model_params = {"alpha": 0.0,
                    "epsilon": 0.0,
                    "intrinsic_rate": 0.0,
                    "w_max": 100.,
                    "w_min": -100.,
                    }
    # Build dictionary of params being fit to pass to learning function
    # according to the initialization dictionary param_conds
    for p in param_conds.keys():
        model_params[p] = params[param_conds[p][3]]
        # print(f"{p}: {model_params[p]}")
    # Compute the weights given the current model parameters
    weights_0 = np.zeros((1, granule_activations.shape[2]))
    weights = run_learning_model(weights_0, granule_activations, CS, model_params)
    if len(t_set_select) > 0:
        granule_activations, SS = convert_predict_to_mean(granule_activations, SS, t_set_select)
    residuals = predict_learning_model(weights, granule_activations, model_params['intrinsic_rate'], 
                                       SS, return_residuals=True, return_y_hat=False)
    # print(f"Residuals {residuals}")
    return residuals


def convert_predict_to_mean(granule_activations, SS, t_set_select):
    """ Gets the means of activations and SS
    """
    ga_out = np.zeros((len(t_set_select), granule_activations.shape[1], granule_activations.shape[2]))
    ss_out = np.zeros((len(t_set_select), SS.shape[1]))
    for t_ind, t_set in enumerate(t_set_select.keys()):
        ga_out[t_ind, :, :] = np.nanmean(granule_activations[t_set_select[t_set], :, :], axis=0)
        ss_out[t_ind, :] = np.nanmean(SS[t_set_select[t_set], :])
    return ga_out, ss_out

def init_learn_fit_params():
    """
    """
    # Format of p0, lower, upper,
    param_conds = {"alpha": (1e-7, 0.0, 1e-2),
                   "epsilon": (1e-5, 0.0, 1e-2),
                   "intrinsic_rate": (0., -np.inf, np.inf),
                   "w_max": (1., 0.5, 100),
                   "w_min": (-1., -100, -0.5),
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


def compute_response(eye_data, granule_cells, weights):
    """
    """
    gc_activations = eye_to_gc_activations(eye_data, granule_cells)
    response = None


def fit_plastic_tuning_model(gc_activations, SS, CS, t_set_select={}):
    """
    """
    param_conds, p0, lower_bounds, upper_bounds = init_learn_fit_params()
    model_const = {}
    lf_args = (param_conds, model_const, t_set_select)
    # Note that differential_evolution() does not allow method specification
    # for the minimization step because it has its own mechanism.
    # We now define the bounds as a list of (min, max) pairs for each element in x
    bounds = [(lb, ub) for lb, ub in zip(lower_bounds, upper_bounds)]
    # differential_evolution function takes the objective function and the bounds as main arguments.
    result = differential_evolution(func=obj_learning_model,
                                    bounds=bounds,
                                    args=(gc_activations, SS, CS, *lf_args),
                                    workers=4, updating='deferred', popsize=12,
                                    disp=True) # Display status messages

    return result


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

    def run_sim(alpha, epsilon, w_max, w_min):
        """
        """
        model_params = {'alpha': alpha,
                        'epsilon': epslilon,
                        'w_max': w_max,
                        'w_min': w_min,
                        }