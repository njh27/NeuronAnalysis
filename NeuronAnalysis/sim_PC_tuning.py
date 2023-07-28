import numpy as np
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



import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
def run_learning_model(weights_0, granule_activations, CS, model_params):
    """
    """
    intrinsic_rate = 1.
    show_plot = False
    shuffle_CSs = False
    granule_activations[np.isnan(granule_activations)] = 0.

    # Define the order of the filter and the cut-off frequency.
    order = 2
    cut_off_frequency = .25/500
    # Design the high-pass Butterworth filter.
    b, a = butter(order, cut_off_frequency, btype='high')
    filter_gc_activations = np.zeros_like(granule_activations)
    # Apply the filter to each column of the data.
    for i in range(0, granule_activations.shape[2]):
        for t in range(0, granule_activations.shape[0]):
            # filter_gc_activations[t, :, i] = filtfilt(b, a, granule_activations[t, :, i]) + np.nanmean(granule_activations[t, :, i])
            # filter_gc_activations[t, :, i] = filter_gc_activations[t, :, i] = high_pass_two_state(granule_activations[t, :, i], 2, 4, 1000)
            filter_gc_activations[t, :, i] = decay_fun(granule_activations[t, :, i], 600, threshold=10)
    granule_activations = np.maximum(filter_gc_activations, 0.)

    weights = np.copy(weights_0)
    # for trial in range(0, granule_activations.shape[0]):
    # rand_t_inds = np.random.randint(0, granule_activations.shape[0], 1000)
    # for trial in rand_t_inds:
    for trial in range(0, granule_activations.shape[0]):
        # trial_rates = weights @ granule_activations[trial, :, :].T + intrinsic_rate
        # Compute LTD first
        if shuffle_CSs:
            CS_trial = shuffle_CS(CS[trial, :], -100, 100)
        else:
            CS_trial = CS[trial, :]
        CS_trial = box_windows(CS_trial, -50, 50, scale=1.0)[None, :]
        # CS_trial *= trial_rates
        LTD = CS_trial @ granule_activations[trial, :, :]
        # LTD[LTD > 0] = 1.
        LTD *= model_params['epsilon']
        LTD *= (model_params['w_min'] - weights)

        if show_plot:
            plt.plot(10*CS_trial.T, color='red')
            plt.plot(granule_activations[trial, :, :])
            plt.show()
        
        # Then LTP
        CS_trial_inv = CS_trial > 0.
        CS_trial[CS_trial_inv] = 0.
        CS_trial[~CS_trial_inv] = 1.
        # CS_trial *= trial_rates
        LTP = CS_trial @ granule_activations[trial, :, :]
        LTP *= model_params['alpha'] 
        LTP *= (model_params['w_max'] - weights)

        # Then add them up and adjust weights
        delta_w = LTP + LTD
        weights += delta_w

        

    return weights


def compute_response(eye_data, granule_cells, weights):
    """
    """
    gc_activations = eye_to_gc_activations(eye_data, granule_cells)
    response = None


class SimPCTuning(object):
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