import numpy as np
from NeuronAnalysis.NN_granule_layer import GC_activations



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

def eye_to_gc_activations(eye_data, granule_cells):
    """
    """
    eye_data_reshape = eye_data.shape
    eye_data = eye_data.reshape(eye_data.shape[0]*eye_data.shape[1], eye_data.shape[2], order='C')
    gc_activations = GC_activations(granule_cells, eye_data)
    gc_activations = gc_activations.reshape((eye_data_reshape[0], eye_data_reshape[1], gc_activations.shape[1]))
    return gc_activations


def run_learning_model(weights_0, granule_activations, CS, model_params):
    """
    """
    granule_activations[np.isnan(granule_activations)] = 0.
    weights = np.copy(weights_0)
    for trial in range(0, granule_activations.shape[0]):
        trial_CS_mod_fun = CS[trial, :][None, :] @ granule_activations[trial, :, :]
        LTP = model_params['alpha'] * trial_CS_mod_fun
        LTP *= (model_params['w_max'] - weights)
        LTD = model_params['epsilon'] * trial_CS_mod_fun
        LTD *= (model_params['w_min'] - weights)
        delta_w = LTP - LTD
        weights += delta_w

    return weights


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