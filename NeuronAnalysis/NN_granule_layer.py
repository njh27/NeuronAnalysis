import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, constraints, initializers
from tensorflow.keras.optimizers import SGD
from random import choices
from NeuronAnalysis.fit_NN_model import FitNNModel
from NeuronAnalysis.general import bin_data



def relu(x, knee):
    return np.maximum(0., x - knee)

def relu_refl(x, knee):
    return np.maximum(0., knee - x)


class FitGCtoPC(FitNNModel):
    """ A class for fitting  the PC neural network using the granule cell input. This class
    fits in the simplest possible way, using ONLY granule cell input (no basket inhibition)
    and allowing negative weights. There are NO LAGS used here either.
    """
    def __init__(self, Neuron, time_window=[0, 800], blocks=None, trial_sets=None,
                    lag_range_pf=[-50, 150], use_series=None):
        super().__init__(Neuron, time_window, blocks, trial_sets, lag_range_pf, use_series)

    def fit_relu_GCs(self, granule_cells, activation_out="relu", intrinsic_rate0=None, 
                     bin_width=10, bin_threshold=5, fit_avg_data=False, quick_lag_step=10, 
                     train_split=1.0, learning_rate=0.02, epochs=200, batch_size=None, 
                     adjust_block_data=None):
        """ Fits the input eye data to the input Neuron according to the activations 
        specified by the input granule_cells using a perceptron neural network.
        """
        if train_split > 1.0:
            raise ValueError("Proportion to fit 'train_split' must be a value less than 1!")

        # Setup all the indices for which trials we will be using and which
        # subset of trials will be used as training vs. test data
        if adjust_block_data is not None:
            firing_rate, all_t_inds = self.get_block_adj_firing_trace(adjust_block_data, fix_time_window=[-300, 0], 
                                                                    bin_width=bin_width, bin_threshold=bin_threshold, 
                                                                    quick_lag_step=quick_lag_step, return_inds=True)
        else:
            firing_rate, all_t_inds = self.get_firing_traces(return_inds=True)
        if len(firing_rate) == 0:
            raise ValueError("No trial data found for input blocks and trial sets.")
        
        # Get indices for training and testing data sets
        select_fit_trials, test_trial_set, train_trial_set, is_test_data = self.split_test_train(
                                                                                all_t_inds, train_split)
        # First get all firing rate data, bin and format
        binned_FR_train, binned_FR_test = self.get_binned_FR_data(firing_rate, select_fit_trials, bin_width, 
                                                                  bin_threshold, fit_avg_data)

        # Finally get the eye data at lag 0 for the matching firing rate trials
        eye_data = self.get_eye_data_traces(self.blocks, all_t_inds, 0)
        # Now bin and reshape eye data
        bin_eye_data_train, bin_eye_data_test = self.get_binned_eye_data(eye_data, select_fit_trials, bin_width, 
                                                                         bin_threshold, fit_avg_data)
        
        # Now get all valid firing rate and eye data by removing nans
        FR_select_train = ~np.isnan(binned_FR_train)
        select_good_train = np.logical_and(~np.any(np.isnan(bin_eye_data_train), axis=1), FR_select_train)
        bin_eye_data_train = bin_eye_data_train[select_good_train, :]
        binned_FR_train = binned_FR_train[select_good_train]
        FR_select_test = ~np.isnan(binned_FR_test)
        select_good_test = np.logical_and(~np.any(np.isnan(bin_eye_data_test), axis=1), FR_select_test)
        bin_eye_data_test = bin_eye_data_test[select_good_test, :]
        binned_FR_test = binned_FR_test[select_good_test]

class MossyFiber(object):
    """ Very simple mossy fiber class that defines a 2D response profile from the relu functions
    """
    def __init__(self, h_knee, v_knee, h_refl, v_refl, h_weight=1., v_weight=1.):
        """ Construct response function given the input parameters. """
        self.h_knee = h_knee
        self.v_knee = v_knee
        self.h_refl = h_refl
        self.v_refl = v_refl
        self.h_weight = h_weight
        self.v_weight = v_weight
        self.h_fun = relu_refl if self.h_refl else relu
        self.v_fun = relu_refl if self.v_refl else relu

    def response(self, h, v):
        """ Returns the response of this mossy fiber given vectors of horizontal and vertical inputs. """
        output = self.h_fun(h, self.h_knee) * self.h_weight
        output += self.v_fun(v, self.v_knee) * self.v_weight
        return output
    

class GranuleCell(object):
    """ Granule cell class that gets inputs from mossy fibers and computes activation response
    """
    def __init__(self, mossy_fibers, mf_weights, activation="relu"):
        """ Construct response function given the input parameters. """
        if activation == "relu":
            self.act_fun = relu
        self.mfs = mossy_fibers
        # Normalize the sum of mf weights to 1 to make sure granule cell can be activated
        self.mf_weights = mf_weights / np.sum(mf_weights)

    def response(self, h, v):
        """ Returns the response of this granule cell given vectors of horizontal and vertical inputs
        by summing the response over its mossy fiber inputs. """
        output = np.zeros(h.shape[0])
        for mf_ind, mf in enumerate(self.mfs):
            output += mf.response(h, v) * self.mf_weights[mf_ind]
        output = self.act_fun(output, 0.)
        return output
    

def make_mossy_fibers(N, knee_win=[-30, 30]):
    """ Makes N mossy fibers for both the horizontal and vertical axis with random response parameters
    and an even distribution of positive and negative sloped activation functions.
    """
    mossy_fibers = []
    # Get random numbers first for speed
    knees = np.random.uniform(knee_win[0], knee_win[1], N)
    is_refl = np.random.choice([True, False], size=N)
    # weights = np.random.uniform(size=(N, 2))
    for n_mf in range(0, N):
        # Make a horizontal and vertical mossy fiber each iteration
        mossy_fibers.append(MossyFiber(knees[n_mf], 0., is_refl[n_mf], False, h_weight=1., v_weight=0.))
        mossy_fibers.append(MossyFiber(0., knees[n_mf], False, is_refl[n_mf], h_weight=0., v_weight=1.))

    return mossy_fibers


def make_granule_cells(N, mossy_fibers):
    """ Make granule cells by choosing 3-5 mossy fibers from "mossy_fibers" and combining them
    with random weights. """
    granule_cells = []
    # Get some random numbers up front
    n_mfs = np.random.randint(3, 6, size=N)
    for n_gc in range(0, N):   
        # Choose a set of mossy fibers and some random weights
        mf_in = choices(mossy_fibers, k=n_mfs[n_gc])
        mf_weights = np.random.uniform(size=len(mf_in))
        granule_cells.append(GranuleCell(mf_in, mf_weights))

    return granule_cells
