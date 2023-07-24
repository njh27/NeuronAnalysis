import numpy as np
import h5py
import tensorflow as tf
from tensorflow.keras import layers, models, constraints, initializers
from tensorflow.keras.optimizers import SGD
from random import choices
from NeuronAnalysis.fit_NN_model import FitNNModel



def relu(x, knee):
    return np.maximum(0., x - knee)

def relu_refl(x, knee):
    return np.maximum(0., knee - x)


def GC_activations(granule_cells, t, threshold=0.):
    # Compute the activations of each granule cell for time/trial index "t"
    gc_activations = []
    for gc in granule_cells:
        gc_activations.append(gc.response(t, threshold=threshold))
    gc_activations = np.column_stack(gc_activations)
    gc_means = np.nanmean(gc_activations, axis=0)
    gc_stds = np.nanstd(gc_activations, axis=0)
    gc_activations = (gc_activations - gc_means[None, :]) / gc_stds[None, :]
    return gc_activations


class timeFitGCtoPC(object):
    """ A class for fitting  the PC neural network using the granule cell input. This class
    fits in the simplest possible way, using ONLY granule cell input (no basket inhibition)
    and allowing negative weights. There are NO LAGS used here either and fits average traces.
    """
    def __init__(self, Neuron, time_window=[0, 800], block=None):
        self.neuron = Neuron
        self.time_window = time_window
        self.block = block
        self.fit_results = {}

    def fit_time_GCs(self, granule_cells, activation_out="relu", intrinsic_rate0=None, 
                     learning_rate=0.01, epochs=200, batch_size=None):
        """ Fits the input eye data to the input Neuron according to the activations 
        specified by the input granule_cells using a perceptron neural network.
        """
        firing_rate = []
        for cond in ["pursuit", "anti_pursuit", "learning", "anti_learning"]:
            firing_rate.append(self.neuron.get_mean_firing_trace(self.time_window, 
                                                            self.block, 
                                                            trial_sets=cond))
        firing_rate = np.hstack(firing_rate)
        if len(firing_rate) == 0:
            raise ValueError("No trial data found for input blocks and trial sets.")
        
        # Now get all valid firing rate and eye data by removing nans
        FR_select = ~np.isnan(firing_rate)
        # Compute the activations of each granule cell across the eye data inputs
        gc_activations = GC_activations(granule_cells, slice(None), threshold=0.)

        # Remove any possible NaN values
        firing_rate = firing_rate[FR_select]
        firing_rate = (firing_rate - np.nanmean(firing_rate)) / np.nanstd(firing_rate)
        gc_activations = gc_activations[FR_select]

        self.activation_out = activation_out
        if intrinsic_rate0 is None:
            intrinsic_rate0 = 0.8 * np.nanmedian(firing_rate)
        # Create the neural network model
        model = models.Sequential([
            layers.Input(shape=(gc_activations.shape[1],)),
            layers.Dense(1, activation=activation_out,
                         kernel_initializer=initializers.RandomNormal(mean=0., stddev=5.),
                         bias_initializer=initializers.Constant(intrinsic_rate0)),
        ])
        clip_value = None
        optimizer = SGD(learning_rate=learning_rate, clipvalue=clip_value)
        optimizer_str = "SGD"

        # Compile the model
        model.compile(optimizer=optimizer_str, loss='mean_squared_error')

        # Train the model
        history = model.fit(gc_activations, firing_rate, epochs=epochs, batch_size=batch_size,
                                        validation_data=None, verbose=0)
        train_loss = history.history['loss']

        # Store this for now so we can call prediction funs
        # below for computing R2.
        self.fit_results['time_GCs'] = {
                                        'coeffs': model.layers[0].get_weights()[0],
                                        'bias': model.layers[0].get_weights()[1],
                                        'granule_cells': granule_cells,
                                        'R2': None,
                                        'train_loss': train_loss,
                                        }
        
        # Compute R2
        y_predicted = self.predict_time_GCs(slice(None))
        mean_rate = np.nanmean(firing_rate)
        sum_squares_error = np.nansum((mean_rate - y_predicted) ** 2)
        sum_squares_total = np.nansum((mean_rate - firing_rate) ** 2)
        self.fit_results['time_GCs']['R2'] = 1 - sum_squares_error/(sum_squares_total)
        print(f"Fit R2 = {self.fit_results['time_GCs']['R2']} with SSE {sum_squares_error} of {sum_squares_total} total.")

        return


    def predict_time_GCs(self, t):
        """
        """
        X_input = GC_activations(self.fit_results['time_GCs']['granule_cells'], t, threshold=0.)
        W = self.fit_results['time_GCs']['coeffs']
        b = self.fit_results['time_GCs']['bias']
        y_hat = np.dot(X_input, W) + b
        if self.activation_out == "relu":
            y_hat = np.maximum(0, y_hat)
        return y_hat


class MossyFiber(object):
    """ Very simple mossy fiber class defined by input response fun vector for using based on
    actual recording mossy fiber responses as inputs
    """
    def __init__(self, response_fun):
        """ Construct response function given the input parameters. """
        self.resp_fun = response_fun / np.amax(response_fun)

    def response(self, t):
        """ Returns the response of this mossy fiber at a given time point/trial index "t". """
        return self.resp_fun[t]
    

class GranuleCell(object):
    """ Granule cell class that gets inputs from mossy fibers and computes activation response
    """
    def __init__(self, mossy_fibers, mf_weights):
        """ Construct response function given the input parameters. """
        self.mfs = mossy_fibers
        # Normalize the sum of mf weights to 1 to make sure granule cell can be activated
        self.mf_weights = mf_weights / np.sum(mf_weights)

    def response(self, t, threshold=0.):
        """ Returns the response of this granule cell at time point/trial index "t". """
        # Threshold needs subtracted
        threshold *= -1.
        output = []
        for mf_ind, mf in enumerate(self.mfs):
            output.append(mf.response(t) * self.mf_weights[mf_ind])
        for mf_ind in range(1, len(output)):
            output[0] += output[mf_ind]
        return output[0]
    

def make_mossy_fibers(mf_H5_file):
    """ Makes mossy fibers by permuting the responses of the actual mossy fibers input in the
    H5 file found in mf_H5_file.
    """
    mf_file = h5py.File(mf_H5_file, 'r')
    mossy_fibers = []
    # For each mf
    for mf_name in mf_file['mf'].keys():
        # For each pursuit direction
        mf_responses = []
        for direction in ["contra", "ipsi", "up", "down"]:
            mf_responses.append(np.nanmean(np.array(mf_file['mf'][mf_name][direction]['firing_rate']), axis=1))
        mossy_fibers.append(MossyFiber(np.hstack(mf_responses)))
        # Now repeat for ipsi/contra flipped
        mf_responses = []
        for direction in ["ipsi", "contra", "up", "down"]:
            mf_responses.append(np.nanmean(np.array(mf_file['mf'][mf_name][direction]['firing_rate']), axis=1))
        mossy_fibers.append(MossyFiber(np.hstack(mf_responses)))
        # Now repeat for up/down flipped
        mf_responses = []
        for direction in ["contra", "ipsi", "down", "up"]:
            mf_responses.append(np.nanmean(np.array(mf_file['mf'][mf_name][direction]['firing_rate']), axis=1))
        mossy_fibers.append(MossyFiber(np.hstack(mf_responses)))
        # Now repeat for both axes flipped
        mf_responses = []
        for direction in ["ipsi", "contra", "down", "up"]:
            mf_responses.append(np.nanmean(np.array(mf_file['mf'][mf_name][direction]['firing_rate']), axis=1))
        mossy_fibers.append(MossyFiber(np.hstack(mf_responses)))

    return mossy_fibers


def make_granule_cells(N, mossy_fibers):
    """ Make granule cells by choosing 4 mossy fibers from "mossy_fibers" and combining them
    with random weights. """
    granule_cells = []
    for _ in range(0, N):   
        # Choose a set of mossy fibers and some random weights
        mf_in = choices(mossy_fibers, k=4)
        mf_weights = np.random.uniform(size=len(mf_in))
        granule_cells.append(GranuleCell(mf_in, mf_weights))

    return granule_cells
