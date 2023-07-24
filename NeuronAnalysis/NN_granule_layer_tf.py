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


def GC_activations(granule_cells, eye_data, threshold=0.):
    # Compute the activations of each granule cell across the 2D eye data inputs
    gc_activations = np.zeros((eye_data.shape[0], len(granule_cells)))
    for gc_ind, gc in enumerate(granule_cells):
        gc_activations[:, gc_ind] = gc.response(eye_data[:, 0], eye_data[:, 1], threshold=threshold)

    # t = np.arange(0, gc_activations.shape[0])
    # phases = np.random.uniform(0, 2*np.pi, gc_activations.shape[1])
    # frequencies = np.random.uniform(8, 20, gc_activations.shape[1]) / 10000
    # for golgi_ind in range(0, gc_activations.shape[1]):
    #     sine_wave = 20 * np.sin(2*np.pi*t*frequencies[golgi_ind] + phases[golgi_ind])
    #     rectified_sine_wave = np.maximum(sine_wave, 0)
    #     gc_activations[:, golgi_ind] -= rectified_sine_wave
    #     gc_activations[:, golgi_ind] = np.maximum(0, gc_activations[:, golgi_ind])

    return gc_activations


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

        # Compute the activations of each granule cell across the eye data inputs
        gc_activations_train = GC_activations(granule_cells, bin_eye_data_train, threshold=0.)
        if is_test_data:
            gc_activations_test = GC_activations(granule_cells, bin_eye_data_test, threshold=0.)
            val_data = (gc_activations_test, binned_FR_test)
        else:
            gc_activations_test = []
            val_data = None
        if np.any(np.any(np.isnan(gc_activations_train))):
            raise ValueError("Nan in here")
        self.activation_out = activation_out
        if intrinsic_rate0 is None:
            intrinsic_rate0 = 0.8 * np.nanmedian(binned_FR_train)
        # Create the neural network model
        model = models.Sequential([
            layers.Input(shape=(gc_activations_train.shape[1],)),
            layers.Dense(1, activation=activation_out,
                         kernel_initializer=initializers.RandomNormal(mean=0., stddev=1.),
                         bias_initializer=initializers.Constant(intrinsic_rate0)),
        ])
        clip_value = None
        optimizer = SGD(learning_rate=learning_rate, clipvalue=clip_value)
        optimizer_str = "SGD"

        # Compile the model
        model.compile(optimizer=optimizer_str, loss='mean_squared_error')

        # Train the model
        if is_test_data:
            val_data = (gc_activations_test, binned_FR_test)
            test_data_only = True
        else:
            val_data = None
            test_data_only = False
        if np.any(np.any(np.isnan(gc_activations_train))):
            raise ValueError("Nans in GC activation!")
        if np.any(np.any(np.isnan(binned_FR_train))):
            raise ValueError("Nans in FR data!")
        history = model.fit(gc_activations_train, binned_FR_train, epochs=epochs, batch_size=batch_size,
                                        validation_data=val_data, verbose=0)
        if is_test_data:
            test_loss = history.history['val_loss']
        else:
            test_loss = None
        train_loss = history.history['loss']

        # Store this for now so we can call predict_gauss_basis_kinematics
        # below for computing R2.
        self.fit_results['relu_GCs'] = {
                                        'coeffs': model.layers[0].get_weights()[0],
                                        'bias': model.layers[0].get_weights()[1],
                                        'granule_cells': granule_cells,
                                        'R2': None,
                                        'is_test_data': is_test_data,
                                        'test_trial_set': test_trial_set,
                                        'train_trial_set': train_trial_set,
                                        'test_loss': test_loss,
                                        'train_loss': train_loss,
                                        }
        
        # Compute R2
        if self.fit_results['relu_GCs']['is_test_data']:
            test_firing_rate = firing_rate[~select_fit_trials, :]
        else:
            # If no test data are available, you need to just compute over all data
            test_firing_rate = firing_rate[select_fit_trials, :]
        if fit_avg_data:
            test_lag_data = self.get_relu_GCs_predict_data_mean(self.blocks, self.trial_sets, 
                                                                test_data_only=test_data_only)
            y_predicted = self.predict_relu_GCs(test_lag_data)
            test_mean_rate = np.nanmean(test_firing_rate, axis=0, keepdims=True)
            sum_squares_error = np.nansum((test_mean_rate - y_predicted) ** 2)
            sum_squares_total = np.nansum((test_mean_rate - np.nanmean(test_mean_rate)) ** 2)
        else:
            y_predicted = self.predict_relu_GCs_by_trial(self.blocks, self.trial_sets, 
                                                         test_data_only=test_data_only)
            sum_squares_error = np.nansum((test_firing_rate - y_predicted) ** 2)
            sum_squares_total = np.nansum((test_firing_rate - np.nanmean(test_firing_rate)) ** 2)
        self.fit_results['relu_GCs']['R2'] = 1 - sum_squares_error/(sum_squares_total)
        print(f"Fit R2 = {self.fit_results['relu_GCs']['R2']} with SSE {sum_squares_error} of {sum_squares_total} total.")

        return
    
    def get_relu_GCs_predict_data_trial(self, blocks, trial_sets,
                                                      return_shape=False,
                                                      test_data_only=True,
                                                      return_inds=False):
        """ Gets behavioral data from blocks and trial sets and formats in a
        way that it can be used to predict firing rate according to the model.
        Data are only retrieved for trials that are valid for the fitted neuron. """
        trial_sets = self.neuron.append_valid_trial_set(trial_sets)
        if test_data_only:
            if self.fit_results['relu_GCs']['is_test_data']:
                trial_sets = trial_sets + [self.fit_results['relu_GCs']['test_trial_set']]
            else:
                print("No test trials are available. Returning everything.")
        eye_data, t_inds = self.get_eye_data_traces(blocks, trial_sets,
                                                    0., return_inds=True)
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
    
    def get_relu_GCs_predict_data_mean(self, blocks, trial_sets, test_data_only=True):
        """ Gets behavioral data from blocks and trial sets and formats in a
        way that it can be used to predict firing rate according to the linear
        eye kinematic model using predict_lin_eye_kinematics.
        Data for predictions are retrieved only for valid neuron trials."""
        trial_sets = self.neuron.append_valid_trial_set(trial_sets)
        if test_data_only:
            if self.fit_results['relu_GCs']['is_test_data']:
                trial_sets = trial_sets + [self.fit_results['relu_GCs']['test_trial_set']]
            else:
                print("No test trials are available. Returning everything.")
        X = np.ones((self.time_window[1]-self.time_window[0], 4))
        X[:, 0], X[:, 1] = self.neuron.session.get_mean_xy_traces(
                                                "eye position", self.time_window,
                                                blocks=blocks,
                                                trial_sets=trial_sets)
        X[:, 2], X[:, 3] = self.neuron.session.get_mean_xy_traces(
                                                "eye velocity", self.time_window,
                                                blocks=blocks,
                                                trial_sets=trial_sets)
        return X

    def predict_relu_GCs(self, X):
        """
        """
        if X.shape[1] != 4:
            raise ValueError("Relu granule cell model is fit for 4 data dimensions but input data dimension is {0}.".format(X.shape[1]))
        X_input = GC_activations(self.fit_results['relu_GCs']['granule_cells'], X, threshold=0.)
        W = self.fit_results['relu_GCs']['coeffs']
        b = self.fit_results['relu_GCs']['bias']
        y_hat = np.dot(X_input, W) + b
        if self.activation_out == "relu":
            y_hat = np.maximum(0, y_hat)
        return y_hat

    def predict_relu_GCs_by_trial(self, blocks, trial_sets, test_data_only=True):
        """
        """
        X, init_shape = self.get_relu_GCs_predict_data_trial(
                                blocks, trial_sets, return_shape=True,
                                test_data_only=test_data_only)
        y_hat = self.predict_relu_GCs(X)
        y_hat = y_hat.reshape(init_shape[0], init_shape[1], order='C')
        return y_hat

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

    def response(self, h, v, threshold=0.):
        """ Returns the response of this granule cell given vectors of horizontal and vertical inputs
        by summing the response over its mossy fiber inputs. """
        # Threshold needs subtracted
        threshold *= -1.
        output = np.ones(h.shape[0]) * threshold
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
