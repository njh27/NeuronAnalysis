import numpy as np
from random import choices
from NeuronAnalysis.fit_NN_model import FitNNModel
from NeuronAnalysis.general import bin_data



def relu(x, knee):
    return np.maximum(0., x - knee)

def relu_refl(x, knee):
    return np.maximum(0., knee - x)


def GC_activations(granule_cells, eye_data):
    # Compute the activations of each granule cell across the 2D eye data inputs
    gc_activations = np.zeros((eye_data.shape[0], len(granule_cells)))
    for gc_ind, gc in enumerate(granule_cells):
        gc_activations[:, gc_ind] += gc.response(eye_data[:, 0], eye_data[:, 1])
        # gc_activations[:, gc_ind] += 0.2 * gc.response(eye_data[:, 2], eye_data[:, 3])

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

    def fit_relu_GCs(self, granule_cells, bin_width=10, bin_threshold=5, fit_avg_data=False, quick_lag_step=10, 
                     adjust_block_data=None):
        """ Fits the input eye data to the input Neuron according to the activations 
        specified by the input granule_cells using a perceptron neural network.
        """
        self.pf_lag, self.mli_lag = self.get_lags_kinematic_fit(quick_lag_step=quick_lag_step)
                
        # First get all firing rate data, bin and format
        if fit_avg_data:
            self.avg_trial_sets = ["pursuit", "anti_pursuit", "learning", "anti_learning", "instruction"]
            self.avg_block = "StabTunePre"
            firing_rate = []
            all_t_inds = []
            bin_eye_data = []
            for t_set in self.avg_trial_sets:
                if adjust_block_data is not None:
                    fr, ati = self.neuron.get_firing_traces_block_adj(self.time_window, self.avg_block, 
                                                        t_set, adjust_block_data, fix_time_window=[-300, 0], 
                                                        bin_width=bin_width, bin_threshold=bin_threshold, 
                                                        lag_range_eye=self.lag_range_pf, quick_lag_step=quick_lag_step, 
                                                        return_inds=True)
                else:
                    fr, ati = self.neuron.get_firing_traces(self.time_window, self.avg_block,
                                                                            t_set, return_inds=True)
                if fr.size == 0:
                    # No data for this so skip
                    continue
                firing_rate.append(np.nanmean(fr, axis=0))
                all_t_inds.append(ati)

                # Now get matching eye data
                eye_data = self.get_eye_data_traces(self.avg_block, ati, self.mli_lag)
                # Now bin and reshape eye data
                bin_eye_data.append(self.get_binned_eye_data(eye_data, bin_width, bin_threshold, True))

            firing_rate = np.vstack(firing_rate)
            if len(firing_rate) == 0:
                raise ValueError("No trial data found for input blocks and trial sets.")
            # Set fit_avg_data FALSE here since we already took the average and it will average over trial sets
            binned_FR = self.get_binned_FR_data(firing_rate, bin_width, bin_threshold, False)
            bin_eye_data = np.vstack(bin_eye_data)
        else:
            if adjust_block_data is not None:
                firing_rate, all_t_inds = self.neuron.get_firing_traces_block_adj(self.time_window, self.blocks, 
                                                    self.trial_sets, adjust_block_data, fix_time_window=[-300, 0], 
                                                    bin_width=bin_width, bin_threshold=bin_threshold, 
                                                    lag_range_eye=self.lag_range_pf, quick_lag_step=quick_lag_step, 
                                                    return_inds=True)
            else:
                firing_rate, all_t_inds = self.neuron.get_firing_traces(self.time_window, self.blocks,
                                                                        self.trial_sets, return_inds=True)
            if len(firing_rate) == 0:
                raise ValueError("No trial data found for input blocks and trial sets.")
            binned_FR = self.get_binned_FR_data(firing_rate, bin_width, bin_threshold, fit_avg_data)
            # Finally get the eye data at lag mli_lag for the matching firing rate trials
            eye_data = self.get_eye_data_traces(self.blocks, all_t_inds, self.mli_lag)
            # Now bin and reshape eye data
            bin_eye_data = self.get_binned_eye_data(eye_data, bin_width, bin_threshold, fit_avg_data)
        
        # Now get all valid firing rate and eye data by removing nans
        FR_select = ~np.isnan(binned_FR)
        select_good= np.logical_and(~np.any(np.isnan(bin_eye_data), axis=1), FR_select)
        bin_eye_data = bin_eye_data[select_good, :]
        binned_FR = binned_FR[select_good]

        # Compute the activations of each granule cell across the eye data inputs
        gc_activations = GC_activations(granule_cells, bin_eye_data)

        # Add column of 1's
        gc_activations = np.hstack((gc_activations, np.ones((gc_activations.shape[0], 1))))
        # And do regression
        coefficients, ssr, _, _ = np.linalg.lstsq(gc_activations, binned_FR, rcond=None)
        coeffs = coefficients[0:-1]
        bias = coefficients[-1]

        # Store this for now so we can call predict_gauss_basis_kinematics
        # below for computing R2.
        self.fit_results['relu_GCs'] = {
                                        'coeffs': coeffs,
                                        'bias': bias,
                                        'granule_cells': granule_cells,
                                        'R2': None,
                                        }
        y_predicted = self.predict_relu_GCs(bin_eye_data)
        mean_rate = np.nanmean(binned_FR, axis=0)
        sum_squares_error = np.nansum((binned_FR - y_predicted) ** 2)
        sum_squares_total = np.nansum((binned_FR - mean_rate) ** 2)
        self.fit_results['relu_GCs']['R2'] = 1 - sum_squares_error/(sum_squares_total)
        print(f"Fit R2 = {self.fit_results['relu_GCs']['R2']} with SSE {sum_squares_error} of {sum_squares_total} total.")

        return
    
    def get_relu_GCs_predict_data_trial(self, blocks, trial_sets,
                                                      return_shape=False,
                                                      return_inds=False):
        """ Gets behavioral data from blocks and trial sets and formats in a
        way that it can be used to predict firing rate according to the model.
        Data are only retrieved for trials that are valid for the fitted neuron. """
        trial_sets = self.neuron.append_valid_trial_set(trial_sets)
        eye_data, t_inds = self.get_eye_data_traces(blocks, trial_sets,
                                                    self.mli_lag, return_inds=True)
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
    
    def get_relu_GCs_predict_data_mean(self, blocks, trial_sets):
        """ Gets behavioral data from blocks and trial sets and formats in a
        way that it can be used to predict firing rate according to the linear
        eye kinematic model using predict_lin_eye_kinematics.
        Data for predictions are retrieved only for valid neuron trials."""
        lag_win = [self.time_window[0] + self.mli_lag,
                   self.time_window[1] + self.mli_lag]
        trial_sets = self.neuron.append_valid_trial_set(trial_sets)
        X = np.ones((self.time_window[1]-self.time_window[0], 4))
        X[:, 0], X[:, 1] = self.neuron.session.get_mean_xy_traces(
                                                "eye position", lag_win,
                                                blocks=blocks,
                                                trial_sets=trial_sets)
        X[:, 2], X[:, 3] = self.neuron.session.get_mean_xy_traces(
                                                "eye velocity", lag_win,
                                                blocks=blocks,
                                                trial_sets=trial_sets)
        return X

    def predict_relu_GCs(self, X):
        """
        """
        if X.shape[1] != 4:
            raise ValueError("Relu granule cell model is fit for 4 data dimensions but input data dimension is {0}.".format(X.shape[1]))
        X_input = GC_activations(self.fit_results['relu_GCs']['granule_cells'], X)
        W = self.fit_results['relu_GCs']['coeffs']
        b = self.fit_results['relu_GCs']['bias']
        y_hat = np.dot(X_input, W) + b
        return y_hat

    def predict_relu_GCs_by_trial(self, blocks, trial_sets):
        """
        """
        X, init_shape = self.get_relu_GCs_predict_data_trial(
                                blocks, trial_sets, return_shape=True)
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
    def __init__(self, mossy_fibers, mf_weights, activation="relu", threshold=0.):
        """ Construct response function given the input parameters. Note that negative thresholds
        will "work" but should yield tonic activity since all weights and inputs are positive.
        """
        if activation == "relu":
            self.act_fun = relu
        self.mfs = mossy_fibers
        self.mf_weights = mf_weights 
        self.threshold = threshold

    def response(self, h, v, threshold=0.):
        """ Returns the response of this granule cell given vectors of horizontal and vertical inputs
        by summing the response over its mossy fiber inputs. """
        # Threshold needs subtracted so use negative
        output = np.full((h.shape[0], ), -1*self.threshold)
        for mf_ind, mf in enumerate(self.mfs):
            output += mf.response(h, v) * self.mf_weights[mf_ind]
        output = self.act_fun(output, 0.)
        return output
    

def make_mossy_fibers(N, knee_win=[-10, 10]):
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
    # n_mfs = np.random.randint(3, 6, size=N)
    # All the weights and thresholds are positive
    thresholds = np.abs(np.random.normal(1.5, 0.5, N))
    for n_gc in range(0, N):   
        # Choose a set of mossy fibers and some random weights
        mf_in = choices(mossy_fibers, k=4)
        mf_weights = np.random.uniform(size=len(mf_in))
        granule_cells.append(GranuleCell(mf_in, mf_weights, threshold=thresholds[n_gc]))

    return granule_cells
