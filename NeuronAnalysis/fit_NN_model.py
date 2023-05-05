import numpy as np
from scipy.optimize import least_squares
import tensorflow as tf
from tensorflow.keras import layers, models, constraints, initializers
from tensorflow.keras.optimizers import SGD
import warnings
from NeuronAnalysis.fit_neuron_to_eye import FitNeuronToEye



def bin_data(data, bin_width, bin_threshold=0):
    """ Gets the nan average of each bin in data for bins in which the number
        of non nan data points is greater than bin_threshold.  Bins less than
        bin threshold non nan data points are returned as nan. Data are binned
        from the first entries over time (axis=1), so if the number of bins
        implied by binwidth exceeds data.shape[1] the last bin will be cut short.
        Input data is assumed to have the shape as output by get_eye_data_traces,
        trial x time x variable and are binned along the time axis.
        bin_threshold must be <= bin_width. """
    if bin_threshold > bin_width:
        raise ValueError("bin_threshold cannot exceed the bin_width")
    if ( (bin_width < 1) or (not isinstance(bin_width, int)) ):
        raise ValueError("bin_width must be positive integer value")

    if data.ndim == 1:
        out_shape = (1, data.shape[0] // bin_width, 1)
        data = data.reshape(1, data.shape[0], 1)
    elif data.ndim == 2:
        out_shape = (data.shape[0], data.shape[1] // bin_width, 1)
        data = data.reshape(data.shape[0], data.shape[1], 1)
    elif data.ndim == 3:
        out_shape = (data.shape[0], data.shape[1] // bin_width, data.shape[2])
    else:
        raise ValueError("Unrecognized data input shape. Input data must be in the form as output by data functions.")
    if bin_width == 1:
        # Nothing to bin over time so just return possibly reshaped data COPY
        return np.copy(data)

    binned_data = np.full(out_shape, np.nan)
    n = 0
    bin_start = 0
    bin_stop = bin_start + bin_width
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        while n < out_shape[1]:
            n_good = np.count_nonzero(~np.isnan(data[:, bin_start:bin_stop, :]), axis=1)
            binned_data[:, n, :] = np.nanmean(data[:, bin_start:bin_stop, :], axis=1)
            binned_data[:, n, :][n_good < bin_threshold] = np.nan
            n += 1
            bin_start += bin_width
            bin_stop = bin_start + bin_width

    return binned_data


# Define Gaussian function
def gaussian(x, mu, sigma, scale):
    return scale * np.exp(-( ((x - mu) ** 2) / (2*(sigma**2))) )

def sigmoid(x, a=1, b=0, c=1, d=0):
    return c / (1 + np.exp(-(x-b)/a)) + d

def sigmoid_activation(x, fixed_scale, fixed_centers, fixed_asymptote=1, fixed_bias=0.0):
    num_sigmoids = len(fixed_centers)
    x_transform = np.zeros((x.size, num_sigmoids))
    for k in range(num_sigmoids):
        x_transform[:, k] = sigmoid(x, fixed_scale, fixed_centers[k], fixed_asymptote, fixed_bias)
    return x_transform

# Define the model function as a linear combination of Gaussian functions
def gaussian_activation(x, fixed_means, fixed_sigmas):
    num_gaussians = len(fixed_means)
    x_transform = np.zeros((x.size, num_gaussians))
    for k in range(num_gaussians):
        x_transform[:, k] = gaussian(x, fixed_means[k], fixed_sigmas[k], scale=1.0)
    return x_transform

def negative_relu(x, c=0.):
    """ Basic relu function but returns negative result. """
    return -1*np.maximum(0., x-c)

def reflected_negative_relu(x, c=0.):
    """ Basic relu function but returns negative result, reflected about y axis. """
    return np.minimum(0., x-c)


def gen_linspace_gaussians(max_min, n_gaussians, stds_gaussians):
    """ Returns means and standard deviations for the number of gaussians input
    in "n_gaussians" with centers linearly spaced over max_min. """
    if isinstance(stds_gaussians, np.ndarray):
        if len(stds_gaussians) == 1:
            stds_gaussians = np.full((n_gaussians, ), stds_gaussians[0])
        if len(stds_gaussians) != n_gaussians:
            raise ValueError("Input standard deviations must be same size as means or 1")
    elif isinstance(stds_gaussians, list):
        if len(stds_gaussians) == 1:
            stds_gaussians = np.full((n_gaussians, ), stds_gaussians[0])
        if len(stds_gaussians) != n_gaussians:
            raise ValueError("Input standard deviations must be same size as means or 1")
    else:
        # We suppose gauss stds is a single numeric value
        stds_gaussians = np.full((n_gaussians, ), stds_gaussians)
    try:
        if len(max_min) > 2:
            raise ValueError("max_min must be 1 or 2 elements specifing the max and min mean for the gaussians.")
    except TypeError:
        # Happens if max_min does not have "len" method, usually because it's a singe number
        max_min = [-1 * max_min, max_min]
    means_gaussians = np.linspace(max_min[0], max_min[1], n_gaussians)

    return means_gaussians, stds_gaussians


def gen_randuniform_gaussians(mean_max_min, std_max_min, n_gaussians):
    """ Returns means and standard deviations for the number of gaussians input
    in "n_gaussians" with centers chosen uniform randomly over mean_max_min and
    standard deviations selected uniform randomly over std_max_min. """
    try:
        if len(mean_max_min) > 2:
            raise ValueError("mean_max_min must be 1 or 2 elements specifing the max and min mean for the gaussian means.")
    except TypeError:
        # Happens if mean_max_min does not have "len" method, usually because it's a singe number
        mean_max_min = np.abs(mean_max_min)
        mean_max_min = [-1 * mean_max_min, mean_max_min]
    try:
        if len(std_max_min) > 2:
            raise ValueError("std_max_min must be 1 or 2 elements specifing the max and min mean for the gaussian STDs.")
    except TypeError:
        # Happens if mean_max_min does not have "len" method, usually because it's a singe number
        std_max_min = np.abs(std_max_min)
        std_max_min = [0.01, std_max_min]
    # STD must be > 0
    std_max_min[0] = max(0.01, std_max_min[0])
    means_gaussians = np.random.uniform(mean_max_min[0], mean_max_min[1], n_gaussians)
    stds_gaussians = np.random.uniform(std_max_min[0], std_max_min[1], n_gaussians)

    return means_gaussians, stds_gaussians


def eye_input_to_PC_gauss_relu(eye_data, gauss_means,
                                gauss_stds):
    """ Takes the total 8 dimensional eye data input (x,y position, and
    velocity times 2 lags) and converts it into the n_gaussians by 4 + 8 relu
    function input model of PC input. Done point by point for n x 4
    input "eye_data". """
    # Currently hard coded but could change in future
    n_eye_dims = 4
    n_eye_lags = 2
    n_total_eye_dims = n_eye_dims * n_eye_lags
    n_gaussians_per_dim = int(len(gauss_means) / n_eye_dims)
    if n_gaussians_per_dim < 1:
        raise ValueError("Not enough gaussian means input to cover {0} dimensions of eye data.".format(n_eye_dims))
    if len(gauss_means) != len(gauss_stds):
        raise ValueError("Must input the same number of means and standard deviations but got {0} means and {1} standard deviations.".format(len(gauss_means), len(gauss_stds)))
    n_features = len(gauss_means) + 8 # Total input featur to PC is gaussians + relus
    first_relu_ind = len(gauss_means)

    # Transform data into "input" n_gaussians dimensional format
    # This is effectively like taking our 4 input data features and passing
    # them through n_guassians number of hidden layer units using a
    # Gaussian activation function and fixed weights plus some relu units
    eye_transform = np.zeros((eye_data.shape[0], n_features))
    for k in range(0, n_eye_dims):
        # First do Gaussian activation on first 4 eye dims
        dim_means = gauss_means[k * n_gaussians_per_dim:(k + 1) * n_gaussians_per_dim]
        dim_stds = gauss_stds[k * n_gaussians_per_dim:(k + 1) * n_gaussians_per_dim]
        eye_transform[:, k * n_gaussians_per_dim:(k + 1) * n_gaussians_per_dim] = gaussian_activation(
                                                                        eye_data[:, k],
                                                                        dim_means,
                                                                        dim_stds)
        # Then relu activation on second 4 eye dims
        eye_transform[:, (first_relu_ind + 2 * k)] = negative_relu(
                                                            eye_data[:, n_eye_dims + k],
                                                            c=0.0)
        eye_transform[:, (first_relu_ind + (2 * k + 1))] = reflected_negative_relu(
                                                            eye_data[:, n_eye_dims + k],
                                                            c=0.0)
    return eye_transform



class FitNNModel(object):
    """ Class that fits neuron firing rates to eye data and is capable of
        calculating and outputting some basic info and predictions. Time window
        indicates the FIRING RATE time window, other data will be lagged relative
        to the fixed firing rate window. """

    def __init__(self, Neuron, time_window=[0, 800], blocks=None, trial_sets=None,
                    lag_range_pf=[-25, 25],
                    use_series=None):
        self.neuron = Neuron
        if use_series is not None:
            if use_series != Neuron.use_series:
                print("Input fit series {0} does not match Neuron's existing default series {1}. Resetting Neuron's series to {2}.".format(use_series, Neuron.use_series, use_series))
                Neuron.use_series = use_series
        self.time_window = np.array(time_window, dtype=np.int32)
        self.blocks = blocks
        # Want to make sure we only pull eye data for trials where this neuron
        # was valid by adding its valid trial set to trial_sets
        self.trial_sets = Neuron.append_valid_trial_set(trial_sets)
        self.lag_range_pf = np.array(lag_range_pf, dtype=np.int32)
        if self.lag_range_pf[1] <= self.lag_range_pf[0]:
            raise ValueError("lag_range_pf[1] must be greater than lag_range_pf[0]")
        self.fit_results = {}

    def get_firing_traces(self, return_inds=False):
        """ Calls the neuron's get firing rate functions using the input blocks
        and time windows used for making the fit object, making this call
        cleaner when used below in other methods.
        """
        fr, fr_inds = self.neuron.get_firing_traces(self.time_window, self.blocks,
                            self.trial_sets, return_inds=True)
        if return_inds:
            return fr, fr_inds
        else:
            return fr

    def get_eye_data_traces(self, blocks, trial_sets, lag=0, return_inds=False):
        """ Gets eye position and velocity in array of trial x self.time_window
            3rd dimension of array is ordered as pursuit, learning position,
            then pursuit, learning velocity.
            Data are only retrieved for valid neuron trials!
        """
        lag_time_window = self.time_window + np.int32(lag)
        if lag_time_window[1] <= lag_time_window[0]:
            raise ValueError("time_window[1] must be greater than time_window[0]")

        trial_sets = self.neuron.append_valid_trial_set(trial_sets)
        pos_p, pos_l, t_inds = self.neuron.session.get_xy_traces("eye position",
                                lag_time_window, blocks, trial_sets,
                                return_inds=True)
        vel_p, vel_l = self.neuron.session.get_xy_traces("eye velocity",
                                lag_time_window, blocks, trial_sets,
                                return_inds=False)
        eye_data = np.stack((pos_p, pos_l, vel_p, vel_l), axis=2)
        if return_inds:
            return eye_data, t_inds
        else:
            return eye_data

    def get_eye_data_traces_all_lags(self):
        """ Gets eye position and velocity in array of trial x
        self.time_window +/- self.lag_range_pf so that all lags_pf of data are
        pulled at once and can later be taken as slices/views for much faster
        fitting over lags_pf and single memory usage.
        3rd dimension of array is ordered as pursuit, learning position,
        then pursuit, learning velocity.
        """
        lag_time_window = [self.time_window[0] + self.lag_range_pf[0],
                            self.time_window[1] + self.lag_range_pf[1]]
        pos_p, pos_l, t_inds = self.neuron.session.get_xy_traces("eye position",
                                lag_time_window, self.blocks, self.trial_sets,
                                return_inds=True)
        vel_p, vel_l = self.neuron.session.get_xy_traces("eye velocity",
                                lag_time_window, self.blocks, self.trial_sets,
                                return_inds=False)
        eye_data = np.stack((pos_p, pos_l, vel_p, vel_l), axis=2)
        self.eye_lag_adjust = self.lag_range_pf[0]
        self.fit_dur = self.time_window[1] - self.time_window[0]
        return eye_data

    def get_eye_lag_slice(self, lag, eye_data):
        """ Will slice out a numpy view of eye_data adjusted to the input lag
        assuming eye_data follows the dimensions produced by
        get_eye_data_traces_all_lags, but can have extra data concatenated
        along axis 2. """
        ind_start = lag - self.eye_lag_adjust
        ind_stop = ind_start + self.fit_dur
        return eye_data[:, ind_start:ind_stop, :]

    def fit_gauss_basis_kinematics(self, pos_means, pos_stds, vel_means, vel_stds,
                                    bin_width=10, bin_threshold=5,
                                    fit_avg_data=False,
                                    quick_lag_step=10,
                                    train_split=1.0,
                                    learning_rate=0.02,
                                    epochs=200,
                                    batch_size=None):
        """ Fits the input neuron eye data to position and velocity using a
        basis set of Gaussians according to the input number of Gaussians over
        the state space ranges specified by pos/vel _range.
        Output "coeffs" are in order the order of the n_gaussians for position
        followed by the n_gaussians for velocity.
        A positive and negative temporal lag is fit first using a simple linear
        model with "get_lags_kinematic_fit" and the standard 4 direction tuning
        trials.
        """
        if train_split > 1.0:
            raise ValueError("Proportion to fit 'train_split' must be a value less than 1!")
        if ( (len(pos_means) != len(pos_stds)) or (len(vel_means) != len(vel_stds)) ):
            raise ValueError("Input means and STDs must match in length!")

        pf_lag, mli_lag = self.get_lags_kinematic_fit(quick_lag_step=quick_lag_step)

        # Setup all the indices for which trials we will be using and which
        # subset of trials will be used as training vs. test data
        firing_rate, all_t_inds = self.get_firing_traces(return_inds=True)
        n_fit_trials = np.int64(np.around(firing_rate.shape[0] * train_split))
        if n_fit_trials < 1:
            raise ValueError("Proportion to fit 'train_split' is too low to fit the minimum of 1 trial out of {0} total trials available.".format(firing_rate.shape[0]))
        n_test_trials = firing_rate.shape[0] - n_fit_trials
        # Now select and remember the trials used for fitting
        fit_trial_set = np.zeros(len(self.neuron.session), dtype='bool')
        n_test_trials = firing_rate.shape[0] - n_fit_trials
        is_test_data = False if n_test_trials == 0 else True
        # test_trial_set = np.zeros(len(self.neuron.session), dtype='bool')
        select_fit_trials = np.zeros(len(all_t_inds), dtype='bool')
        fit_trial_inds = np.random.choice(np.arange(0, firing_rate.shape[0]), n_fit_trials, replace=False)
        select_fit_trials[fit_trial_inds] = True # Index into trials used for this fitting object
        fit_trial_set[all_t_inds[select_fit_trials]] = True # Index into all trials in the session
        test_trial_set = ~fit_trial_set

        """ Here we have to do some work to get all the data in the correct format """
        # First get all firing rate data, bin and format
        if fit_avg_data:
            mean_rate_train = np.nanmean(firing_rate[select_fit_trials, :], axis=0, keepdims=True)
            binned_FR_train = bin_data(mean_rate_train, bin_width, bin_threshold)
            mean_rate_test = np.nanmean(firing_rate[~select_fit_trials, :], axis=0, keepdims=True)
            binned_FR_test = bin_data(mean_rate_test, bin_width, bin_threshold)
        else:
            binned_FR_train = bin_data(firing_rate[select_fit_trials, :], bin_width, bin_threshold)
            binned_FR_test = bin_data(firing_rate[~select_fit_trials, :], bin_width, bin_threshold)
        binned_FR_train = binned_FR_train.reshape(binned_FR_train.shape[0]*binned_FR_train.shape[1], order='C')
        FR_select_train = ~np.isnan(binned_FR_train)
        binned_FR_test = binned_FR_test.reshape(binned_FR_test.shape[0]*binned_FR_test.shape[1], order='C')
        FR_select_test = ~np.isnan(binned_FR_test)

        # Now get all the eye data at correct lags, bin and format
        # Get all the eye data at the desired lags
        eye_data_pf = self.get_eye_data_traces(self.blocks, self.trial_sets,
                            pf_lag)
        eye_data_mli = self.get_eye_data_traces(self.blocks, self.trial_sets,
                            mli_lag)
        eye_data = np.concatenate((eye_data_pf, eye_data_mli), axis=2)

        # Use bin smoothing on data before fitting
        bin_eye_data_train = bin_data(eye_data[select_fit_trials, :, :], bin_width, bin_threshold)
        bin_eye_data_test = bin_data(eye_data[~select_fit_trials, :, :], bin_width, bin_threshold)
        if fit_avg_data:
            bin_eye_data_train = np.nanmean(bin_eye_data_train, axis=0, keepdims=True)
            bin_eye_data_test = np.nanmean(bin_eye_data_test, axis=0, keepdims=True)
        # Reshape to 2D matrix
        bin_eye_data_train = bin_eye_data_train.reshape(
                                bin_eye_data_train.shape[0]*bin_eye_data_train.shape[1], bin_eye_data_train.shape[2], order='C')
        bin_eye_data_test = bin_eye_data_test.reshape(
                                bin_eye_data_test.shape[0]*bin_eye_data_test.shape[1], bin_eye_data_test.shape[2], order='C')

        # Now get all valid firing rate and eye data by removing nans
        select_good_train = np.logical_and(~np.any(np.isnan(bin_eye_data_train), axis=1), FR_select_train)
        bin_eye_data_train = bin_eye_data_train[select_good_train, :]
        binned_FR_train = binned_FR_train[select_good_train]

        select_good_test = np.logical_and(~np.any(np.isnan(bin_eye_data_test), axis=1), FR_select_test)
        bin_eye_data_test = bin_eye_data_test[select_good_test, :]
        binned_FR_test = binned_FR_test[select_good_test]

        """ Now data are setup and formated, we need to transform them into the
        input Gaussian space that spans position and velocity. """
        # First need to stack means and STDs for conversion function
        gauss_means = np.hstack([pos_means,
                                 pos_means,
                                 vel_means,
                                 vel_means])
        gauss_stds = np.hstack([pos_stds,
                                pos_stds,
                                vel_stds,
                                vel_stds])
        n_gaussians = len(gauss_means)
        # Now implement the input layer activation function
        eye_input_train = eye_input_to_PC_gauss_relu(bin_eye_data_train,
                                        gauss_means, gauss_stds)
        if is_test_data:
            eye_input_test = eye_input_to_PC_gauss_relu(bin_eye_data_test,
                                            gauss_means, gauss_stds)
            val_data = (eye_input_test, binned_FR_test)
        else:
            eye_input_test = []
            val_data = None

        # Create the neural network model
        model = models.Sequential([
            layers.Input(shape=(n_gaussians + 8,)),
            layers.Dense(1, activation="relu", kernel_constraint=constraints.NonNeg()),
        ])
        clip_value = None
        optimizer = SGD(learning_rate=learning_rate, clipvalue=clip_value)
        optimizer_str = "SGD"

        # Compile the model
        model.compile(optimizer=optimizer_str, loss='mean_squared_error')

        # Train the model
        if is_test_data:
            val_data = (eye_input_test, binned_FR_test)
            test_data_only = True
        else:
            val_data = None
            test_data_only = False
        history = model.fit(eye_input_train, binned_FR_train, epochs=epochs, batch_size=batch_size,
                                        validation_data=val_data, verbose=0)
        if is_test_data:
            test_loss = history.history['val_loss']
        else:
            test_loss = None
        train_loss = history.history['loss']

        # Store this for now so we can call predict_gauss_basis_kinematics
        # below for computing R2.
        self.fit_results['gauss_basis_kinematics'] = {
                                'pf_lag': pf_lag,
                                'mli_lag': mli_lag,
                                'coeffs': model.layers[0].get_weights()[0],
                                'bias': model.layers[0].get_weights()[1],
                                'n_gaussians': n_gaussians,
                                'pos_means': pos_means,
                                'pos_stds': pos_stds,
                                'vel_means': vel_means,
                                'vel_stds': vel_stds,
                                'R2': None,
                                'predict_fun': self.predict_gauss_basis_kinematics,
                                'fit_trial_set': fit_trial_set,
                                'test_trial_set': test_trial_set,
                                'is_test_data': is_test_data,
                                'test_loss': test_loss,
                                'train_loss': train_loss,
                                'model': model}
        # Compute R2
        if self.fit_results['gauss_basis_kinematics']['is_test_data']:
            test_firing_rate = firing_rate[~select_fit_trials, :]
        else:
            # If no test data are available, you need to just compute over all data
            test_firing_rate = firing_rate[select_fit_trials, :]
        if fit_avg_data:
            test_lag_data = self.get_gauss_basis_kinematics_predict_data_mean(
                                    self.blocks, self.trial_sets, test_data_only=test_data_only, verbose=False)
            y_predicted = self.predict_gauss_basis_kinematics(test_lag_data)
            test_mean_rate = np.nanmean(test_firing_rate, axis=0, keepdims=True)
            sum_squares_error = np.nansum((test_mean_rate - y_predicted) ** 2)
            sum_squares_total = np.nansum((test_mean_rate - np.nanmean(test_mean_rate)) ** 2)
        else:
            y_predicted = self.predict_gauss_basis_kinematics_by_trial(
                                    self.blocks, self.trial_sets, test_data_only=test_data_only, verbose=False)
            sum_squares_error = np.nansum((test_firing_rate - y_predicted) ** 2)
            sum_squares_total = np.nansum((test_firing_rate - np.nanmean(test_firing_rate)) ** 2)
            print(sum_squares_error, sum_squares_total)
        self.fit_results['gauss_basis_kinematics']['R2'] = 1 - sum_squares_error/(sum_squares_total)

        return

    def fit_learning_rates(self, blocks, trial_sets, bin_width=10, bin_threshold=5):
        """ Need the trials from blocks and trial_sets to be ORDERED! Weights will
        be updated from one trial to the next as if they are ordered and will
        not check if the numbers are correct because it could fail for various
        reasons like aborted trials. """

        ftol=1e-8
        xtol=1e-8
        gtol=1e-8
        max_nfev=200000
        loss='linear'

        """ Get all the binned firing rate data """
        firing_rate, all_t_inds = self.neuron.get_firing_traces(self.time_window,
                                            blocks, trial_sets, return_inds=True)
        CS_bin_evts = self.neuron.get_CS_dataseries_by_trial(self.time_window,
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
        eye_data, initial_shape = self.get_gauss_basis_kinematics_predict_data_trial(
                                        blocks, trial_sets,
                                        return_shape=True, test_data_only=False)
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
        # Firing rate data is only NaN where data for a trial does not cover self.time_window
        # So we need to find this separate from saccades and can set to 0.0 to ignore
        # We will AND this with where eye is NaN because both should be if data are truly missing
        is_missing_data = np.isnan(binned_FR) & eye_is_nan
        binned_FR[is_missing_data] = 0.0

        # Need the means and stds for converting state to input
        pos_means = self.fit_results['gauss_basis_kinematics']['pos_means']
        vel_means = self.fit_results['gauss_basis_kinematics']['vel_means']
        gauss_means = np.hstack([pos_means,
                                 pos_means,
                                 vel_means,
                                 vel_means])
        pos_stds = self.fit_results['gauss_basis_kinematics']['pos_stds']
        vel_stds = self.fit_results['gauss_basis_kinematics']['vel_stds']
        gauss_stds = np.hstack([pos_stds,
                                pos_stds,
                                vel_stds,
                                vel_stds])
        n_gaussians = len(gauss_means)

        # Defining learning function within scope so we have access to "self"
        # and specifically the weights. Get here to save space
        W = np.zeros((n_gaussians, 1))
        W_full = np.copy(self.fit_results['gauss_basis_kinematics']['coeffs'])
        W_0 = self.fit_results['gauss_basis_kinematics']['coeffs'][0:n_gaussians]
        b = self.fit_results['gauss_basis_kinematics']['bias']
        def learning_function(params, x, y):
            """ Defines the model we are fitting to the data """
            nonlocal W
            # Separate behavior state from CS inputs
            state = x[:, 0:-1]
            CS = x[:, -1]
            y_hat = np.zeros(x.shape[0])
            # Reset weights to initial fit values
            W[:] = W_0
            W_full[0:n_gaussians] = W_0
            alpha = params[0]
            beta = params[1]
            for trial in range(0, n_trials):
                state_trial = state[trial*n_obs_pt:(trial + 1)*n_obs_pt, :] # State for this trial
                eye_is_nan_trial = eye_is_nan[trial*n_obs_pt:(trial + 1)*n_obs_pt] # Nan state points for this trial

                # Convert state to input layer activations
                state_input = eye_input_to_PC_gauss_relu(state_trial,
                                                gauss_means, gauss_stds)
                # Set inputs derived from nan points to 0.0 so that the weights
                # for these states are not affected during nans
                state_input[eye_is_nan_trial, :] = 0.0
                # Expected rate this trial given updated weights
                # Use maximum here because of relu activation of output
                y_hat_trial = np.maximum(0, np.dot(state_input, W_full) + b).squeeze()
                # Store prediction for current trial
                y_hat[trial*n_obs_pt:(trial + 1)*n_obs_pt] = y_hat_trial
                # Update weights for next trial based on activations in this trial
                state_input = state_input[:, 0:n_gaussians]
                CS_trial = CS[trial*n_obs_pt:(trial + 1)*n_obs_pt] # CS for this trial
                CS_on_Inputs = np.dot(CS_trial, state_input) # Sum of CS over activation for each input unit
                # W += ( alpha * W * X_input - beta * CS * X_input )
                """ CS only learning with no LTP! """
                # W += ( (1 + 1/alpha) * (W - W_0) - (1 - 1/beta) * CS_on_Inputs[:, None] )
                W += ( alpha * (W_0 - W) - beta * CS_on_Inputs[:, None] )
                W_full[0:n_gaussians] = W
            missing_y_hat = np.isnan(y_hat)
            residuals = (y[~missing_y_hat] - y_hat[~missing_y_hat]) ** 2
            return residuals

        p0 = np.array([0.001, 0.005])
        # Set lower and upper bounds for each parameter
        lower_bounds = np.array([0, 0])
        upper_bounds = np.array([np.inf, np.inf])
        """ INPUT NEEDS TO BE BIN EYE DATA WITH A LAST COLUMN OF CS APPENDED! """

        # return bin_eye_data, binned_CS, binned_FR, p0, lower_bounds, upper_bounds, n_trials, n_obs_pt, eye_is_nan, gauss_means, gauss_stds

        fit_inputs = np.hstack([bin_eye_data, binned_CS[:, None]])
        # Fit the learning rates to the data
        result = least_squares(learning_function, p0, args=(fit_inputs, binned_FR),
                                bounds=(lower_bounds, upper_bounds),
                                ftol=ftol,
                                xtol=xtol,
                                gtol=gtol,
                                max_nfev=max_nfev,
                                loss=loss)
        self.fit_results['gauss_basis_kinematics']['alpha'] = result.x[0]
        self.fit_results['gauss_basis_kinematics']['beta'] = result.x[1]

        return result

    def get_learning_weights_by_trial(self, blocks, trial_sets, W_0=None,
                                        bin_width=10, bin_threshold=5):
        """ Need the trials from blocks and trial_sets to be ORDERED! """
        """ Get all the binned firing rate data """
        firing_rate, all_t_inds = self.neuron.get_firing_traces(self.time_window,
                                            blocks, trial_sets, return_inds=True)
        CS_bin_evts = self.neuron.get_CS_dataseries_by_trial(self.time_window,
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
        eye_data, initial_shape = self.get_gauss_basis_kinematics_predict_data_trial(
                                        blocks, trial_sets,
                                        return_shape=True, test_data_only=False)
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
        fit_inputs = np.hstack([bin_eye_data, binned_CS[:, None]])
        # Make an index of all nans that we can use in objective function to set
        # the unit activations to 0.0
        eye_is_nan = np.any(np.isnan(bin_eye_data), axis=1)
        # Firing rate data is only NaN where data for a trial does not cover self.time_window
        # So we need to find this separate from saccades and can set to 0.0 to ignore
        # We will AND this with where eye is NaN because both should be if data are truly missing
        is_missing_data = np.isnan(binned_FR) & eye_is_nan
        binned_FR[is_missing_data] = 0.0

        # Need the means and stds for converting state to input
        pos_means = self.fit_results['gauss_basis_kinematics']['pos_means']
        vel_means = self.fit_results['gauss_basis_kinematics']['vel_means']
        gauss_means = np.hstack([pos_means,
                                 pos_means,
                                 vel_means,
                                 vel_means])
        pos_stds = self.fit_results['gauss_basis_kinematics']['pos_stds']
        vel_stds = self.fit_results['gauss_basis_kinematics']['vel_stds']
        gauss_stds = np.hstack([pos_stds,
                                pos_stds,
                                vel_stds,
                                vel_stds])
        n_gaussians = len(gauss_means)

        if W_0 is None:
            W_0 = self.fit_results['gauss_basis_kinematics']['coeffs'][0:n_gaussians]
        if W_0.shape[0] != n_gaussians:
            raise ValueError("Input W_0 must have match the fit coefficients shape of {0}.".format(n_gaussians))
        weights_by_trial = {t_num: np.zeros(W_0.shape) for t_num in all_t_inds}
        b = self.fit_results['gauss_basis_kinematics']['bias']

        # Separate behavior state from CS inputs
        state = fit_inputs[:, 0:-1]
        CS = fit_inputs[:, -1]
        alpha = self.fit_results['gauss_basis_kinematics']['alpha']
        beta = self.fit_results['gauss_basis_kinematics']['beta']
        W = np.zeros(W_0.shape) # Place to store updating result and copy to output
        W[:] = W_0 # Initialize storage to start values
        return state, CS, alpha, beta, W, W_0, eye_is_nan, gauss_means, gauss_stds, n_trials, n_obs_pt
        for trial_ind, trial_num in zip(range(0, n_trials), all_t_inds):
            weights_by_trial[trial_num][:] = W # Copy W for this trial, befoe updating at end of loop
            state_trial = state[trial_ind*n_obs_pt:(trial_ind + 1)*n_obs_pt, :] # State for this trial
            eye_is_nan_trial = eye_is_nan[trial_ind*n_obs_pt:(trial_ind + 1)*n_obs_pt] # Nan state points for this trial
            # Convert state to input layer activations
            state_input = eye_input_to_PC_gauss_relu(state_trial,
                                            gauss_means, gauss_stds)
            # Set inputs derived from nan points to 0.0 so that the weights
            # for these states are not affected during nans
            state_input[eye_is_nan_trial, :] = 0.0
            state_input = state_input[:, 0:n_gaussians]
            # Update weights for next trial based on activations in this trial
            CS_trial = CS[trial_ind*n_obs_pt:(trial_ind + 1)*n_obs_pt] # CS for this trial
            CS_on_Inputs = np.dot(CS_trial, state_input) # Sum of CS over activation for each input unit
            # W += ( alpha * W * X_input - beta * CS * X_input )
            """ CS only learning with no LTP! """
            # W += ( (1 + 1/alpha) * (W - W_0) - (1 - 1/beta) * CS_on_Inputs[:, None] )
            W += ( alpha * (W_0 - W) - beta * CS_on_Inputs[:, None] )

        return weights_by_trial

    def get_gauss_basis_kinematics_predict_data_trial(self, blocks, trial_sets,
                                                      return_shape=False,
                                                      test_data_only=True,
                                                      verbose=False):
        """ Gets behavioral data from blocks and trial sets and formats in a
        way that it can be used to predict firing rate according to the linear
        eye kinematic model using predict_lin_eye_kinematics.
        Data are only retrieved for trials that are valid for the fitted neuron. """
        trial_sets = self.neuron.append_valid_trial_set(trial_sets)
        if test_data_only:
            if self.fit_results['gauss_basis_kinematics']['is_test_data']:
                trial_sets = trial_sets + [self.fit_results['gauss_basis_kinematics']['test_trial_set']]
            else:
                print("No test trials are available. Returning everything.")
        eye_data_pf = self.get_eye_data_traces(blocks, trial_sets,
                            self.fit_results['gauss_basis_kinematics']['pf_lag'])
        eye_data_mli = self.get_eye_data_traces(blocks, trial_sets,
                            self.fit_results['gauss_basis_kinematics']['mli_lag'])

        if verbose: print("PF lag:", self.fit_results['gauss_basis_kinematics']['pf_lag'])
        if verbose: print("MLI lag:", self.fit_results['gauss_basis_kinematics']['mli_lag'])
        eye_data = np.concatenate((eye_data_pf, eye_data_mli), axis=2)
        initial_shape = eye_data.shape
        eye_data = eye_data.reshape(eye_data.shape[0]*eye_data.shape[1], eye_data.shape[2], order='C')
        if return_shape:
            return eye_data, initial_shape
        else:
            return eye_data

    def get_gauss_basis_kinematics_predict_data_mean(self, blocks, trial_sets,
                                            test_data_only=True, verbose=False):
        """ Gets behavioral data from blocks and trial sets and formats in a
        way that it can be used to predict firing rate according to the linear
        eye kinematic model using predict_lin_eye_kinematics.
        Data for predictions are retrieved only for valid neuron trials."""
        lagged_pf_win = [self.time_window[0] + self.fit_results['gauss_basis_kinematics']['pf_lag'],
                          self.time_window[1] + self.fit_results['gauss_basis_kinematics']['pf_lag']
                         ]
        lagged_mli_win = [self.time_window[0] + self.fit_results['gauss_basis_kinematics']['mli_lag'],
                          self.time_window[1] + self.fit_results['gauss_basis_kinematics']['mli_lag']
                         ]
        if verbose: print("PF lag:", self.fit_results['gauss_basis_kinematics']['pf_lag'])
        if verbose: print("MLI lag:", self.fit_results['gauss_basis_kinematics']['mli_lag'])

        trial_sets = self.neuron.append_valid_trial_set(trial_sets)
        if test_data_only:
            if self.fit_results['gauss_basis_kinematics']['is_test_data']:
                trial_sets = trial_sets + [self.fit_results['gauss_basis_kinematics']['test_trial_set']]
            else:
                print("No test trials are available. Returning everything.")
        X = np.ones((self.time_window[1]-self.time_window[0], 8))
        X[:, 0], X[:, 1] = self.neuron.session.get_mean_xy_traces(
                                                "eye position", lagged_pf_win,
                                                blocks=blocks,
                                                trial_sets=trial_sets)
        X[:, 2], X[:, 3] = self.neuron.session.get_mean_xy_traces(
                                                "eye velocity", lagged_pf_win,
                                                blocks=blocks,
                                                trial_sets=trial_sets)
        X[:, 4], X[:, 5] = self.neuron.session.get_mean_xy_traces(
                                                "eye position", lagged_mli_win,
                                                blocks=blocks,
                                                trial_sets=trial_sets)
        X[:, 6], X[:, 7] = self.neuron.session.get_mean_xy_traces(
                                                "eye velocity", lagged_mli_win,
                                                blocks=blocks,
                                                trial_sets=trial_sets)
        return X

    def predict_gauss_basis_kinematics(self, X, return_manual=False):
        """
        """
        if X.shape[1] != 8:
            raise ValueError("Gaussian basis kinematics model is fit for 8 data dimensions but input data dimension is {0}.".format(X.shape[1]))

        pos_means = self.fit_results['gauss_basis_kinematics']['pos_means']
        vel_means = self.fit_results['gauss_basis_kinematics']['vel_means']
        gauss_means = np.hstack([pos_means,
                                 pos_means,
                                 vel_means,
                                 vel_means])
        pos_stds = self.fit_results['gauss_basis_kinematics']['pos_stds']
        vel_stds = self.fit_results['gauss_basis_kinematics']['vel_stds']
        gauss_stds = np.hstack([pos_stds,
                                pos_stds,
                                vel_stds,
                                vel_stds])
        X_input = eye_input_to_PC_gauss_relu(X,
                                        gauss_means, gauss_stds)
        # y_hat = X_input @ self.fit_results['gauss_basis_kinematics']['coeffs']
        # y_hat += self.fit_results['gauss_basis_kinematics']['bias']
        W = self.fit_results['gauss_basis_kinematics']['coeffs']
        b = self.fit_results['gauss_basis_kinematics']['bias']
        y_hat = np.maximum(0, np.dot(X_input, W) + b)
        y_hat_model = self.fit_results['gauss_basis_kinematics']['model'].predict(X_input).squeeze()
        if return_manual:
            return y_hat_model, y_hat
        else:
            return y_hat_model

    def predict_gauss_basis_kinematics_by_trial(self, blocks, trial_sets,
                                            test_data_only=True, verbose=False):
        """
        """
        X, init_shape = self.get_gauss_basis_kinematics_predict_data_trial(
                                blocks, trial_sets, return_shape=True,
                                test_data_only=test_data_only, verbose=verbose)
        y_hat = self.predict_gauss_basis_kinematics(X)
        y_hat = y_hat.reshape(init_shape[0], init_shape[1], order='C')
        return y_hat

    def get_lags_kinematic_fit(self, quick_lag_step=10):
        """ Uses the simple position/velocity only linear model on each of the
        4 direction tuning trials in block "StandTunePre" NO MATTER WHAT BLOCKS
        ARE INPUT FOR THE REMAINING FITS! The optimal lag is found for each
        direction AVERAGE and the best lag for the highest and lowest firing
        rate directions are returned.
        """
        # Hard coding bins to 1, and the blcoks
        lag_fit_time_window = [0, 250]
        fit_obj_time_window = [lag_fit_time_window[0] + self.lag_range_pf[0],
                               lag_fit_time_window[1] + self.lag_range_pf[1]]
        lag_fit_blocks = ['StandTunePre']
        bin_width = 1
        bin_threshold = 1

        max_peak_mod = -np.inf
        min_peak_mod = np.inf
        pos_lag = np.nan
        neg_lag = np.nan
        for t_set in ['pursuit', 'anti_pursuit', 'learning', 'anti_learning']:
            lag_fit_neuron_eye = FitNeuronToEye(self.neuron,
                                    time_window=fit_obj_time_window,
                                    blocks=lag_fit_blocks, trial_sets=t_set,
                                    lag_range_eye=self.lag_range_pf,
                                    lag_range_slip=self.lag_range_pf,
                                    dc_win=[0, 100], use_series=None)
            lag_fit_neuron_eye.fit_lin_eye_kinematics(bin_width=bin_width,
                                    bin_threshold=bin_threshold,
                                    fit_constant=True, fit_avg_data=False,
                                    quick_lag_step=quick_lag_step,
                                    ignore_acc=True)
            X = lag_fit_neuron_eye.get_lin_eye_kin_predict_data(lag_fit_blocks,
                                    t_set, verbose=False)
            y_hat = lag_fit_neuron_eye.predict_lin_eye_kinematics(X)

            peak_mod_ind = np.nanargmax(np.abs(y_hat))
            FR_peak_mod = y_hat[peak_mod_ind]
            if FR_peak_mod > max_peak_mod:
                max_peak_mod = FR_peak_mod
                pos_lag = lag_fit_neuron_eye.fit_results['lin_eye_kinematics']['eye_lag']
            if FR_peak_mod < min_peak_mod:
                min_peak_mod = FR_peak_mod
                neg_lag = lag_fit_neuron_eye.fit_results['lin_eye_kinematics']['eye_lag']
        print("PF lag:", pos_lag, "MLI lag:", neg_lag)
        return pos_lag, neg_lag

    def plot_2D_pos_fit(self, xy_pos_lims=None, fixed_xy_vel=[15, 6], pos_resolution=0.25):
        """ Makes plot of model prediction for fixed velocity across a grid of
        position spanning xy_pos_lims at resolution pos_resolution.
        call: plt.pcolormesh(pos_vals, pos_vals, fr_hat_mat) to plot output
        """
        if xy_pos_lims is None:
            xy_pos_lims = [np.amin(self.fit_results['gauss_basis_kinematics']['pos_means']),
                            np.amax(self.fit_results['gauss_basis_kinematics']['pos_means'])]
        if xy_pos_lims[0] is None:
            xy_pos_lims[0] = np.amin(self.fit_results['gauss_basis_kinematics']['pos_means'])
        if xy_pos_lims[1] is None:
            xy_pos_lims[1] = np.amax(self.fit_results['gauss_basis_kinematics']['pos_means'])
        pos_vals = np.arange(xy_pos_lims[0], xy_pos_lims[1] + pos_resolution, pos_resolution)
        x, y = np.meshgrid(pos_vals, pos_vals)
        X = np.zeros((x.size, 8))
        X[:, 0] = np.ravel(x)
        X[:, 1] = np.ravel(y)
        X[:, 2] = fixed_xy_vel[0]
        X[:, 3] = fixed_xy_vel[1]

        # Based on current fixed velocity, infer position input for MLIs?
        # Not sure if or how to do this at the moment...
        X[:, 4] = np.ravel(x)
        X[:, 5] = np.ravel(y)
        X[:, 6] = fixed_xy_vel[0]
        X[:, 7] = fixed_xy_vel[1]

        fr_hat = self.predict_gauss_basis_kinematics(X)
        fr_hat_mat = fr_hat.reshape(len(pos_vals), len(pos_vals))
        return pos_vals, fr_hat_mat

    def plot_2D_vel_fit(self, xy_vel_lims=None, fixed_xy_pos=[4, 1], vel_resolution=0.25):
        """ Makes plot of model prediction for fixed position across a grid of
        velocity spanning xy_vel_lims at resolution vel_resolution.
        call: plt.pcolormesh(vel_vals, vel_vals, fr_hat_mat) to plot output
        """
        if xy_vel_lims is None:
            xy_vel_lims = [np.amin(self.fit_results['gauss_basis_kinematics']['vel_means']),
                            np.amax(self.fit_results['gauss_basis_kinematics']['vel_means'])]
        if xy_vel_lims[0] is None:
            xy_vel_lims[0] = np.amin(self.fit_results['gauss_basis_kinematics']['vel_means'])
        if xy_vel_lims[1] is None:
            xy_vel_lims[1] = np.amax(self.fit_results['gauss_basis_kinematics']['vel_means'])
        vel_vals = np.arange(xy_vel_lims[0], xy_vel_lims[1] + vel_resolution, vel_resolution)
        x, y = np.meshgrid(vel_vals, vel_vals)
        X = np.zeros((x.size, 8))
        X[:, 0] = fixed_xy_pos[0]
        X[:, 1] = fixed_xy_pos[1]
        X[:, 2] = np.ravel(x)
        X[:, 3] = np.ravel(y)

        # Based on current fixed velocity, infer position input for MLIs?
        # Not sure if or how to do this at the moment...
        X[:, 4] = fixed_xy_pos[0]
        X[:, 5] = fixed_xy_pos[1]
        X[:, 6] = np.ravel(x)
        X[:, 7] = np.ravel(y)

        fr_hat = self.predict_gauss_basis_kinematics(X)
        fr_hat_mat = fr_hat.reshape(len(vel_vals), len(vel_vals))
        return vel_vals, fr_hat_mat





class CustomSigmoid(layers.Layer):
    def __init__(self, scale=1.0, shift=0.0, asymptote=1.0, bias=0.0, num_outputs=1, **kwargs):
        super(CustomSigmoid, self).__init__(**kwargs)
        self.scale_init = scale
        self.shift_init = shift
        self.asymptote_init = asymptote
        self.bias_init = bias
        self.num_outputs = num_outputs

    def build(self, input_shape):
        self.scale = self.add_weight("scale", shape=[self.num_outputs],
                                     initializer=initializers.Constant(self.scale_init),
                                     trainable=True)
        self.shift = self.add_weight("shift", shape=[self.num_outputs],
                                     initializer=initializers.Constant(self.shift_init),
                                     trainable=True)
        self.asymptote = self.add_weight("asymptote", shape=[self.num_outputs],
                                         initializer=initializers.Constant(self.asymptote_init),
                                         trainable=True)
        self.bias = self.add_weight("bias", shape=[self.num_outputs],
                                     initializer=initializers.Constant(self.bias_init),
                                     trainable=False)

    def call(self, inputs):
        return self.asymptote / (1 + tf.math.exp(-(inputs - self.shift) / self.scale)) + self.bias
