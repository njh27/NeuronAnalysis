import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
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



def binary_param(p):
    return np.around(sigmoid(p))

# Define Gaussian function
def gaussian(x, mu, sigma, scale):
    return scale * np.exp(-( ((x - mu) ** 2) / (2*(sigma**2))) )

def sigmoid(x, a=1, b=0):
    return 1 / (1 + np.exp(-(a * x + b)))

def sigmoid_activation(x, fixed_scale, fixed_centers):
    num_sigmoids = len(fixed_centers)
    x_transform = np.zeros((x.size, num_sigmoids))
    for k in range(num_sigmoids):
        x_transform[:, k] = sigmoid(x, fixed_scale, fixed_centers[k])
    return x_transform

# Define the model function as a linear combination of Gaussian functions
def gaussian_activation(x, fixed_means, fixed_sigma):
    num_gaussians = len(fixed_means)
    x_transform = np.zeros((x.size, num_gaussians))
    for k in range(num_gaussians):
        x_transform[:, k] = gaussian(x, fixed_means[k], fixed_sigma, scale=1.0)
    return x_transform

def downward_relu(x, a, b, c=0.):
    """ Rectified linear function that always chooses a slope that goes
    negative on the y-axis as you move away from the rectification point c. """
    # binary parameter selecting whether we will use only negative or positive values of x
    a = binary_param(a)
    # weight slope is set depending on a to be "downward"
    if b >= 0:
        b = -1*b if a == 1 else b
    else:
        b = -1*b if a == 0 else b
    # c is the rectification point
    return b * ( a * np.maximum(0, x - c) + ((a + 1) % 2) * np.minimum(0, x - c) )



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

    def fit_gauss_basis_kinematics(self, n_gaussians, std_gaussians,
                                    bin_width=10, bin_threshold=5,
                                    fit_avg_data=False,
                                    quick_lag_step=10,
                                    train_split=1.0):
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

        pf_lag, mli_lag = self.get_lags_kinematic_fit(bin_width=bin_width,
                                                bin_threshold=bin_threshold,
                                                quick_lag_step=quick_lag_step)

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
        # eye_data_pf = self.get_eye_data_traces(self.blocks, self.trial_sets,
        #                     pf_lag)
        # eye_data_mli = self.get_eye_data_traces(self.blocks, self.trial_sets,
        #                     mli_lag)
        # eye_data = np.concatenate((eye_data_pf, eye_data_mli), axis=2)
        eye_data = self.get_eye_data_traces(self.blocks, self.trial_sets,
                            pf_lag)
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
        # Find max position and velocity values so we can pick appropriate Gaussian centers
        max_abs_eye = np.maximum(np.nanmax(np.abs(bin_eye_data_train), axis=0), np.nanmax(np.abs(bin_eye_data_test), axis=0))
        max_abs_pos = max(np.amax(max_abs_eye[0:2]), 1) #np.amax(max_abs_eye[4:6]))
        max_abs_vel = max(np.amax(max_abs_eye[2:4]), 1) #np.amax(max_abs_eye[6:8]))
        pos_range = np.ceil(max_abs_pos + std_gaussians/2)
        vel_range = np.ceil(max_abs_vel + std_gaussians/2)
        print("Set pos_range to: ", pos_range, "and vel range to: ", vel_range)

        # Set up the basic values and fit function for basis set
        # Use inputs to set these variables and keep in scope for wrapper
        pos_fixed_means = np.linspace(-pos_range, pos_range, n_gaussians)
        vel_fixed_means = np.linspace(-vel_range, vel_range, n_gaussians)
        pos_fixed_std = std_gaussians
        vel_fixed_std = std_gaussians
        n_features = n_gaussians * bin_eye_data_train.shape[1]

        # Transform data into "input" n_gaussians dimensional format
        # This is effectively like taking our 4 input data features and passing
        # them through n_guassians number of hidden layer units using a
        # Gaussian activation function and fixed weights
        eye_input_train = np.zeros((bin_eye_data_train.shape[0], n_features))
        eye_input_test = np.zeros((bin_eye_data_test.shape[0], n_features))

        # eye_input_train_s = np.zeros((bin_eye_data_train.shape[0], n_features))
        # eye_input_test_s = np.zeros((bin_eye_data_test.shape[0], n_features))
        for k in range(0, 4):
            fixed_means = pos_fixed_means if k < 2 else vel_fixed_means
            fixed_sigma = pos_fixed_std if k < 2 else vel_fixed_std
            eye_input_train[:, k * n_gaussians:(k + 1) * n_gaussians] = gaussian_activation(
                                                                            bin_eye_data_train[:, k],
                                                                            fixed_means,
                                                                            fixed_sigma)
            eye_input_test[:, k * n_gaussians:(k + 1) * n_gaussians] = gaussian_activation(
                                                                            bin_eye_data_test[:, k],
                                                                            fixed_means,
                                                                            fixed_sigma)

            # eye_input_train_s[:, k * n_gaussians:(k + 1) * n_gaussians] = sigmoid_activation(
            #                                                                 bin_eye_data_train[:, k],
            #                                                                 1.0, fixed_means)
            # eye_input_test_s[:, k * n_gaussians:(k + 1) * n_gaussians] = sigmoid_activation(
            #                                                                 bin_eye_data_test[:, k],
            #                                                                 1.0, fixed_means)

        return eye_input_train, eye_input_test, binned_FR_train, binned_FR_test, pos_fixed_means, vel_fixed_means, pos_fixed_std, vel_fixed_std

        # Store this for now so we can call predict_gauss_basis_kinematics
        # below for computing R2.
        self.fit_results['gauss_basis_kinematics'] = {
                                'pf_lag': pf_lag,
                                'mli_lag': mli_lag,
                                'coeffs': popt,
                                'n_gaussians': n_gaussians,
                                'pos_means': pos_fixed_means,
                                'pos_stds': pos_fixed_std,
                                'vel_means': vel_fixed_means,
                                'vel_stds': vel_fixed_std,
                                'R2': None,
                                'predict_fun': self.predict_gauss_basis_kinematics,
                                'fit_trial_set': fit_trial_set,
                                'test_trial_set': test_trial_set,
                                'is_test_data': is_test_data}
        # Compute R2
        if self.fit_results['gauss_basis_kinematics']['is_test_data']:
            test_firing_rate = firing_rate[~select_fit_trials, :]
        else:
            # If no test data are available, you need to just compute over all data
            # so invert the selected trials
            test_firing_rate = firing_rate[select_fit_trials, :]
        if fit_avg_data:
            test_lag_data = self.get_gauss_basis_kinematics_predict_data_mean(
                                    self.blocks, self.trial_sets, test_data_only=True, verbose=False)
            y_predicted = self.predict_gauss_basis_kinematics(test_lag_data)
            test_mean_rate = np.nanmean(test_firing_rate, axis=0, keepdims=True)
            sum_squares_error = np.nansum((test_mean_rate - y_predicted) ** 2)
            sum_squares_total = np.nansum((test_mean_rate - np.nanmean(test_mean_rate)) ** 2)
        else:
            y_predicted = self.predict_gauss_basis_kinematics_by_trial(
                                    self.blocks, self.trial_sets, test_data_only=True, verbose=False)
            sum_squares_error = np.nansum((test_firing_rate - y_predicted) ** 2)
            sum_squares_total = np.nansum((test_firing_rate - np.nanmean(test_firing_rate)) ** 2)
            print(sum_squares_error, sum_squares_total)
        self.fit_results['gauss_basis_kinematics']['R2'] = 1 - sum_squares_error/(sum_squares_total)

        return

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
        print("GOT EYE OF SHape", eye_data_pf.shape)
        print("trial sets", trial_sets)
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

    def predict_gauss_basis_kinematics(self, X):
        """
        """
        if X.shape[1] != 8:
            raise ValueError("Gaussian basis kinematics model is fit for 8 data dimensions but input data dimension is {0}.".format(X.shape[1]))
        scales = self.fit_results['gauss_basis_kinematics']['coeffs']
        n_gaussians = self.fit_results['gauss_basis_kinematics']['n_gaussians']
        y_hat = np.zeros(X.shape[0]) + scales[-1] # Add intrisic rate constant
        # Added in fitted Gaussian basis
        for k in range(4):
            use_means = self.fit_results['gauss_basis_kinematics']['pos_means'] if k < 2 else self.fit_results['gauss_basis_kinematics']['vel_means']
            use_std = self.fit_results['gauss_basis_kinematics']['pos_stds'] if k < 2 else self.fit_results['gauss_basis_kinematics']['vel_stds']
            y_hat += gaussian_basis_set(X[:, k], scales[k * n_gaussians:(k + 1) * n_gaussians],
                                                                use_means, use_std)
        # Now add in the fitted linear components
        for k_ind, k in enumerate(range(4, 8)):
            a = scales[4 * n_gaussians + k_ind * 2]
            b = scales[4 * n_gaussians + k_ind * 2 + 1]
            y_hat += downward_relu(X[:, k], a, b, c=0.)
        return y_hat

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

    def get_lags_kinematic_fit(self, bin_width=10, bin_threshold=1, quick_lag_step=10):
        """ Uses the simple position/velocity only linear model on each of the
        4 direction tuning trials in block "StandTunePre" NO MATTER WHAT BLOCKS
        ARE INPUT FOR THE REMAINING FITS! The optimal lag is found fo reach
        direction and the best lag for the highest and lowest firing rate
        directions are returned.
        """
        lag_fit_time_window = [0, 250]
        fit_obj_time_window = [lag_fit_time_window[0] + self.lag_range_pf[0],
                               lag_fit_time_window[1] + self.lag_range_pf[1]]
        lag_fit_blocks = ['StandTunePre']

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





    def stuff(self):
        pass
