import numpy as np
from scipy.optimize import least_squares
import tensorflow as tf
from tensorflow.keras import layers, models, constraints, initializers
from tensorflow.keras.optimizers import SGD
import warnings
from NeuronAnalysis.fit_neuron_to_eye import FitNeuronToEye
from NeuronAnalysis.general import box_windows
import NeuronAnalysis.activation_functions as af
from NeuronAnalysis.fit_learning_rates import py_learning_function



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
                                    activation_out="relu",
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
        n_gaussians_per_dim = [len(pos_means), len(pos_means),
                               len(vel_means), len(vel_means)]
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
        eye_input_train = af.eye_input_to_PC_gauss_relu(bin_eye_data_train,
                                        gauss_means, gauss_stds,
                                        n_gaussians_per_dim=n_gaussians_per_dim)
        if is_test_data:
            eye_input_test = af.eye_input_to_PC_gauss_relu(bin_eye_data_test,
                                            gauss_means, gauss_stds,
                                            n_gaussians_per_dim=n_gaussians_per_dim)
            val_data = (eye_input_test, binned_FR_test)
        else:
            eye_input_test = []
            val_data = None
        self.activation_out = activation_out
        intrinsic_rate0 = min(np.nanmedian(binned_FR_train), 50)
        # Create the neural network model
        model = models.Sequential([
            layers.Input(shape=(n_gaussians + 8,)),
            layers.Dense(1, activation=activation_out,
                            kernel_constraint=constraints.NonNeg(),
                            bias_initializer=initializers.Constant(intrinsic_rate0)),
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

    def get_gauss_basis_kinematics_predict_data_trial(self, blocks, trial_sets,
                                                      return_shape=False,
                                                      test_data_only=True,
                                                      return_inds=False,
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
        eye_data_pf, t_inds = self.get_eye_data_traces(blocks, trial_sets,
                            self.fit_results['gauss_basis_kinematics']['pf_lag'],
                            return_inds=True)
        eye_data_mli = self.get_eye_data_traces(blocks, trial_sets,
                            self.fit_results['gauss_basis_kinematics']['mli_lag'])

        if verbose: print("PF lag:", self.fit_results['gauss_basis_kinematics']['pf_lag'])
        if verbose: print("MLI lag:", self.fit_results['gauss_basis_kinematics']['mli_lag'])
        eye_data = np.concatenate((eye_data_pf, eye_data_mli), axis=2)
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

    def predict_gauss_basis_kinematics(self, X, return_model=False):
        """
        """
        if X.shape[1] != 8:
            raise ValueError("Gaussian basis kinematics model is fit for 8 data dimensions but input data dimension is {0}.".format(X.shape[1]))

        pos_means = self.fit_results['gauss_basis_kinematics']['pos_means']
        vel_means = self.fit_results['gauss_basis_kinematics']['vel_means']
        n_gaussians_per_dim = [len(pos_means), len(pos_means),
                               len(vel_means), len(vel_means)]
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
        X_input = af.eye_input_to_PC_gauss_relu(X,
                                        gauss_means, gauss_stds,
                                        n_gaussians_per_dim=n_gaussians_per_dim)
        # y_hat = X_input @ self.fit_results['gauss_basis_kinematics']['coeffs']
        # y_hat += self.fit_results['gauss_basis_kinematics']['bias']
        W = self.fit_results['gauss_basis_kinematics']['coeffs']
        b = self.fit_results['gauss_basis_kinematics']['bias']
        y_hat = np.dot(X_input, W) + b
        if self.activation_out == "relu":
            y_hat = np.maximum(0, y_hat)

        if return_model:
            y_hat_model = self.fit_results['gauss_basis_kinematics']['model'].predict(X_input).squeeze()
            return y_hat, y_hat_model
        else:
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



""" SOME FUNCTIONS FOR GETTING DATA TO PREDICT FIRING BASED ON PLASTIC WEIGHTS """
""" ********************************************************************** """
def comp_learning_response(NN_FIT, X_trial, W_trial, return_comp=False):
    """
    """
    if X_trial.shape[2] != 8:
        raise ValueError("Gaussian basis kinematics model is fit for 8 data dimensions but input data dimension is {0}.".format(X.shape[1]))

    pos_means = NN_FIT.fit_results['gauss_basis_kinematics']['pos_means']
    vel_means = NN_FIT.fit_results['gauss_basis_kinematics']['vel_means']
    n_gaussians_per_dim = [len(pos_means), len(pos_means),
                           len(vel_means), len(vel_means)]
    gauss_means = np.hstack([pos_means,
                             pos_means,
                             vel_means,
                             vel_means])
    pos_stds = NN_FIT.fit_results['gauss_basis_kinematics']['pos_stds']
    vel_stds = NN_FIT.fit_results['gauss_basis_kinematics']['vel_stds']
    gauss_stds = np.hstack([pos_stds,
                            pos_stds,
                            vel_stds,
                            vel_stds])

    n_gaussians = len(gauss_means)
    y_hat = np.zeros((X_trial.shape[0], X_trial.shape[1]))
    W = np.copy(NN_FIT.fit_results['gauss_basis_kinematics']['coeffs'])
    b = NN_FIT.fit_results['gauss_basis_kinematics']['bias']
    pf_in = np.zeros((X_trial.shape[0], X_trial.shape[1]))
    mli_in = np.zeros((X_trial.shape[0], X_trial.shape[1]))
    for t_ind in range(0, X_trial.shape[0]):
        # Transform X_data for this trial into input space
        X_input = af.eye_input_to_PC_gauss_relu(X_trial[t_ind, :, :],
                                        gauss_means, gauss_stds,
                                        n_gaussians_per_dim=n_gaussians_per_dim)
        # Each trial update the weights for W
        W[:, 0] = W_trial[t_ind, :]
        y_hat[t_ind, :] = (np.dot(X_input, W) + b).squeeze()
        pf_in[t_ind, :] = (np.dot(X_input[:, 0:n_gaussians], W[0:n_gaussians, 0]) + b).squeeze()
        mli_in[t_ind, :] = (np.dot(X_input[:, n_gaussians:], W[n_gaussians:, 0]) + b).squeeze()
        if NN_FIT.activation_out == "relu":
            y_hat[t_ind, :] = np.maximum(0., y_hat[t_ind, :])
            pf_in[t_ind, :] = np.maximum(0., pf_in[t_ind, :])
            mli_in[t_ind, :] = np.maximum(0., mli_in[t_ind, :])

    if return_comp:
        return y_hat, pf_in, mli_in
    else:
        return y_hat


def predict_learning_response_by_trial(NN_FIT, blocks, trial_sets, weights_by_trial,
                                        return_comp=False, test_data_only=False,
                                        verbose=False):
    """
    """
    X, init_shape, t_inds = NN_FIT.get_gauss_basis_kinematics_predict_data_trial(
                            blocks, trial_sets, return_shape=True,
                            test_data_only=test_data_only, return_inds=True,
                            verbose=verbose)
    X_trial = X.reshape(init_shape)
    # Get weights in a single matrix to pass through here
    W_trial = np.zeros((len(weights_by_trial), weights_by_trial[t_inds[0]].shape[0]))
    # Go through t_nums IN ORDER
    for t_i, t in enumerate(t_inds):
        try:
            W_trial[t_i, :] = weights_by_trial[t].squeeze()
        except KeyError:
            print("weights by trial does not contain weights for requested trial number {0}.".format(t))
            continue
    if return_comp:
        y_hat, pf_in, mli_in = comp_learning_response(NN_FIT, X_trial, W_trial,
                                        return_comp=return_comp)
        return y_hat, pf_in, mli_in
    else:
        y_hat = comp_learning_response(NN_FIT, X_trial, W_trial)
        return y_hat


""" SOME HELPERS FOR GETTING THE EYE DATA TO FIT FOR PLASTIC WEIGHTS """
""" *********************************************************************** """
def get_eye_data_traces_win(NN_FIT, blocks, trial_sets, time_window, lag=0,
                            return_inds=False):
    """ Gets eye position and velocity in array of trial x time_window
        3rd dimension of array is ordered as pursuit, learning position,
        then pursuit, learning velocity.
        Data are only retrieved for valid neuron trials!
    """
    lag_time_window = time_window + np.int32(lag)
    if lag_time_window[1] <= lag_time_window[0]:
        raise ValueError("time_window[1] must be greater than time_window[0]")

    trial_sets = NN_FIT.neuron.append_valid_trial_set(trial_sets)
    pos_p, pos_l, t_inds = NN_FIT.neuron.session.get_xy_traces("eye position",
                            lag_time_window, blocks, trial_sets,
                            return_inds=True)
    vel_p, vel_l = NN_FIT.neuron.session.get_xy_traces("eye velocity",
                            lag_time_window, blocks, trial_sets,
                            return_inds=False)
    eye_data = np.stack((pos_p, pos_l, vel_p, vel_l), axis=2)
    if return_inds:
        return eye_data, t_inds
    else:
        return eye_data


def get_plasticity_data_trial_win(NN_FIT, blocks, trial_sets, time_window,
                                    return_shape=False, return_inds=False):
    """ Gets behavioral data from blocks and trial sets and formats in a
    way that it can be used to predict firing rate according to the linear
    eye kinematic model using predict_lin_eye_kinematics.
    Data are only retrieved for trials that are valid for the fitted neuron. """
    trial_sets = NN_FIT.neuron.append_valid_trial_set(trial_sets)
    eye_data_pf, t_inds = get_eye_data_traces_win(NN_FIT, blocks, trial_sets,
                            time_window,
                            NN_FIT.fit_results['gauss_basis_kinematics']['pf_lag'],
                            return_inds=True)
    eye_data_mli = get_eye_data_traces_win(NN_FIT, blocks, trial_sets,
                            time_window,
                            NN_FIT.fit_results['gauss_basis_kinematics']['mli_lag'])
    eye_data = np.concatenate((eye_data_pf, eye_data_mli), axis=2)
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


""" THESE ARE THE LEARNING RULE PLASTICITY FUNCTIONS """
""" *********************************************************************** """

def f_pf_CS_LTD(CS_trial_bin, tau_1, tau_2, scale=1.0, delay=0, zeta_f_move=None):
    """ Computes the parallel fiber LTD as a function of time of the complex
    spike input f_CS with a kernel scaled from tau_1 to tau_2 with peak equal to
    scale and with CSs shifted by an amoutn of time "delay" INDICES (not time!). """
    # Just CS point plasticity
    if zeta_f_move is None:
        pf_CS_LTD = box_windows(CS_trial_bin, tau_1, tau_2, scale=scale)
    else:
        pf_CS_LTD = box_windows(CS_trial_bin, tau_1, tau_2, scale=1.0)
        add_zeta = (pf_CS_LTD * zeta_f_move)
        pf_CS_LTD *= scale
        pf_CS_LTD += add_zeta
    # Shift pf_CS_LTD LTD envelope according to delay_LTD
    delay = int(delay)
    if delay == 0:
        # No delay so we are done
        return pf_CS_LTD
    elif delay < 0:
        pf_CS_LTD[-delay:] = pf_CS_LTD[0:delay]
        pf_CS_LTD[0:-delay] = 0.0
    else:
        # Implies delay > 0
        pf_CS_LTD[0:-delay] = pf_CS_LTD[delay:]
        pf_CS_LTD[-delay:] = 0.0
    return pf_CS_LTD

def f_pf_move_LTD(pf_CS_LTD, move_m_trial, move_LTD_scale):
    """
    """
    # Add a term with movement magnitude times weight
    # pf_CS_LTD += (pf_CS_LTD * np.sqrt(move_m_trial) * move_LTD_scale)
    return pf_CS_LTD

def f_pf_LTD(pf_CS_LTD, state_input_pf, pf_LTD, W_pf=None, W_min_pf=0.0):
    """ Updates the parallel fiber LTD function of time "pf_CS_LTD" to be scaled
    by PC firing rate if input, then scales by the pf state input and finally
    weights by the current weight function. pf_CS_LTD is MODIFIED IN PLACE!
    The output contains NEGATIVE values in pf_LTD. """
    # Sum of pf_CS_LTD weighted by activation for each input unit
    pf_LTD = np.dot(pf_CS_LTD, state_input_pf, out=pf_LTD)
    # Set state modification scaling according to current weight
    if W_pf is not None:
        pf_LTD *= (W_min_pf - W_pf) # Will all be negative values
    else:
        pf_LTD *= -1.0
    return pf_LTD

def f_pf_CS_LTP(CS_trial_bin, tau_1, tau_2, scale=1.0, zeta_f_move=None):
    """ Assumes CS_trial_bin is an array of 0's and 1's where 1's indicate that
    CS LTD is taking place and 0's indicate no CS related LTD. This function
    will then invert the CS train and form windows of LTP where LTD is absent.
    """
    if zeta_f_move is None:
        pf_CS_LTP = box_windows(CS_trial_bin, tau_1, tau_2, scale=scale)
    else:
        pf_CS_LTP = box_windows(CS_trial_bin, tau_1, tau_2, scale=1.0)
        add_zeta = (pf_CS_LTP * zeta_f_move)
        pf_CS_LTP *= scale
        pf_CS_LTP += add_zeta
    return pf_CS_LTP

def f_pf_static_LTP(pf_LTP_funs, pf_CS_LTD, static_weight_LTP, zeta_f_move=None):
    """
    """
    pf_LTP_funs += static_weight_LTP
    pf_LTP_funs[pf_CS_LTD > 0.0] = 0.0
    if zeta_f_move is not None:
        pf_LTP_funs += ( zeta_f_move)
    return pf_LTP_funs

def f_pf_FR_LTP(pf_LTP_funs, PC_FR, PC_FR_weight_LTP, zeta_f_move=None):
    """
    """
    # Add a term with firing rate times weight of constant LTP
    pf_LTP_funs += (PC_FR * PC_FR_weight_LTP)
    if zeta_f_move is not None:
        pf_LTP_funs += (PC_FR * zeta_f_move)
    return pf_LTP_funs

def f_pf_move_LTP(pf_LTP_funs, move_m_trial, move_LTP_scale):
    """
    """
    # Add a term with movement magnitude times weight
    # pf_LTP_funs *= np.sqrt(move_m_trial * move_LTP_scale + 1)
    return pf_LTP_funs

def f_pf_LTP(pf_LTP_funs, state_input_pf, pf_LTP, W_pf=None, W_max_pf=None):
    """ Updates the parallel fiber LTP function of time "pf_CS_LTP" to be scaled
    by PC firing rate if input, then scales by the pf state input and finally
    weights by the current weight function. pf_LTP is MODIFIED IN PLACE!
    W_max_pf should probably be input as FITTED PARAMETER! variable and is
    critical if using weight updates. Same for PC_FR_weight_LTP.
    """
    # Convert LTP functions to parallel fiber input space
    pf_LTP = np.dot(pf_LTP_funs, state_input_pf, out=pf_LTP)
    if W_pf is not None:
        if ( (W_max_pf is None) or (W_max_pf <= 0) ):
            raise ValueError("If updating weights by inputting values for W_pf, a W_max_pf > 0 must also be specified.")
        pf_LTP *= (W_max_pf - W_pf)
    return pf_LTP

def f_mli_CS_LTP(CS_trial_bin, tau_1, tau_2, scale=1.0, delay=0):
    """ Computes the MLI LTP as a function of time of the complex
    spike input f_CS with a kernel scaled from tau_1 to tau_2 with peak equal to
    scale and with CSs shifted by an amoutn of time "delay" INDICES (not time!). """
    # Just CS point plasticity
    mli_CS_LTP = box_windows(CS_trial_bin, tau_1, tau_2, scale=scale)
    # Shift mli_CS_LTP LTP envelope according to delay
    delay = int(delay)
    if delay == 0:
        # No delay so we are done
        return mli_CS_LTP
    elif delay < 0:
        mli_CS_LTP[-delay:] = mli_CS_LTP[0:delay]
        mli_CS_LTP[0:-delay] = 0.0
    else:
        # Implies delay > 0
        mli_CS_LTP[0:-delay] = mli_CS_LTP[delay:]
        mli_CS_LTP[-delay:] = 0.0
    return mli_CS_LTP

def f_mli_LTP(mli_CS_LTP, state_input_mli, W_mli=None, W_max_mli=None):
    """ Updates the MLI LTP function of time "mli_CS_LTP" to be scaled
    by weights by the current weight function. mli_CS_LTP is MODIFIED IN PLACE!"""
    # Sum of mli_CS_LTP weighted by activation for each input unit
    mli_LTP = np.dot(mli_CS_LTP, state_input_mli)
    # Set state modification scaling according to current weight
    if W_mli is not None:
        if ( (W_max_mli is None) or (W_max_mli <= 0) ):
            raise ValueError("If updating weights by inputting values for W_mli, a W_max_mli > 0 must also be specified.")
        mli_LTP *= (W_max_mli - W_mli)
    return mli_LTP

def f_mli_CS_LTD(CS_trial_bin, tau_1, tau_2, scale=1.0):
    """ Assumes CS_trial_bin is an array of 0's and 1's where 1's indicate that
    CS is taking place and 0's indicate no CS related LTD. This function
    will then invert the CS train and form windows of LTD where CS is absent.
    """
    mli_CS_LTD = box_windows(CS_trial_bin, tau_1, tau_2, scale=scale)
    return mli_CS_LTD

def f_mli_FR_LTD(PC_FR, PC_FR_weight_LTD_mli):
    """
    """
    # inv_PC_FR = 1 - PC_FR
    # inv_PC_FR[inv_PC_FR < 0.0] = 0.0
    # mli_FR_LTD = inv_PC_FR * PC_FR_weight_LTD_mli
    # Add a term with firing rate times weight of constant LTD
    mli_FR_LTD = PC_FR * PC_FR_weight_LTD_mli
    return mli_FR_LTD

def f_mli_pf_LTD(state_input_pf, W_pf, PC_FR_weight_LTD_mli):
    """
    """
    mli_FR_LTD = np.dot(state_input_pf, W_pf) * PC_FR_weight_LTD_mli
    return mli_FR_LTD

def f_mli_static_LTD(mli_CS_LTP, static_weight_mli_LTD):
    """ Inverts the input pf_CS_LTD fun so that it is opposite.
    """
    # Inverts the CS function
    mli_static_LTD = np.zeros_like(mli_CS_LTP)
    mli_static_LTD[mli_CS_LTP == 0.0] = static_weight_mli_LTD
    return mli_static_LTD

def f_mli_LTD(mli_LTD_funs, state_input_mli, W_mli=None, W_min_mli=0.0):
    """ Updates the parallel fiber LTP function of time "mli_CS_LTD" to be scaled
    by PC firing rate if input, then scales by the pf state input and finally
    weights by the current weight function. pf_LTP is MODIFIED IN PLACE!
    W_max_pf should probably be input as FITTED PARAMETER! variable and is
    critical if using weight updates. Same for PC_FR_weight_LTP.
    """
    # Convert LTD functions to MLI input space
    mli_LTD = np.dot(mli_LTD_funs, state_input_mli)
    if W_mli is not None:
        mli_LTD *= (W_min_mli - W_mli) # Will all be negative values
    else:
        mli_LTD *= -1.0
    return mli_LTD

""" *********************************************************************** """

def learning_function(params, x, y, W_0_pf, W_0_mli, b, *args, **kwargs):
    """ Defines the learning model we are fitting to the data """
    # Separate behavior state from CS inputs
    state = x[:, 0:-1]
    CS = x[:, -1]
    # Extract other precomputed necessary args
    bin_width = args[0]
    n_trials = args[1]
    n_obs_pt = args[2]
    is_missing_data = args[3]
    n_gaussians_per_dim = args[4]
    gauss_means = args[5]
    gauss_stds = args[6]
    n_gaussians = args[7]
    W_full = args[8]
    state_input = args[9]
    y_hat_trial = args[10]
    pf_LTD = args[11]
    pf_LTP = args[12]
    W_min_pf = 0.0
    W_min_mli = 0.0

    # Parse parameters to be fit
    alpha = params[0] / 1e4
    beta = params[1] / 1e4
    gamma = params[2] / 1e4
    epsilon = params[3] / 1e4
    W_max_pf = params[4]
    move_LTD_scale = params[5]
    move_LTP_scale = params[6]
    move_magn = np.linalg.norm(x[:, 2:4], axis=1)
    pf_scale = params[7]
    mli_scale = params[8]
    # Set weights to initial fit values
    W_pf = np.copy(W_0_pf)
    W_pf *= pf_scale
    W_mli = np.copy(W_0_mli)
    W_mli *= mli_scale
    # Ensure W_pf values are within range and store in output W_full
    W_pf[(W_pf > W_max_pf)] = W_max_pf
    W_pf[(W_pf < W_min_pf)] = W_min_pf
    W_mli[(W_mli < W_min_mli)] = W_min_mli
    if kwargs['UPDATE_MLI_WEIGHTS']:
        omega = params[5] / 1e4
        psi = params[6] / 1e4
        chi = params[7] / 1e4
        phi = params[8] / 1e4
        W_max_mli = params[9]
        W_mli[(W_mli > W_max_mli)] = W_max_mli
    W_full[0:n_gaussians] = W_pf
    W_full[n_gaussians:] = W_mli

    residuals = 0.0
    for trial in range(0, n_trials):
        state_trial = state[trial*n_obs_pt:(trial + 1)*n_obs_pt, :] # State for this trial
        move_m_trial = move_magn[trial*n_obs_pt:(trial + 1)*n_obs_pt] # Movement for this trial
        y_obs_trial = np.copy(y[trial*n_obs_pt:(trial + 1)*n_obs_pt]) # Observed FR for this trial
        is_missing_data_trial = is_missing_data[trial*n_obs_pt:(trial + 1)*n_obs_pt] # Nan state points for this trial

        # Convert state to input layer activations
        state_input = af.eye_input_to_PC_gauss_relu(state_trial,
                                        gauss_means, gauss_stds,
                                        n_gaussians_per_dim, state_input)
        # Set inputs derived from nan points to 0.0 so that the weights
        # for these states are not affected during nans
        state_input[is_missing_data_trial, :] = 0.0
        # No movement weight to missing/saccade data
        move_m_trial[is_missing_data_trial] = 0.0
        # Expected rate this trial given updated weights
        # Use maximum here because of relu activation of output
        y_hat_trial = np.dot(state_input, W_full, out=y_hat_trial)
        y_hat_trial += b # Add the bias term
        if kwargs['activation_out'] == "relu":
            y_hat_trial[y_hat_trial < 0.0] = 0.0
        # Now we can convert any nans to 0.0 so they don't affect residuals
        y_hat_trial[is_missing_data_trial] = 0.0
        y_obs_trial[is_missing_data_trial] = 0.0
        # Add residuals for current trial
        residuals += np.sum((y_obs_trial - y_hat_trial)**2)

        # Update weights for next trial based on activations in this trial
        state_input_pf = state_input[:, 0:n_gaussians]
        # Rescaled trial firing rate in proportion to max OVERWRITES y_obs_trial!
        y_obs_trial = y_obs_trial / kwargs['FR_MAX']
        # Binary CS for this trial
        CS_trial_bin = CS[trial*n_obs_pt:(trial + 1)*n_obs_pt]

        zeta_f_move = np.sqrt(move_m_trial) * move_LTD_scale
        # Get LTD function for parallel fibers
        pf_CS_LTD = f_pf_CS_LTD(CS_trial_bin, kwargs['tau_rise_CS'],
                          kwargs['tau_decay_CS'], epsilon, 0.0, zeta_f_move=None)
        # Add to pf_CS_LTD in place
        # pf_CS_LTD = f_pf_move_LTD(pf_CS_LTD, move_m_trial, move_LTD_scale)
        # Convert to LTD input for Purkinje cell
        pf_LTD = f_pf_LTD(pf_CS_LTD, state_input_pf, pf_LTD, W_pf=W_pf, W_min_pf=W_min_pf)

        zeta_f_move = np.sqrt(move_m_trial) * move_LTP_scale
        # Create the LTP function for parallel fibers
        pf_LTP_funs = f_pf_CS_LTP(CS_trial_bin, kwargs['tau_rise_CS_LTP'],
                        kwargs['tau_decay_CS_LTP'], alpha, zeta_f_move)
        # These functions add on to pf_LTP_funs in place
        pf_LTP_funs = f_pf_FR_LTP(pf_LTP_funs, y_obs_trial, beta, zeta_f_move)
        pf_LTP_funs = f_pf_static_LTP(pf_LTP_funs, pf_CS_LTD, gamma, zeta_f_move)
        # pf_LTP_funs = f_pf_move_LTP(pf_LTP_funs, move_m_trial, move_LTP_scale)
        # Convert to LTP input for Purkinje cell
        pf_LTP = f_pf_LTP(pf_LTP_funs, state_input_pf, pf_LTP, W_pf=W_pf, W_max_pf=W_max_pf)
        # Compute delta W_pf as LTP + LTD inputs and update W_pf
        W_pf += ( pf_LTP + pf_LTD )

        # Ensure W_pf values are within range and store in output W_full
        W_pf[(W_pf > W_max_pf)] = W_max_pf
        W_pf[(W_pf < W_min_pf)] = W_min_pf
        W_full[0:n_gaussians] = W_pf

        if kwargs['UPDATE_MLI_WEIGHTS']:
            # MLI state input is all <= 0, so need to multiply by -1 here
            state_input_mli = -1.0 * state_input[:, n_gaussians:]
            # Create the MLI LTP weighting function
            mli_CS_LTP = f_mli_CS_LTP(CS_trial_bin, kwargs['tau_rise_CS_mli_LTP'],
                              kwargs['tau_decay_CS_mli_LTP'], omega, 0.0)
            # Convert to LTP input for Purkinje cell MLI weights
            mli_LTP = f_mli_LTP(mli_CS_LTP, state_input_mli, W_mli, W_max_mli)

            # Create the LTD function for MLIs
            # mli_LTD_funs = f_mli_CS_LTD(CS_trial_bin, kwargs['tau_rise_CS_mli_LTD'],
            #                 kwargs['tau_decay_CS_mli_LTD'], psi)
            # mli_LTD_funs = f_mli_FR_LTD(y_obs_trial, chi)
            mli_LTD_funs = f_mli_static_LTD(mli_CS_LTP, phi)
            mli_LTD_funs[mli_CS_LTP > 0.0] = 0.0
            # Convert to LTD input for MLI
            mli_LTD = f_mli_LTD(mli_LTD_funs, state_input_mli, W_mli, W_min_mli)
            # Ensure W_mli values are within range and store in output W_full
            W_mli += ( mli_LTP[:, None] + mli_LTD[:, None] )
            W_mli[(W_mli > W_max_mli)] = W_max_mli
            W_mli[(W_mli < W_min_mli)] = W_min_mli
            W_full[n_gaussians:] = W_mli

    return residuals

def fit_learning_rates(NN_FIT, blocks, trial_sets, learn_t_win=None,
                        bin_width=10, bin_threshold=5):
    """ Need the trials from blocks and trial_sets to be ORDERED! Weights will
    be updated from one trial to the next as if they are ordered and will
    not check if the numbers are correct because it could fail for various
    reasons like aborted trials. """
    ftol=1e-2
    xtol=1e-8
    gtol=1e-8
    max_nfev=200000
    loss='linear'

    if learn_t_win is None:
        learn_t_win = NN_FIT.time_window
    NN_FIT.learn_rates_time_window = learn_t_win
    """ Get all the binned firing rate data. Get the trial indices and use those
    to get behavior since neural data can be fewer trials. """
    firing_rate, all_t_inds = NN_FIT.neuron.get_firing_traces(learn_t_win,
                                        blocks, trial_sets, return_inds=True)
    CS_bin_evts = NN_FIT.neuron.get_CS_dataseries_by_trial(learn_t_win,
                                blocks, all_t_inds, nan_sacc=False)

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
    eye_data, initial_shape = get_plasticity_data_trial_win(NN_FIT,
                                    blocks, all_t_inds, learn_t_win,
                                    return_shape=True)
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
    # Firing rate data is only NaN where data for a trial does not cover NN_FIT.time_window
    # So we need to find this separate from saccades and can set to 0.0 to ignore
    # We will AND this with where eye is NaN because both should be if data are truly missing
    is_missing_data = np.isnan(binned_FR) | eye_is_nan

    # Need the means and stds for converting state to input
    pos_means = NN_FIT.fit_results['gauss_basis_kinematics']['pos_means']
    vel_means = NN_FIT.fit_results['gauss_basis_kinematics']['vel_means']
    n_gaussians_per_dim = np.array([len(pos_means), len(pos_means),
                           len(vel_means), len(vel_means)], dtype=np.int32)
    gauss_means = np.hstack([pos_means,
                             pos_means,
                             vel_means,
                             vel_means], dtype=np.float64)
    pos_stds = np.float64(NN_FIT.fit_results['gauss_basis_kinematics']['pos_stds'])
    vel_stds = np.float64(NN_FIT.fit_results['gauss_basis_kinematics']['vel_stds'])
    gauss_stds = np.hstack([pos_stds,
                            pos_stds,
                            vel_stds,
                            vel_stds], dtype=np.float64)
    n_gaussians = np.int32(len(gauss_means))

    # Defining learning function within scope so we have access to "NN_FIT"
    # and specifically the weights. Get here to save space
    W_0_pf = np.float64(NN_FIT.fit_results['gauss_basis_kinematics']['coeffs'][0:n_gaussians].squeeze())
    W_0_mli = np.float64(NN_FIT.fit_results['gauss_basis_kinematics']['coeffs'][n_gaussians:].squeeze())
    b = np.float64(NN_FIT.fit_results['gauss_basis_kinematics']['bias'])
    # Initialize W_full to pass to objective function
    W_full = np.zeros((n_gaussians+8, ), dtype=np.float64)

    lf_kwargs = {'tau_rise_CS': int(np.around(25 /bin_width)),
                 'tau_decay_CS': int(np.around(0 /bin_width)),
                 'tau_rise_CS_LTP': int(np.around(-100 /bin_width)),
                 'tau_decay_CS_LTP': int(np.around(200 /bin_width)),
                 # 'tau_rise_CS_mli_LTP': int(np.around(80 /bin_width)),
                 # 'tau_decay_CS_mli_LTP': int(np.around(-40 /bin_width)),
                 # 'tau_rise_CS_mli_LTD': int(np.around(-40 /bin_width)),
                 # 'tau_decay_CS_mli_LTD': int(np.around(100 /bin_width)),
                 'FR_MAX': 500,
                 'UPDATE_MLI_WEIGHTS': False,
                 'activation_out': NN_FIT.activation_out,
                 }
    # Format of p0, upper, lower, index order for each variable to make this legible
    param_conds = {"alpha": (4.0, 0, np.inf, 0),
                   "beta": (1.0, 0, np.inf, 1),
                   "gamma": (1.0, 0, np.inf, 2),
                   "epsilon": (4.0, 0, np.inf, 3),
                   "W_max_pf": (10*np.amax(W_0_pf), np.amax(W_0_pf), np.inf, 4),
                   "move_LTD_scale": (1.0, 0.0, np.inf, 5),
                   "move_LTP_scale": (1.0, 0.0, np.inf, 6),
                   "pf_scale": (1.0, 0.7, 1.3, 7),
                   "mli_scale": (1.0, 0.7, 1.3, 8),
            }
    if lf_kwargs['UPDATE_MLI_WEIGHTS']:
        raise ValueError("check param nums")
        param_conds.update({"omega": (1.0, 0, np.inf, 5),
                            "psi": (1.0, 0, np.inf, 6),
                            "chi": (1.0, 0, np.inf, 7),
                            "phi": (1.0, 0, np.inf, 8),
                            "W_max_mli": (10*np.amax(W_0_mli), np.amax(W_0_mli), np.inf, 9),
                            })
    rescale_1e4 = ["alpha", "beta", "gamma", "epsilon",
                   "omega", "psi", "chi", "phi"]

    # Make sure params are in correct order and saved for input to least_squares
    p0 = [x[1][0] for x in sorted(param_conds.items(), key=lambda item: item[1][3])]
    lower_bounds = [x[1][1] for x in sorted(param_conds.items(), key=lambda item: item[1][3])]
    upper_bounds = [x[1][2] for x in sorted(param_conds.items(), key=lambda item: item[1][3])]

    # Finally append CS to inputs and get other args needed for learning function
    fit_inputs = np.hstack([bin_eye_data, binned_CS[:, None]])
    state_input = np.zeros((n_obs_pt, n_gaussians+8))
    y_hat_trial = np.zeros((n_obs_pt, ))
    pf_LTD = np.zeros((n_gaussians))
    pf_LTP = np.zeros((n_gaussians))
    lf_args = (bin_width, n_trials, n_obs_pt, is_missing_data,
                n_gaussians_per_dim, gauss_means, gauss_stds, n_gaussians,
                W_full, state_input, y_hat_trial, pf_LTD, pf_LTP)

    # Fit the learning rates to the data
    result = least_squares(learning_function, p0,
                            args=(fit_inputs, binned_FR, W_0_pf, W_0_mli, b, *lf_args),
                            kwargs=lf_kwargs,
                            bounds=(lower_bounds, upper_bounds),
                            ftol=ftol,
                            xtol=xtol,
                            gtol=gtol,
                            max_nfev=max_nfev,
                            loss=loss)
    for key in param_conds.keys():
        param_ind = param_conds[key][3]
        NN_FIT.fit_results['gauss_basis_kinematics'][key] = result.x[param_ind]
        if key in rescale_1e4:
            NN_FIT.fit_results['gauss_basis_kinematics'][key] /= 1e4
    for key in lf_kwargs.keys():
        NN_FIT.fit_results['gauss_basis_kinematics'][key] = lf_kwargs[key]

    return result

def get_learning_weights_by_trial(NN_FIT, blocks, trial_sets, W_0_pf=None,
                                    W_0_mli=None, bin_width=10, bin_threshold=5):
    """ Need the trials from blocks and trial_sets to be ORDERED! """
    """ Get all the binned firing rate data. Get the trial indices and use those
    to get behavior since neural data can be fewer trials. """
    firing_rate, all_t_inds = NN_FIT.neuron.get_firing_traces(
                                        NN_FIT.learn_rates_time_window,
                                        blocks, trial_sets, return_inds=True)
    CS_bin_evts = NN_FIT.neuron.get_CS_dataseries_by_trial(
                                NN_FIT.learn_rates_time_window,
                                blocks, all_t_inds, nan_sacc=False)
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
    eye_data, initial_shape = get_plasticity_data_trial_win(NN_FIT,
                                    blocks, all_t_inds,
                                    NN_FIT.learn_rates_time_window,
                                    return_shape=True)
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
    # Firing rate data is only NaN where data for a trial does not cover NN_FIT.time_window
    # So we need to find this separate from saccades and can set to 0.0 to ignore
    # We will AND this with where eye is NaN because both should be if data are truly missing
    is_missing_data = np.isnan(binned_FR) | eye_is_nan

    # Need the means and stds for converting state to input
    pos_means = NN_FIT.fit_results['gauss_basis_kinematics']['pos_means']
    vel_means = NN_FIT.fit_results['gauss_basis_kinematics']['vel_means']
    n_gaussians_per_dim = [len(pos_means), len(pos_means),
                           len(vel_means), len(vel_means)]
    gauss_means = np.hstack([pos_means,
                             pos_means,
                             vel_means,
                             vel_means])
    pos_stds = NN_FIT.fit_results['gauss_basis_kinematics']['pos_stds']
    vel_stds = NN_FIT.fit_results['gauss_basis_kinematics']['vel_stds']
    gauss_stds = np.hstack([pos_stds,
                            pos_stds,
                            vel_stds,
                            vel_stds])
    n_gaussians = len(gauss_means)

    if W_0_pf is None:
        W_0_pf = NN_FIT.fit_results['gauss_basis_kinematics']['coeffs'][0:n_gaussians].squeeze()
    if W_0_pf.shape[0] != n_gaussians:
        raise ValueError("Input W_0_pf must have match the fit coefficients shape of {0}.".format(n_gaussians))
    if W_0_mli is None:
        W_0_mli = NN_FIT.fit_results['gauss_basis_kinematics']['coeffs'][n_gaussians:].squeeze()
    if W_0_mli.shape[0] != 8:
        raise ValueError("Input W_0_mli must have match the MLI coefficients shape of 8.")
    b = NN_FIT.fit_results['gauss_basis_kinematics']['bias'].squeeze()
    W_min_pf = 0.0
    W_min_mli = 0.0

    # Separate behavior state from CS inputs
    state = fit_inputs[:, 0:-1]
    CS = fit_inputs[:, -1]
    # Fixed input params into one dict to match above
    kwargs = {}
    for key in NN_FIT.fit_results['gauss_basis_kinematics'].keys():
        if "tau" in key:
            kwargs[key] = NN_FIT.fit_results['gauss_basis_kinematics'][key]
        elif key in ["UPDATE_MLI_WEIGHTS"]:
            kwargs[key] = NN_FIT.fit_results['gauss_basis_kinematics'][key]
    FR_MAX = NN_FIT.fit_results['gauss_basis_kinematics']['FR_MAX']
    # Fit parameters
    alpha = NN_FIT.fit_results['gauss_basis_kinematics']['alpha']
    beta = NN_FIT.fit_results['gauss_basis_kinematics']['beta']
    gamma = NN_FIT.fit_results['gauss_basis_kinematics']['gamma']
    epsilon = NN_FIT.fit_results['gauss_basis_kinematics']['epsilon']
    W_max_pf = NN_FIT.fit_results['gauss_basis_kinematics']['W_max_pf']
    move_LTD_scale = NN_FIT.fit_results['gauss_basis_kinematics']['move_LTD_scale']
    move_LTP_scale = NN_FIT.fit_results['gauss_basis_kinematics']['move_LTP_scale']
    move_magn = np.linalg.norm(bin_eye_data[:, 2:4], axis=1)
    pf_scale = NN_FIT.fit_results['gauss_basis_kinematics']['pf_scale']
    mli_scale = NN_FIT.fit_results['gauss_basis_kinematics']['mli_scale']
    W_pf = np.zeros(W_0_pf.shape) # Place to store updating result and copy to output
    W_pf[:] = pf_scale * W_0_pf # Initialize storage to start values
    W_mli = np.zeros(W_0_mli.shape) # Place to store updating result and copy to output
    W_mli[:] = mli_scale * W_0_mli # Initialize storage to start values
    # Ensure W_pf values are within range and store in output W_full
    W_pf[(W_pf > W_max_pf)] = W_max_pf
    W_pf[(W_pf < W_min_pf)] = W_min_pf
    W_mli[(W_mli < W_min_mli)] = W_min_mli
    if kwargs['UPDATE_MLI_WEIGHTS']:
        omega = NN_FIT.fit_results['gauss_basis_kinematics']['omega']
        psi = NN_FIT.fit_results['gauss_basis_kinematics']['psi']
        chi = NN_FIT.fit_results['gauss_basis_kinematics']['chi']
        phi = NN_FIT.fit_results['gauss_basis_kinematics']['phi']
        W_max_mli = NN_FIT.fit_results['gauss_basis_kinematics']['W_max_mli']
        W_mli[(W_mli > W_max_mli)] = W_max_mli
    W_full = np.concatenate((W_pf, W_mli))
    state_input = np.zeros((n_obs_pt, n_gaussians+8))
    y_hat_trial = np.zeros((n_obs_pt, ))
    pf_LTD = np.zeros((n_gaussians))
    pf_LTP = np.zeros((n_gaussians))
    weights_by_trial = {t_num: np.zeros(W_full.shape) for t_num in all_t_inds}

    for trial_ind, trial_num in zip(range(0, n_trials), all_t_inds):
        weights_by_trial[trial_num][:] = W_full # Copy W for this trial, befoe updating at end of loop
        state_trial = state[trial_ind*n_obs_pt:(trial_ind + 1)*n_obs_pt, :] # State for this trial
        move_m_trial = move_magn[trial_ind*n_obs_pt:(trial_ind + 1)*n_obs_pt] # Movement for this trial
        y_obs_trial = binned_FR[trial_ind*n_obs_pt:(trial_ind + 1)*n_obs_pt] # Observed FR for this trial
        is_missing_data_trial = is_missing_data[trial_ind*n_obs_pt:(trial_ind + 1)*n_obs_pt] # Nan state points for this trial
        # Convert state to input layer activations
        state_input = af.eye_input_to_PC_gauss_relu(state_trial,
                                        gauss_means, gauss_stds,
                                        n_gaussians_per_dim=n_gaussians_per_dim)
        # Set inputs derived from nan points to 0.0 so t hat the weights
        # for these states are not affected during nans
        state_input[is_missing_data_trial, :] = 0.0
        # No movement weight to missing/saccade data
        move_m_trial[is_missing_data_trial] = 0.0
        y_obs_trial[is_missing_data_trial] = 0.0
        state_input_pf = state_input[:, 0:n_gaussians]

        # Rescaled trial firing rate in proportion to max
        y_obs_trial = y_obs_trial / FR_MAX
        # Binary CS for this trial
        CS_trial_bin = CS[trial_ind*n_obs_pt:(trial_ind + 1)*n_obs_pt]

        zeta_f_move = np.sqrt(move_m_trial) * move_LTD_scale
        # Get LTD function for parallel fibers
        pf_CS_LTD = f_pf_CS_LTD(CS_trial_bin, kwargs['tau_rise_CS'],
                          kwargs['tau_decay_CS'], epsilon, 0.0, zeta_f_move=None)
        # Add to pf_CS_LTD in place
        # pf_CS_LTD = f_pf_move_LTD(pf_CS_LTD, move_m_trial, move_LTD_scale)
        # Convert to LTD input for Purkinje cell
        pf_LTD = f_pf_LTD(pf_CS_LTD, state_input_pf, pf_LTD, W_pf=W_pf, W_min_pf=W_min_pf)

        zeta_f_move = np.sqrt(move_m_trial) * move_LTP_scale
        # Create the LTP function for parallel fibers
        pf_LTP_funs = f_pf_CS_LTP(CS_trial_bin, kwargs['tau_rise_CS_LTP'],
                        kwargs['tau_decay_CS_LTP'], alpha, zeta_f_move)
        # These functions add on to pf_LTP_funs in place
        pf_LTP_funs = f_pf_FR_LTP(pf_LTP_funs, y_obs_trial, beta, zeta_f_move)
        pf_LTP_funs = f_pf_static_LTP(pf_LTP_funs, pf_CS_LTD, gamma, zeta_f_move)
        # pf_LTP_funs = f_pf_move_LTP(pf_LTP_funs, move_m_trial, move_LTP_scale)
        # Convert to LTP input for Purkinje cell
        pf_LTP = f_pf_LTP(pf_LTP_funs, state_input_pf, pf_LTP, W_pf=W_pf, W_max_pf=W_max_pf)
        # Compute delta W_pf as LTP + LTD inputs and update W_pf
        W_pf += ( pf_LTP + pf_LTD )

        # Ensure W_pf values are within range and store in output W_full
        W_pf[(W_pf > W_max_pf)] = W_max_pf
        W_pf[(W_pf < W_min_pf)] = W_min_pf
        W_full[0:n_gaussians] = W_pf

        if kwargs['UPDATE_MLI_WEIGHTS']:
            # MLI state input is all <= 0, so need to multiply by -1 here
            state_input_mli = -1.0 * state_input[:, n_gaussians:]
            # Create the MLI LTP weighting function
            mli_CS_LTP = f_mli_CS_LTP(CS_trial_bin, kwargs['tau_rise_CS_mli_LTP'],
                              kwargs['tau_decay_CS_mli_LTP'], omega, 0.0)
            # Convert to LTP input for Purkinje cell MLI weights
            mli_LTP = f_mli_LTP(mli_CS_LTP, state_input_mli, W_mli, W_max_mli)

            # Create the LTD function for MLIs
            # mli_LTD_funs = f_mli_CS_LTD(CS_trial_bin, kwargs['tau_rise_CS_mli_LTD'],
            #                 kwargs['tau_decay_CS_mli_LTD'], psi)
            # mli_LTD_funs = f_mli_FR_LTD(y_obs_trial, chi)
            mli_LTD_funs = f_mli_static_LTD(mli_CS_LTP, phi)
            mli_LTD_funs[mli_CS_LTP > 0.0] = 0.0
            # Convert to LTD input for MLI
            mli_LTD = f_mli_LTD(mli_LTD_funs, state_input_mli, W_mli, W_min_mli)
            # Ensure W_mli values are within range and store in output W_full
            W_mli += ( mli_LTP[:, None] + mli_LTD[:, None] )
            W_mli[(W_mli > W_max_mli)] = W_max_mli
            W_mli[(W_mli < W_min_mli)] = W_min_mli
            W_full[n_gaussians:] = W_mli

    return weights_by_trial


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
