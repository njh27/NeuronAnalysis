import numpy as np
from numpy import linalg as la
from scipy.optimize import minimize
import warnings
from SessionAnalysis.utils import eye_data_series



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


class FitNeuronToEye(object):
    """ Class that fits neuron firing rates to eye data and is capable of
        calculating and outputting some basic info and predictions. Time window
        indicates the FIRING RATE time window, other data will be lagged relative
        to the fixed firing rate window. """

    def __init__(self, Neuron, time_window=[0, 800], blocks=None, trial_sets=None,
                    lag_range_eye=[-25, 25], lag_range_slip=[60, 120],
                    dc_win=[0, 100], use_series=None):
        self.neuron = Neuron
        if use_series is not None:
            if use_series != Neuron.use_series:
                print("Input fit series {0} does not match Neuron's existing default series {1}. Resetting Neuron's series to {2}.".format(use_series, Neuron.use_series, use_series))
                Neuron.use_series = use_series
        self.time_window = np.array(time_window, dtype=np.int32)
        self.blocks = blocks
        # Want to make sure we only pull eye data for trials where this neuron
        # was valid by adding its valid trial set to trial_sets
        if trial_sets is None:
            trial_sets = Neuron.name
        elif isinstance(trial_sets, list):
            trial_sets.append(Neuron.name)
        else:
            trial_sets = [trial_sets]
            trial_sets.append(Neuron.name)
        self.trial_sets = trial_sets
        self.lag_range_eye = np.array(lag_range_eye, dtype=np.int32)
        self.lag_range_slip = np.array(lag_range_slip, dtype=np.int32)
        if self.lag_range_eye[1] <= self.lag_range_eye[0]:
            raise ValueError("lag_range_eye[1] must be greater than lag_range_eye[0]")
        if self.lag_range_slip[1] <= self.lag_range_slip[0]:
            raise ValueError("lag_range_slip[1] must be greater than lag_range_slip[0]")
        self.dc_inds = np.array([dc_win[0] - time_window[0], dc_win[1] - time_window[0]], dtype=np.int32)
        self.fit_results = {}
        # self.FR = None
        # self.eye = None
        # self.slip = None

    def get_firing_traces(self):
        """ Calls the neuron's get firing rate functions using the input blocks
        and time windows used for making the fit object, making this call
        cleaner when used below in other methods.
        """
        fr = self.neuron.get_firing_traces(self.time_window, self.blocks,
                            self.trial_sets, return_inds=False)
        return fr

    def get_eye_data_traces(self, lag=0):
        """ Gets eye position and velocity in array of trial x self.time_window
            3rd dimension of array is ordered as pursuit, learning position,
            then pursuit, learning velocity.
        """
        lag_time_window = self.time_window + np.int32(lag)
        if lag_time_window[1] <= lag_time_window[0]:
            raise ValueError("time_window[1] must be greater than time_window[0]")

        pos_p, pos_l, t_inds = self.neuron.session.get_xy_traces("eye position",
                                lag_time_window, self.blocks, self.trial_sets,
                                return_inds=True)
        vel_p, vel_l = self.neuron.session.get_xy_traces("eye velocity",
                                lag_time_window, self.blocks, self.trial_sets,
                                return_inds=False)
        eye_data = np.stack((pos_p, pos_l, vel_p, vel_l), axis=2)
        return eye_data

    def get_eye_data_traces_all_lags(self):
        """ Gets eye position and velocity in array of trial x
        self.time_window +/- self.lag_range_eye so that all lags of data are
        pulled at once and can later be taken as slices/views for much faster
        fitting over lags and single memory usage.
        3rd dimension of array is ordered as pursuit, learning position,
        then pursuit, learning velocity.
        """
        lag_time_window = [self.time_window[0] + self.lag_range_eye[0],
                            self.time_window[1] + self.lag_range_eye[1]]
        pos_p, pos_l, t_inds = self.neuron.session.get_xy_traces("eye position",
                                lag_time_window, self.blocks, self.trial_sets,
                                return_inds=True)
        vel_p, vel_l = self.neuron.session.get_xy_traces("eye velocity",
                                lag_time_window, self.blocks, self.trial_sets,
                                return_inds=False)
        eye_data = np.stack((pos_p, pos_l, vel_p, vel_l), axis=2)
        self.eye_lag_adjust = self.lag_range_eye[0]
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


    def get_slip_data_traces(self, lag=0):
        """ Returns - time_window by len(maestro_PL2_data) by 2 array of
        retinal slip data. 3rd dimension of array is ordered as pursuit,
        learning slip.
        """
        lag_time_window = self.time_window + np.int32(lag)
        if lag_time_window[1] <= lag_time_window[0]:
            raise ValueError("time_window[1] must be greater than time_window[0]")

        slip_p, slip_l, t_inds = self.neuron.session.get_xy_traces("slip",
                                lag_time_window, self.blocks, self.trial_sets,
                                return_inds=True)
        slip_data = np.stack((slip_p, slip_l), axis=2)
        return slip_data

    def get_slip_data_traces_all_lags(self):
        """ Gets eye position and velocity in array of trial x
        self.time_window +/- self.lag_range_eye so that all lags of data are
        pulled at once and can later be taken as slices/views for much faster
        fitting over lags and single memory usage.
        3rd dimension of array is ordered as pursuit, learning position,
        then pursuit, learning velocity.
        """
        lag_time_window = [self.time_window[0] + self.lag_range_slip[0],
                            self.time_window[1] + self.lag_range_slip[1]]
        slip_p, slip_l, t_inds = self.neuron.session.get_xy_traces("slip",
                                lag_time_window, self.blocks, self.trial_sets,
                                return_inds=True)
        slip_data = np.stack((slip_p, slip_l), axis=2)
        self.slip_lag_adjust = self.lag_range_slip[0]
        self.fit_dur_slip = self.time_window[1] - self.time_window[0]
        return slip_data

    def get_slip_lag_slice(self, lag, slip_data):
        """ Will slice out a numpy view of eye_data adjusted to the input lag
        assuming eye_data follows the dimensions produced by
        get_eye_data_traces_all_lags, but can have extra data concatenated
        along axis 2. """
        ind_start = lag - self.slip_lag_adjust
        ind_stop = ind_start + self.fit_dur_slip
        return slip_data[:, ind_start:ind_stop, :]

    def fit_lin_eye_kinematics(self, bin_width=10, bin_threshold=1,
                                fit_constant=True, fit_avg_data=False):
        """ Fits the input neuron eye data to position, velocity, acceleration
        linear model (in 2 dimensions -- one pursuit axis and one learing axis)
        for the blocks and trial_sets input.
        Output "coeffs" are in order: position pursuit, position learning
                                      velocity pursuit, velocity learning
                                      acceleration pursuit, acceleration learning
                                      constant offset
        """
        lags = np.arange(self.lag_range_eye[0], self.lag_range_eye[1] + 1)
        R2 = []
        coefficients = []
        s_dim2 = 7 if fit_constant else 6
        firing_rate = self.get_firing_traces()
        if not fit_constant:
            dc_trial_rate = np.mean(firing_rate[:, self.dc_inds[0]:self.dc_inds[1]], axis=1)
            firing_rate = firing_rate - dc_trial_rate[:, None]
        if fit_avg_data:
            firing_rate = np.nanmean(firing_rate, axis=0, keepdims=True)
        binned_FR = bin_data(firing_rate, bin_width, bin_threshold)
        eye_data_all_lags = self.get_eye_data_traces_all_lags()
        # Initialize empty eye_data array that we can fill from slices of all data
        eye_data = np.ones((eye_data_all_lags.shape[0], self.fit_dur, s_dim2))
        for lag in lags:
            eye_data[:, :, 0:4] = self.get_eye_lag_slice(lag, eye_data_all_lags)
            eye_data[:, :, 4:6] = eye_data_series.acc_from_vel(eye_data[:, :, 2:4],
                            filter_win=self.neuron.session.saccade_ind_cushion)
            # Use bin smoothing on data before fitting
            bin_eye_data = bin_data(eye_data, bin_width, bin_threshold)
            if fit_avg_data:
                bin_eye_data = np.nanmean(bin_eye_data, axis=0, keepdims=True)
            bin_eye_data = bin_eye_data.reshape(bin_eye_data.shape[0]*bin_eye_data.shape[1], bin_eye_data.shape[2], order='C')
            temp_FR = binned_FR.reshape(binned_FR.shape[0]*binned_FR.shape[1], order='C')
            select_good = ~np.any(np.isnan(bin_eye_data), axis=1)
            bin_eye_data = bin_eye_data[select_good, :]
            temp_FR = temp_FR[select_good]
            coefficients.append(np.linalg.lstsq(bin_eye_data, temp_FR, rcond=None)[0])
            y_mean = np.mean(temp_FR)
            y_predicted = np.matmul(bin_eye_data, coefficients[-1])
            sum_squares_error = ((temp_FR - y_predicted) ** 2).sum()
            sum_squares_total = ((temp_FR - y_mean) ** 2).sum()
            R2.append(1 - sum_squares_error/(sum_squares_total))

        # Choose peak R2 value with minimum absolute value lag
        max_ind = np.where(R2 == np.amax(R2))[0]
        max_ind = max_ind[np.argmin(np.abs(lags[max_ind]))]
        dc_offset = coefficients[max_ind][-1] if fit_constant else 0.
        self.fit_results['lin_eye_kinematics'] = {
                                'eye_lag': lags[max_ind],
                                'slip_lag': None,
                                'coeffs': coefficients[max_ind],
                                'R2': R2[max_ind],
                                'all_R2': R2,
                                'use_constant': fit_constant,
                                'dc_offset': dc_offset,
                                'predict_fun': self.predict_lin_eye_kinematics}

    def get_lin_eye_kin_predict_data(self, blocks, trial_sets, verbose=False):
        """ Gets behavioral data from blocks and trial sets and formats in a
        way that it can be used to predict firing rate according to the linear
        eye kinematic model using predict_lin_eye_kinematics. """
        lagged_eye_win = [self.time_window[0] + self.fit_results['lin_eye_kinematics']['eye_lag'],
                          self.time_window[1] + self.fit_results['lin_eye_kinematics']['eye_lag']
                         ]
        if verbose: print("EYE lag:", self.fit_results['lin_eye_kinematics']['eye_lag'])
        s_dim2 = 7 if self.fit_results['lin_eye_kinematics']['use_constant'] else 6
        X = np.ones((self.time_window[1]-self.time_window[0], s_dim2))
        X[:, 0], X[:, 1] = self.neuron.session.get_mean_xy_traces(
                                                "eye position", lagged_eye_win,
                                                blocks=blocks,
                                                trial_sets=trial_sets)
        X[:, 2], X[:, 3] = self.neuron.session.get_mean_xy_traces(
                                                "eye velocity", lagged_eye_win,
                                                blocks=blocks,
                                                trial_sets=trial_sets)
        X[:, 4:6] = eye_data_series.acc_from_vel(X[:, 2:4], filter_win=29, axis=0)
        return X

    def predict_lin_eye_kinematics(self, X):
        """
        """
        if self.fit_results['lin_eye_kinematics']['use_constant']:
            if ~np.all(X[:, -1]):
                # Add column of 1's for constant
                X = np.hstack((X, np.ones((X.shape[0], 1))))
        if X.shape[1] != self.fit_results['lin_eye_kinematics']['coeffs'].shape[0]:
            raise ValueError("Linear eye kinematics is fit with 6 non-constant coefficients but input data dimension is {0}.".format(X.shape[1]))
        y_hat = np.matmul(X, self.fit_results['lin_eye_kinematics']['coeffs'])
        return y_hat

    def fit_eye_slip_interaction(self, bin_width=10, bin_threshold=1,
                                    fit_constant=True, fit_avg_data=False):
        """ Fits the input neuron eye data to position, velocity, and a
        slip x velocity interaction term.
        Output "coeffs" are in order: position pursuit, position learning
                                      velocity pursuit, velocity learning
                                      slip x velocity pursuit, slip x velocity learning
                                      constant offset
        """
        lags_eye = np.arange(self.lag_range_eye[0], self.lag_range_eye[1] + 1)
        lags_slip = np.arange(self.lag_range_slip[0], self.lag_range_slip[1] + 1)

        R2 = []
        lags_used = np.zeros((2, len(lags_eye) * len(lags_slip)), dtype=np.int64)
        n_fit = 0
        coefficients = []
        s_dim2 = 9 if fit_constant else 8
        firing_rate = self.get_firing_traces()
        print("the non fit constant rate should be per trial to avoid biases!?")
        if not fit_constant:
            dc_trial_rate = np.mean(firing_rate[:, self.dc_inds[0]:self.dc_inds[1]], axis=1)
            firing_rate = firing_rate - dc_trial_rate[:, None]
        if fit_avg_data:
            firing_rate = np.nanmean(firing_rate, axis=0, keepdims=True)
        binned_FR = bin_data(firing_rate, bin_width, bin_threshold)
        eye_data_all_lags = self.get_eye_data_traces_all_lags()
        slip_data_all_lags = self.get_slip_data_traces_all_lags()
        # Initialize empty eye_data array that we can fill from slices of all data
        eye_data = np.ones((eye_data_all_lags.shape[0], self.fit_dur, s_dim2))
        for elag in lags_eye:
            for slag in lags_slip:
                eye_data[:, :, 0:4] = self.get_eye_lag_slice(elag, eye_data_all_lags)
                eye_data[:, :, 4:6] = self.get_slip_lag_slice(slag, slip_data_all_lags)
                eye_data[:, :, 6:8] = self.get_slip_lag_slice(slag, slip_data_all_lags)
                # Use bin smoothing on data before fitting
                bin_eye_data = bin_data(eye_data, bin_width, bin_threshold)
                if fit_avg_data:
                    bin_eye_data = np.nanmean(eye_data, axis=0, keepdims=True)
                # Convert slip terms to position and velocity interactions
                bin_eye_data[:, :, 4:6] *= bin_eye_data[:, :, 0:2]
                bin_eye_data[:, :, 6:8] *= bin_eye_data[:, :, 2:4]
                bin_eye_data = bin_eye_data.reshape(bin_eye_data.shape[0]*bin_eye_data.shape[1], bin_eye_data.shape[2], order='C')
                temp_FR = binned_FR.reshape(binned_FR.shape[0]*binned_FR.shape[1], order='C')
                select_good = ~np.any(np.isnan(bin_eye_data), axis=1)
                bin_eye_data = bin_eye_data[select_good, :]
                temp_FR = temp_FR[select_good]

                coefficients.append(np.linalg.lstsq(bin_eye_data, temp_FR, rcond=None)[0])
                y_mean = np.mean(temp_FR)
                y_predicted = np.matmul(bin_eye_data, coefficients[-1])
                sum_squares_error = ((temp_FR - y_predicted) ** 2).sum()
                sum_squares_total = ((temp_FR - y_mean) ** 2).sum()
                R2.append(1 - sum_squares_error/(sum_squares_total))
                lags_used[0, n_fit] = elag
                lags_used[1, n_fit] = slag
                n_fit += 1

        # Choose peak R2 value with minimum absolute value lag
        max_ind = np.where(R2 == np.amax(R2))[0][0]
        dc_offset = coefficients[max_ind][-1] if fit_constant else 0.
        self.fit_results['eye_slip_interaction'] = {
                                'eye_lag': lags_used[0, max_ind],
                                'slip_lag': lags_used[1, max_ind],
                                'coeffs': coefficients[max_ind],
                                'R2': R2[max_ind],
                                'all_R2': R2,
                                'use_constant': fit_constant,
                                'dc_offset': dc_offset,
                                'predict_fun': self.predict_eye_slip_interaction}

    def get_eye_slip_inter_predict_data(self, blocks, trial_sets, verbose=False):
        """ Gets behavioral data from blocks and trial sets and formats in a
        way that it can be used to predict firing rate according to the eye
        slip interaction model using predict_eye_slip_interaction. """
        slip_lagged_eye_win = [self.time_window[0] + self.fit_results['eye_slip_interaction']['eye_lag'],
                               self.time_window[1] + self.fit_results['eye_slip_interaction']['eye_lag']
                              ]
        slip_lagged_slip_win = [self.time_window[0] + self.fit_results['eye_slip_interaction']['slip_lag'],
                                self.time_window[1] + self.fit_results['eye_slip_interaction']['slip_lag']
                               ]
        if verbose: print("EYE lag:", self.fit_results['eye_slip_interaction']['eye_lag'])
        if verbose: print("SLIP lag:", self.fit_results['eye_slip_interaction']['slip_lag'])
        s_dim2 = 9 if self.fit_results['eye_slip_interaction']['use_constant'] else 8
        X = np.ones((self.time_window[1]-self.time_window[0], s_dim2))


        X[:, 0], X[:, 1] = self.neuron.session.get_mean_xy_traces(
                                                "eye position",
                                                slip_lagged_eye_win,
                                                blocks=blocks,
                                                trial_sets=trial_sets)
        X[:, 2], X[:, 3] = self.neuron.session.get_mean_xy_traces(
                                                "eye velocity",
                                                slip_lagged_eye_win,
                                                blocks=blocks,
                                                trial_sets=trial_sets)
        X[:, 4], X[:, 5] = self.neuron.session.get_mean_xy_traces(
                                                "slip", slip_lagged_slip_win,
                                                blocks=blocks,
                                                trial_sets=trial_sets)
        X[:, 6:8] = X[:, 4:6]
        X[:, 4:6] *= X[:, 0:2]
        X[:, 6:8] *= X[:, 2:4]
        return X

    def predict_eye_slip_interaction(self, X):
        """
        """
        if self.fit_results['eye_slip_interaction']['use_constant']:
            if ~np.all(X[:, -1]):
                # Add column of 1's for constant
                X = np.hstack((X, np.ones((X.shape[0], 1))))
        if X.shape[1] != self.fit_results['eye_slip_interaction']['coeffs'].shape[0]:
            raise ValueError("Eye slip interaction is fit with 8 non-constant coefficients but input data dimension is {0}.".format(X.shape[1]))
        y_hat = np.matmul(X, self.fit_results['eye_slip_interaction']['coeffs'])
        return y_hat

    def fit_acc_kinem_interaction(self, bin_width=10, bin_threshold=1,
                                    fit_constant=True, fit_avg_data=False):
        """ Fits the input neuron eye data to position, velocity, and acceration
        as in the "standard" kinematic model but adds a separate lag for
        acceleration and acc x position and acc x velocity interaction terms
        to make more fair comparison to slip interaction model.
        Output "coeffs" are in order: position pursuit, position learning
                                      velocity pursuit, velocity learning
                                      slip x velocity pursuit, slip x velocity learning
                                      constant offset
        """
        lags_eye = np.arange(self.lag_range_eye[0], self.lag_range_eye[1] + 1)
        lags_acc = np.arange(self.lag_range_eye[0], self.lag_range_eye[1] + 1)
        # lags_slip = np.arange(self.lag_range_slip[0], self.lag_range_slip[1] + 1)

        R2 = []
        lags_used = np.zeros((2, len(lags_eye) * len(lags_acc)), dtype=np.int64)
        n_fit = 0
        coefficients = []
        s_dim2 = 11 if fit_constant else 10
        firing_rate = self.get_firing_traces()
        if not fit_constant:
            dc_trial_rate = np.mean(firing_rate[:, self.dc_inds[0]:self.dc_inds[1]], axis=1)
            firing_rate = firing_rate - dc_trial_rate[:, None]
        if fit_avg_data:
            firing_rate = np.nanmean(firing_rate, axis=0, keepdims=True)
        binned_FR = bin_data(firing_rate, bin_width, bin_threshold)
        eye_data_all_lags = self.get_eye_data_traces_all_lags()
        # Initialize empty eye_data array that we can fill from slices of all data
        eye_data = np.ones((eye_data_all_lags.shape[0], self.fit_dur, s_dim2))
        for elag in lags_eye:
            for alag in lags_acc:
                eye_data[:, :, 0:4] = self.get_eye_lag_slice(elag, eye_data_all_lags)
                # separate call at acc lag
                alag_eye_data = self.get_eye_lag_slice(alag, eye_data_all_lags)
                eye_data[:, :, 4:6] = eye_data_series.acc_from_vel(alag_eye_data[:, :, 2:4],
                                filter_win=self.neuron.session.saccade_ind_cushion)
                eye_data[:, :, 6:8] = eye_data[:, :, 4:6]
                eye_data[:, :, 8:10] = eye_data[:, :, 4:6]
                # Use bin smoothing on data before fitting
                bin_eye_data = bin_data(eye_data, bin_width, bin_threshold)
                if fit_avg_data:
                    bin_eye_data = np.nanmean(bin_eye_data, axis=0, keepdims=True)
                # Convert extra acc terms to position and velocity interactions
                bin_eye_data[:, :, 6:8] *= bin_eye_data[:, :, 0:2]
                bin_eye_data[:, :, 8:10] *= bin_eye_data[:, :, 2:4]
                bin_eye_data = bin_eye_data.reshape(bin_eye_data.shape[0]*bin_eye_data.shape[1], bin_eye_data.shape[2], order='C')
                temp_FR = binned_FR.reshape(binned_FR.shape[0]*binned_FR.shape[1], order='C')
                select_good = ~np.any(np.isnan(bin_eye_data), axis=1)
                bin_eye_data = bin_eye_data[select_good, :]
                temp_FR = temp_FR[select_good]
                coefficients.append(np.linalg.lstsq(bin_eye_data, temp_FR, rcond=None)[0])
                y_mean = np.mean(temp_FR)
                y_predicted = np.matmul(bin_eye_data, coefficients[-1])
                sum_squares_error = ((temp_FR - y_predicted) ** 2).sum()
                sum_squares_total = ((temp_FR - y_mean) ** 2).sum()
                R2.append(1 - sum_squares_error/(sum_squares_total))
                lags_used[0, n_fit] = elag
                lags_used[1, n_fit] = alag
                n_fit += 1

        # Choose peak R2 value with minimum absolute value lag
        max_ind = np.where(R2 == np.amax(R2))[0][0]
        dc_offset = coefficients[max_ind][-1] if fit_constant else 0.
        self.fit_results['acc_kinem_interaction'] = {
                                'eye_lag': lags_used[0, max_ind],
                                'acc_lag': lags_used[1, max_ind],
                                'coeffs': coefficients[max_ind],
                                'R2': R2[max_ind],
                                'all_R2': R2,
                                'use_constant': fit_constant,
                                'dc_offset': dc_offset,
                                'predict_fun': self.predict_acc_kinem_interaction}

    def get_acc_kinem_inter_predict_data(self, blocks, trial_sets, verbose=False):
        """ Gets behavioral data from blocks and trial sets and formats in a
        way that it can be used to predict firing rate according to the
        acceleration interaction model using predict_acc_kinem_interaction. """
        acc_lagged_eye_win = [self.time_window[0] + self.fit_results['acc_kinem_interaction']['eye_lag'],
                               self.time_window[1] + self.fit_results['acc_kinem_interaction']['eye_lag']
                              ]
        slip_lagged_acc_win = [self.time_window[0] + self.fit_results['acc_kinem_interaction']['acc_lag'],
                                self.time_window[1] + self.fit_results['acc_kinem_interaction']['acc_lag']
                               ]
        if verbose: print("EYE lag:", self.fit_results['acc_kinem_interaction']['eye_lag'])
        if verbose: print("ACC lag:", self.fit_results['acc_kinem_interaction']['acc_lag'])
        s_dim2 = 11 if self.fit_results['acc_kinem_interaction']['use_constant'] else 10
        X = np.ones((self.time_window[1]-self.time_window[0], s_dim2))
        X[:, 0], X[:, 1] = self.neuron.session.get_mean_xy_traces(
                                                "eye position", acc_lagged_eye_win,
                                                blocks=blocks,
                                                trial_sets=trial_sets)
        X[:, 2], X[:, 3] = self.neuron.session.get_mean_xy_traces(
                                                "eye velocity", acc_lagged_eye_win,
                                                blocks=blocks,
                                                trial_sets=trial_sets)
        acc_velx, acc_vely = self.neuron.session.get_mean_xy_traces(
                                                    "eye velocity",
                                                    slip_lagged_acc_win,
                                                    blocks=blocks,
                                                    trial_sets=trial_sets)
        X[:, 4:6] = eye_data_series.acc_from_vel(np.stack((acc_velx, acc_vely), axis=1),
                                                 filter_win=29, axis=0)
        X[:, 6:8] = X[:, 0:2] * X[:, 4:6]
        X[:, 8:10] = X[:, 2:4] * X[:, 4:6]
        return X

    def predict_acc_kinem_interaction(self, X):
        """
        """
        if self.fit_results['acc_kinem_interaction']['use_constant']:
            if ~np.all(X[:, -1]):
                # Add column of 1's for constant
                X = np.hstack((X, np.ones((X.shape[0], 1))))
        if X.shape[1] != self.fit_results['acc_kinem_interaction']['coeffs'].shape[0]:
            raise ValueError("Kinematic acceleration interaction is fit with 10 non-constant coefficients but input data dimension is {0}.".format(X.shape[1]))
        y_hat = np.matmul(X, self.fit_results['acc_kinem_interaction']['coeffs'])
        return y_hat





    def do_eye_lags(self, lag_range_eye=None):
        if lag_range_eye is not None:
            if lag_range_eye[0:2] != self.lag_range_eye:
                print("Lag range was reset from {} to {}".format(self.lag_range_eye, lag_range_eye[0:2]))
                self.lag_range_eye = lag_range_eye[0:2]
            if len(lag_range_eye) == 3:
                eye_step = lag_range_eye[2]
            else:
                eye_step = 1
        else:
            eye_step = 1
        return eye_step

    def do_slip_lags(self, lag_range_slip):
        if lag_range_slip is not None:
            if lag_range_slip[0:2] != self.lag_range_slip:
                print("Slip lag range was reset from {} to {}".format(self.lag_range_slip, lag_range_slip[0:2]))
                self.lag_range_slip = lag_range_slip[0:2]
            if len(lag_range_slip) == 3:
                slip_step = lag_range_slip[2]
            else:
                slip_step = 1
        else:
            slip_step = 1
        return slip_step

    def set_eye_fit_data(self, lag=0, bin_width=1):
        self.eye = eye_data_window(self.data, self.time_window + lag)
        self.eye = np.dstack((self.eye, acc_from_vel(self.eye[:, :, 2:4], max(self.data[0]['saccade_time_cushion'] - 1, 9))))
        self.eye = nan_sac_data_window(self.data, self.time_window + lag, self.eye)
        self.eye = bin_data(self.eye, bin_width, bin_threshold=0)
        self.eye = self.eye.reshape(self.eye.shape[0]*self.eye.shape[1], self.eye.shape[2], order='F')
        self.eye = self.eye[np.all(~np.isnan(self.eye), axis=1), :]

    def set_slip_fit_data(self, lag=0, bin_width=1):
        self.slip = eye_data_window(self.data, self.time_window + lag)
        self.slip = np.dstack((self.slip, slip_data_window(self.data, self.time_window + lag)))
        self.slip = nan_sac_data_window(self.data, self.time_window + self.fit_results['eye_lag'], self.slip)
        self.slip = bin_data(self.slip, bin_width, bin_threshold=0)
        self.slip = self.slip.reshape(self.slip.shape[0]*self.slip.shape[1], self.slip.shape[2], order='F')
        self.slip = self.slip[np.all(~np.isnan(self.slip), axis=1), :]

    def set_FR_fit_data(self, bin_width=1):
        self.FR = firing_rate_window(self.data, self.time_window, self.neuron, self.FR_name)
        self.FR = nan_sac_data_window(self.data, self.time_window + self.fit_results['eye_lag'], self.FR)
        self.FR = bin_data(self.FR, bin_width, bin_threshold=0)
        self.FR = self.FR.reshape(self.FR.shape[0]*self.FR.shape[1], order='F')
        self.FR = self.FR[~np.isnan(self.FR)]

    def fit_piece_linear(self, lag_range_eye=None, bin_width=1, constant=False):
        # The pieces are for x < 0 and x>= 0 for each column of eye data and the
        # output coefficients for eye data with n dimension swill be x[0] =>
        # x[0]+, x[1] => x[1]+, x[n+1] => x[0]-, x[n+2] => x[1]-, ...

        if lag_range_eye is not None:
            if lag_range_eye != self.lag_range_eye:
                print("Lag range was reset from {} to {}".format(self.lag_range_eye, lag_range_eye))
                self.lag_range_eye = lag_range_eye

        lags = np.arange(self.lag_range_eye[0], self.lag_range_eye[1] + 1)
        R2 = []
        coefficients = []
        firing_rate = firing_rate_window(self.data, self.time_window, self.neuron, self.FR_name)
        for lag in lags:
            eye_data = eye_data_window(self.data, self.time_window + lag)
            eye_data = np.dstack((eye_data, acc_from_vel(eye_data[:, :, 2:4], max(self.data[0]['saccade_time_cushion'] - 1, 9))))
            # Nan saccades
            temp_FR, eye_data = nan_sac_data_window(self.data, self.time_window + lag, firing_rate, eye_data)
            temp_FR = bin_data(temp_FR, bin_width, bin_threshold=0)
            eye_data = bin_data(eye_data, bin_width, bin_threshold=0)
            eye_data = eye_data.reshape(eye_data.shape[0]*eye_data.shape[1], eye_data.shape[2], order='F')
            temp_FR = temp_FR.flatten(order='F')
            keep_index = np.all(~np.isnan(np.column_stack((eye_data, temp_FR))), axis=1)
            eye_data = eye_data[keep_index, :]
            temp_FR = temp_FR[keep_index]

            if constant:
                piece_eye = np.zeros((eye_data.shape[0], 2 * eye_data.shape[1] + 1))
                piece_eye[:, -1] = 1
            else:
                piece_eye = np.zeros((eye_data.shape[0], 2 * eye_data.shape[1]))
            for column in range(0, eye_data.shape[1]):
                plus_index = eye_data[:, column] >= 0
                piece_eye[plus_index, column] = eye_data[plus_index, column]
                piece_eye[~plus_index, column + eye_data.shape[1]] = eye_data[~plus_index, column]
            coefficients.append(np.linalg.lstsq(piece_eye, temp_FR, rcond=None)[0])
            y_mean = np.mean(temp_FR)
            y_predicted = np.matmul(piece_eye, coefficients[-1])
            sum_squares_error = ((temp_FR - y_predicted) ** 2).sum()
            sum_squares_total = ((temp_FR - y_mean) ** 2).sum()
            R2.append(1 - sum_squares_error/(sum_squares_total))

        # Choose peak R2 value with minimum absolute value lag
        max_ind = np.where(R2 == np.amax(R2))[0]
        max_ind = max_ind[np.argmin(np.abs(lags[max_ind]))]

        self.fit_results['eye_lag'] = lags[max_ind]
        self.fit_results['slip_lag'] = None
        self.fit_results['coeffs'] = coefficients[max_ind]
        self.fit_results['R2'] = R2[max_ind]
        self.fit_results['all_R2'] = R2
        self.fit_results['model_type'] = 'piece_linear'
        self.fit_results['use_constant'] = constant
        self.set_FR_fit_data(bin_width)
        self.set_eye_fit_data(self.fit_results['eye_lag'], bin_width)

    def fit_slip_lag(self, trial_names=None, lag_range_slip=None):
        # Split data by trial name
        trials_by_type = {}
        for trial in range(0, len(self.data)):
            if trial_names is not None:
                if self.data[trial]['trial_name'] not in trial_names:
                    continue
            if self.data[trial]['trial_name'] not in trials_by_type:
                trials_by_type[self.data[trial]['trial_name']] = []
            trials_by_type[self.data[trial]['trial_name']].append(self.data[trial])
        if 'eye_lag' not in self.fit_results:
            eye_lag = 0
        else:
            eye_lag = self.fit_results['eye_lag']

        slip_step = self.do_slip_lags(lag_range_slip)
        slip_lags = np.arange(self.lag_range_slip[0], self.lag_range_slip[1] + 1, slip_step)
        R2 = np.zeros(slip_lags.size)
        coefficients = np.zeros((slip_lags.size, 5))
        n_s_lags = -1
        for s_lag in slip_lags:
            n_s_lags += 1
            # Get spikes and slip data separate for each trial name
            all_slip = []
            all_rate = []
            for trial_name in trials_by_type:
                _, _, slip, firing_rate, _ = get_slip_data(trials_by_type[trial_name], self.time_window, eye_lag, s_lag, self.FR_name, self.neuron, bin_width=1, avg='trial')
                all_slip.append(slip)
                all_rate.append(firing_rate)
            all_slip = np.vstack(all_slip)
            all_rate = np.concatenate(all_rate)
            keep_index = np.all(~np.isnan(np.column_stack((all_slip, all_rate))), axis=1)
            all_slip = all_slip[keep_index, :]
            all_rate = all_rate[keep_index]

            piece_slip = np.zeros((all_slip.shape[0], 5))
            for column in range(0, 2):
                plus_index = all_slip[:, column] >= 0
                piece_slip[plus_index, column] = all_slip[plus_index, column]
                piece_slip[~plus_index, column + 2] = all_slip[~plus_index, column]
            piece_slip[:, -1] = 1
            piece_slip[:, 0:2] = np.log(piece_slip[:, 0:2] + 1)
            piece_slip[:, 2:4] = -1 * np.log(-1 * piece_slip[:, 2:4] + 1)

            coefficients[n_s_lags, :] = np.linalg.lstsq(piece_slip, all_rate, rcond=None)[0]
            y_mean = np.mean(all_rate)
            y_predicted = np.matmul(piece_slip, coefficients[n_s_lags])
            sum_squares_error = ((all_rate - y_predicted) ** 2).sum()
            sum_squares_total = ((all_rate - y_mean) ** 2).sum()
            R2[n_s_lags] = 1 - sum_squares_error/(sum_squares_total)

        # Choose peak R2 value FROM SMOOTHED DATA with minimum absolute value lag
        # low_filt = 50
        # b_filt, a_filt = signal.butter(8, low_filt/500)
        # smooth_R2 = signal.filtfilt(b_filt, a_filt, R2, axis=0, padlen=int(.25 * len(R2)))
        sigma = 2
        gauss_filter = signal.gaussian(sigma*3*2 + 1, sigma)
        gauss_filter = gauss_filter / np.sum(gauss_filter)
        smooth_R2 = np.convolve(R2, gauss_filter, mode='same')
        max_ind = np.where(smooth_R2 == np.amax(smooth_R2))[0]
        max_ind = max_ind[np.argmin(np.abs(slip_lags[max_ind]))]
        self.fit_results['slip_lag'] = slip_lags[max_ind]
        self.fit_results['model_type'] = 'slip_lag'
        self.fit_results['R2'] = R2
        self.fit_results['smooth_R2'] = smooth_R2

    def fit_eye_lag(self, slip_lag=None, trial_names=None, lag_range_eye=None):

        if slip_lag is not None:
            n_coeffs = 13
            use_slip = True
        else:
            n_coeffs = 9
            use_slip = False
            slip_lag = 0

        # Split data by trial name
        trials_by_type = {}
        for trial in range(0, len(self.data)):
            if trial_names is not None:
                if self.data[trial]['trial_name'] not in trial_names:
                    continue
            if self.data[trial]['trial_name'] not in trials_by_type:
                trials_by_type[self.data[trial]['trial_name']] = []
            trials_by_type[self.data[trial]['trial_name']].append(self.data[trial])

        eye_step = self.do_eye_lags(lag_range_eye)
        eye_lags = np.arange(self.lag_range_eye[0], self.lag_range_eye[1] + 1, eye_step)
        R2 = np.zeros(eye_lags.size)
        coefficients = np.zeros((eye_lags.size, n_coeffs))
        n_e_lags = -1
        for e_lag in eye_lags:
            n_e_lags += 1
            # Get spikes and eye data separate for each trial name
            all_eye = []
            all_rate = []
            for trial_name in trials_by_type:
                position, velocity, slip, firing_rate, _ = get_slip_data(trials_by_type[trial_name], self.time_window, e_lag, slip_lag, self.FR_name, self.neuron, bin_width=1, avg='trial')
                all_eye.append(np.hstack((position, velocity, slip)))
                all_rate.append(firing_rate)
            all_eye = np.vstack(all_eye)
            if not use_slip:
                all_eye = all_eye[:, 0:4]
            all_rate = np.concatenate(all_rate)
            keep_index = np.all(~np.isnan(np.column_stack((all_eye, all_rate))), axis=1)
            all_eye = all_eye[keep_index, :]
            all_rate = all_rate[keep_index]
            piece_eye = np.zeros((all_eye.shape[0], n_coeffs))
            for column in range(0, int((n_coeffs-1)/2)):
                plus_index = all_eye[:, column] >= 0
                piece_eye[plus_index, column] = all_eye[plus_index, column]
                piece_eye[~plus_index, column + int((n_coeffs-1)/2)] = all_eye[~plus_index, column]
            piece_eye[:, -1] = 1
            ind_end = int((n_coeffs-1)/2)
            if n_coeffs > 9:
                piece_eye[:, 4:6] = np.log(piece_eye[:, 4:6] + 1)
                piece_eye[:, 10:12] = -1 * np.log(-1 * piece_eye[:, 10:12] + 1)
            # piece_eye[:, 0:ind_end] = np.log(piece_eye[:, 0:ind_end] + 1)
            # piece_eye[:, ind_end:n_coeffs-1] = -1 * np.log(-1 * piece_eye[:, ind_end:n_coeffs-1] + 1)

            coefficients[n_e_lags, :] = np.linalg.lstsq(piece_eye, all_rate, rcond=None)[0]
            y_mean = np.mean(all_rate)
            y_predicted = np.matmul(piece_eye, coefficients[n_e_lags])
            sum_squares_error = ((all_rate - y_predicted) ** 2).sum()
            sum_squares_total = ((all_rate - y_mean) ** 2).sum()
            R2[n_e_lags] = 1 - sum_squares_error/(sum_squares_total)

        # Choose peak R2 value FROM SMOOTHED DATA with minimum absolute value lag
        sigma = 2
        gauss_filter = signal.gaussian(sigma*3*2 + 1, sigma)
        gauss_filter = gauss_filter / np.sum(gauss_filter)
        smooth_R2 = np.convolve(R2, gauss_filter, mode='same')
        max_ind = np.where(smooth_R2 == np.amax(smooth_R2))[0]
        max_ind = max_ind[np.argmin(np.abs(eye_lags[max_ind]))]
        self.fit_results['eye_lag'] = eye_lags[max_ind]
        self.fit_results['model_type'] = 'eye_lag'
        self.fit_results['R2'] = R2
        self.fit_results['smooth_R2'] = smooth_R2

    def fit_piece_linear_interaction_fixlag(self, eye_lag, slip_lag, bin_width=1, constant=False):
        # The pieces are for x < 0 and x>= 0 for each column of eye data and the
        # output coefficients for eye data with n dimension swill be x[0] =>
        # x[0]+, x[1] => x[1]+, x[n+1] => x[0]-, x[n+2] => x[1]-, ...

        firing_rate = firing_rate_window(self.data, self.time_window, self.neuron, self.FR_name)
        eye_data = eye_data_window(self.data, self.time_window + eye_lag)
        slip_data = slip_data_window(self.data, self.time_window + slip_lag)
        all_data = np.dstack((eye_data, slip_data))
        firing_rate, all_data = nan_sac_data_window(self.data, self.time_window + eye_lag, firing_rate, all_data)

        firing_rate = bin_data(firing_rate, bin_width, bin_threshold=0)
        all_data = bin_data(all_data, bin_width, bin_threshold=0)
        firing_rate = firing_rate.flatten(order='F')
        all_data = all_data.reshape(all_data.shape[0]*all_data.shape[1], all_data.shape[2], order='F')
        keep_index = np.all(~np.isnan(np.column_stack((all_data, firing_rate))), axis=1)
        firing_rate = firing_rate[keep_index]
        all_data = all_data[keep_index, :]

        if constant:
            piece_slip = np.zeros((all_data.shape[0], 2 * all_data.shape[1] + 61))
            piece_slip[:, -1] = 1
        else:
            piece_slip = np.zeros((all_data.shape[0], 2 * all_data.shape[1] + 60))
        for column in range(0, all_data.shape[1]):
            plus_index = all_data[:, column] >= 0
            piece_slip[plus_index, column] = all_data[plus_index, column]
            piece_slip[~plus_index, column + all_data.shape[1]] = all_data[~plus_index, column]

        piece_slip[:, 4:6] = np.log(piece_slip[:, 4:6] + 1)
        piece_slip[:, 10:12] = -1 * np.log(-1 * piece_slip[:, 10:12] + 1)
        # piece_slip[:, 0:6] = np.log(piece_slip[:, 0:6] + 1)
        # piece_slip[:, 6:12] = -1 * np.log(-1 * piece_slip[:, 6:12] + 1)

        n_interaction = 2 * all_data.shape[1] - 1
        for column1 in range(0, 2 * all_data.shape[1]):
            for column2 in range(column1 + 1, 2 * all_data.shape[1]):
                if column2 == column1 + 6:
                    # This is an impossible interaction by definition so skip
                    continue
                n_interaction += 1
                piece_slip[:, n_interaction] = piece_slip[:, column1] * piece_slip[:, column2]

        coefficients = np.linalg.lstsq(piece_slip, firing_rate, rcond=None)[0]
        y_mean = np.mean(firing_rate)
        y_predicted = np.matmul(piece_slip, coefficients)
        sum_squares_error = ((firing_rate - y_predicted) ** 2).sum()
        sum_squares_total = ((firing_rate - y_mean) ** 2).sum()
        R2 = 1 - sum_squares_error/(sum_squares_total)

        self.fit_results['eye_lag'] = eye_lag
        self.fit_results['slip_lag'] = slip_lag
        self.fit_results['coeffs'] = coefficients
        self.fit_results['R2'] = R2
        self.fit_results['model_type'] = 'piece_linear'
        self.fit_results['use_constant'] = constant
        self.set_FR_fit_data(bin_width)
        self.set_slip_fit_data(self.fit_results['slip_lag'], bin_width)

    def predict_piece_linear(self, x_predict):

        if self.fit_results['model_type'] != "piece_linear":
            raise RuntimeError("piecewise linear model must be the current model fit to use this prediction")

        if self.eye is not None:
            if x_predict.shape[1] != self.eye.shape[1]:
                x_predict = x_predict.transpose()
            if x_predict.shape[1] != self.eye.shape[1]:
                raise ValueError("Input points for computing predictions must have the same dimension as eye fitted data")

        if self.slip is not None:
            if x_predict.shape[1] != self.slip.shape[1]:
                x_predict = x_predict.transpose()
            if x_predict.shape[1] != self.slip.shape[1]:
                raise ValueError("Input points for computing predictions must have the same dimension as slip fitted data")

        if self.fit_results['use_constant']:
            piece_eye = np.zeros((x_predict.shape[0], 2 * x_predict.shape[1] + 1))
            piece_eye[:, -1] = 1
        else:
            piece_eye = np.zeros((x_predict.shape[0], 2 * x_predict.shape[1]))
        for column in range(0, x_predict.shape[1]):
            plus_index = x_predict[:, column] >= 0
            piece_eye[plus_index, column] = x_predict[plus_index, column]
            piece_eye[~plus_index, column + x_predict.shape[1]] = x_predict[~plus_index, column]

        y_hat = np.matmul(piece_eye, self.fit_results['coeffs'])

        return y_hat


    def predict_piece_linear_interaction(self, x_predict):

        if self.fit_results['model_type'] != "piece_linear":
            raise RuntimeError("piecewise linear model must be the current model fit to use this prediction")

        if self.eye is not None:
            if x_predict.shape[1] != self.eye.shape[1]:
                x_predict = x_predict.transpose()
            if x_predict.shape[1] != self.eye.shape[1]:
                raise ValueError("Input points for computing predictions must have the same dimension as eye fitted data")

        if self.slip is not None:
            if x_predict.shape[1] != self.slip.shape[1]:
                x_predict = x_predict.transpose()
            if x_predict.shape[1] != self.slip.shape[1]:
                raise ValueError("Input points for computing predictions must have the same dimension as slip fitted data")

        if self.fit_results['use_constant']:
            piece_slip = np.zeros((x_predict.shape[0], 2 * x_predict.shape[1] + 61))
            piece_slip[:, -1] = 1
        else:
            piece_slip = np.zeros((x_predict.shape[0], 2 * x_predict.shape[1] + 60))
        for column in range(0, x_predict.shape[1]):
            plus_index = x_predict[:, column] >= 0
            piece_slip[plus_index, column] = x_predict[plus_index, column]
            piece_slip[~plus_index, column + x_predict.shape[1]] = x_predict[~plus_index, column]

        piece_slip[:, 4:6] = np.log(piece_slip[:, 4:6] + 1)
        piece_slip[:, 10:12] = -1 * np.log(-1 * piece_slip[:, 10:12] + 1)
        # piece_slip[:, 0:6] = np.log(piece_slip[:, 0:6] + 1)
        # piece_slip[:, 6:12] = -1 * np.log(-1 * piece_slip[:, 6:12] + 1)

        n_interaction = 2 * self.slip.shape[1] - 1
        for column1 in range(0, 2 * self.slip.shape[1]):
            for column2 in range(column1 + 1, 2 * self.slip.shape[1]):
                if column2 == column1 + 6:
                    # This is an impossible interaction by definition so skip
                    continue
                n_interaction += 1
                piece_slip[:, n_interaction] = piece_slip[:, column1] * piece_slip[:, column2]

        y_hat = np.matmul(piece_slip, self.fit_results['coeffs'])

        return y_hat
