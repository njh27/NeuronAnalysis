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
            trial_sets = [Neuron.name]
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

    def fit_pcwise_lin_eye_kinematics(self, bin_width=10, bin_threshold=1,
                                fit_constant=True, fit_avg_data=False,
                                quick_lag_step=10, use_knees=False):
        """ Fits the input neuron eye data to position, velocity, acceleration
        linear model (in 2 dimensions -- one pursuit axis and one learing axis)
        for the blocks and trial_sets input.
        Output "coeffs" are in order: position pursuit, position learning
                                      velocity pursuit, velocity learning
                                      acceleration pursuit, acceleration learning
                                      constant offset
        """
        quick_lag_step = int(np.around(quick_lag_step))
        if quick_lag_step < 1:
            raise ValueError("quick_lag_step must be positive integer")
        if quick_lag_step > (self.lag_range_eye[1] - self.lag_range_eye[0]):
            raise ValueError("quick_lag_step is too large relative to lag_range_eye")
        half_lag_step = np.int32(np.around(quick_lag_step / 2))
        lags = np.arange(self.lag_range_eye[0], self.lag_range_eye[1] + half_lag_step + 1, quick_lag_step)
        lags[-1] = self.lag_range_eye[1]

        R2 = []
        coefficients = []
        s_dim2 = 9 if fit_constant else 8
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
        if use_knees:
            knees = [self.fit_results['4D_planes']['pursuit_knee'], self.fit_results['4D_planes']['learning_knee']]
        else:
            knees = [0., 0.]
        # First loop over lags using quick_lag_step intervals
        for lag in lags:
            eye_data[:, :, 0:4] = self.get_eye_lag_slice(lag, eye_data_all_lags)


            # Copy over velocity data to make room for positions
            eye_data[:, :, 4:6] = eye_data[:, :, 2:4]
            eye_data[:, :, 2:4] = eye_data[:, :, 0:2]
            # Need to get the +/- position data separate
            X_select = eye_data[:, :, 0] >= knees[0]
            # eye_data[X_select, 0] = eye_data[X_select, 0]
            eye_data[~X_select, 0] = 0.0 # Less than knee dim0 = 0
            eye_data[X_select, 2] = 0.0 # Less than knee dim2 = 0
            X_select = eye_data[:, :, 1] >= knees[1]
            # eye_data[X_select, 0] = eye_data[X_select, 0]
            eye_data[~X_select, 1] = 0.0 # Less than knee dim1 = 0
            eye_data[X_select, 3] = 0.0 # Less than knee dim3 = 0


            eye_data[:, :, 6:8] = eye_data_series.acc_from_vel(eye_data[:, :, 4:6],
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

        if quick_lag_step > 1:
            # Do fine resolution loop
            max_ind = np.where(R2 == np.amax(R2))[0]
            max_ind = max_ind[np.argmin(np.abs(lags[max_ind]))]
            best_lag = lags[max_ind]
            # Make new lags centered on this best_lag
            lag_start = max(lags[0], best_lag - quick_lag_step)
            lag_stop = min(lags[-1], best_lag + quick_lag_step)
            lags = np.arange(lag_start, lag_stop + 1, 1)
            # Reset fit measures
            R2 = []
            coefficients = []
            for lag in lags:
                eye_data[:, :, 0:4] = self.get_eye_lag_slice(lag, eye_data_all_lags)
                # Copy over velocity data to make room for positions
                eye_data[:, :, 4:6] = eye_data[:, :, 2:4]
                eye_data[:, :, 2:4] = eye_data[:, :, 0:2]
                # Need to get the +/- position data separate
                X_select = eye_data[:, :, 0] >= knees[0]
                # eye_data[X_select, 0] = eye_data[X_select, 0]
                eye_data[~X_select, 0] = 0.0 # Less than knee dim0 = 0
                eye_data[X_select, 2] = 0.0 # Less than knee dim2 = 0
                X_select = eye_data[:, :, 1] >= knees[1]
                # eye_data[X_select, 0] = eye_data[X_select, 0]
                eye_data[~X_select, 1] = 0.0 # Less than knee dim1 = 0
                eye_data[X_select, 3] = 0.0 # Less than knee dim3 = 0
                eye_data[:, :, 6:8] = eye_data_series.acc_from_vel(eye_data[:, :, 4:6],
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
        self.fit_results['pcwise_lin_eye_kinematics'] = {
                                'eye_lag': lags[max_ind],
                                'slip_lag': None,
                                'coeffs': coefficients[max_ind],
                                'R2': R2[max_ind],
                                'all_R2': R2,
                                'use_constant': fit_constant,
                                'dc_offset': dc_offset,
                                'predict_fun': self.predict_lin_eye_kinematics}

    def get_pcwise_lin_eye_kin_predict_data(self, blocks, trial_sets, verbose=False):
        """ Gets behavioral data from blocks and trial sets and formats in a
        way that it can be used to predict firing rate according to the linear
        eye kinematic model using predict_lin_eye_kinematics. """
        lagged_eye_win = [self.time_window[0] + self.fit_results['pcwise_lin_eye_kinematics']['eye_lag'],
                          self.time_window[1] + self.fit_results['pcwise_lin_eye_kinematics']['eye_lag']
                         ]
        if verbose: print("EYE lag:", self.fit_results['pcwise_lin_eye_kinematics']['eye_lag'])
        s_dim2 = 9 if self.fit_results['lin_eye_kinematics']['use_constant'] else 8
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

    def predict_pcwise_lin_eye_kinematics(self, X):
        """
        """
        if self.fit_results['pcwise_lin_eye_kinematics']['use_constant']:
            if ~np.all(X[:, -1]):
                # Add column of 1's for constant
                X = np.hstack((X, np.ones((X.shape[0], 1))))
        if X.shape[1] != self.fit_results['pcwise_lin_eye_kinematics']['coeffs'].shape[0]:
            raise ValueError("Piecewise linear eye kinematics is fit with 8 non-constant coefficients but input data dimension is {0}.".format(X.shape[1]))
        y_hat = np.matmul(X, self.fit_results['pcwise_lin_eye_kinematics']['coeffs'])
        return y_hat

    def fit_lin_eye_kinematics(self, bin_width=10, bin_threshold=1,
                                fit_constant=True, fit_avg_data=False,
                                quick_lag_step=10):
        """ Fits the input neuron eye data to position, velocity, acceleration
        linear model (in 2 dimensions -- one pursuit axis and one learing axis)
        for the blocks and trial_sets input.
        Output "coeffs" are in order: position pursuit, position learning
                                      velocity pursuit, velocity learning
                                      acceleration pursuit, acceleration learning
                                      constant offset
        """
        quick_lag_step = int(np.around(quick_lag_step))
        if quick_lag_step < 1:
            raise ValueError("quick_lag_step must be positive integer")
        if quick_lag_step > (self.lag_range_eye[1] - self.lag_range_eye[0]):
            raise ValueError("quick_lag_step is too large relative to lag_range_eye")
        half_lag_step = np.int32(np.around(quick_lag_step / 2))
        lags = np.arange(self.lag_range_eye[0], self.lag_range_eye[1] + half_lag_step + 1, quick_lag_step)
        lags[-1] = self.lag_range_eye[1]

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
        # First loop over lags using quick_lag_step intervals
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

        if quick_lag_step > 1:
            # Do fine resolution loop
            max_ind = np.where(R2 == np.amax(R2))[0]
            max_ind = max_ind[np.argmin(np.abs(lags[max_ind]))]
            best_lag = lags[max_ind]
            # Make new lags centered on this best_lag
            lag_start = max(lags[0], best_lag - quick_lag_step)
            lag_stop = min(lags[-1], best_lag + quick_lag_step)
            lags = np.arange(lag_start, lag_stop + 1, 1)
            # Reset fit measures
            R2 = []
            coefficients = []
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
                                    fit_constant=True, fit_avg_data=False,
                                    quick_lag_step=10):
        """ Fits the input neuron eye data to position, velocity, and a
        slip x velocity interaction term.
        Output "coeffs" are in order: position pursuit, position learning
                                      velocity pursuit, velocity learning
                                      slip x velocity pursuit, slip x velocity learning
                                      constant offset
        """
        quick_lag_step = int(np.around(quick_lag_step))
        if quick_lag_step < 1:
            raise ValueError("quick_lag_step must be positive integer")
        if quick_lag_step > (self.lag_range_eye[1] - self.lag_range_eye[0]):
            raise ValueError("quick_lag_step is too large relative to lag_range_eye")
        if quick_lag_step > (self.lag_range_slip[1] - self.lag_range_slip[0]):
            raise ValueError("quick_lag_step is too large relative to lag_range_slip")
        half_lag_step = np.int32(np.around(quick_lag_step / 2))
        lags_eye = np.arange(self.lag_range_eye[0], self.lag_range_eye[1] + half_lag_step + 1, quick_lag_step)
        lags_eye[-1] = self.lag_range_eye[1]
        lags_slip = np.arange(self.lag_range_slip[0], self.lag_range_slip[1] + half_lag_step + 1, quick_lag_step)
        lags_slip[-1] = self.lag_range_slip[1]

        R2 = []
        coefficients = []
        lags_used = np.zeros((2, len(lags_eye) * len(lags_slip)), dtype=np.int64)
        n_fit = 0
        s_dim2 = 9 if fit_constant else 8
        firing_rate = self.get_firing_traces()
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
        # First loop over lags using quick_lag_step intervals
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

        if quick_lag_step > 1:
            # Do fine resolution loop
            max_ind = np.where(R2 == np.amax(R2))[0][0]
            best_eye_lag = lags_used[0, max_ind]
            # Make new lags_eye centered on this best_eye_lag
            lag_start_eye = max(lags_eye[0], best_eye_lag - quick_lag_step)
            lag_stop_eye = min(lags_eye[-1], best_eye_lag + quick_lag_step)
            lags_eye = np.arange(lag_start_eye, lag_stop_eye + 1, 1)
            best_slip_lag = lags_used[1, max_ind]
            # Make new lags_eye centered on this best_eye_lag
            lag_start_slip = max(lags_slip[0], best_slip_lag - quick_lag_step)
            lag_stop_slip = min(lags_slip[-1], best_slip_lag + quick_lag_step)
            lags_slip = np.arange(lag_start_slip, lag_stop_slip + 1, 1)
            # Reset fit measures
            R2 = []
            coefficients = []
            lags_used = np.zeros((2, len(lags_eye) * len(lags_slip)), dtype=np.int64)
            n_fit = 0
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
                                    fit_constant=True, fit_avg_data=False,
                                    quick_lag_step=10):
        """ Fits the input neuron eye data to position, velocity, and acceration
        as in the "standard" kinematic model but adds a separate lag for
        acceleration and acc x position and acc x velocity interaction terms
        to make more fair comparison to slip interaction model.
        Output "coeffs" are in order: position pursuit, position learning
                                      velocity pursuit, velocity learning
                                      slip x velocity pursuit, slip x velocity learning
                                      constant offset
        """
        quick_lag_step = int(np.around(quick_lag_step))
        if quick_lag_step < 1:
            raise ValueError("quick_lag_step must be positive integer")
        if quick_lag_step > (self.lag_range_eye[1] - self.lag_range_eye[0]):
            raise ValueError("quick_lag_step is too large relative to lag_range_eye")
        half_lag_step = np.int32(np.around(quick_lag_step / 2))
        lags_eye = np.arange(self.lag_range_eye[0], self.lag_range_eye[1] + half_lag_step + 1, quick_lag_step)
        lags_eye[-1] = self.lag_range_eye[1]
        lags_acc = np.arange(self.lag_range_eye[0], self.lag_range_eye[1] + half_lag_step + 1, quick_lag_step)
        lags_acc[-1] = self.lag_range_eye[1]

        R2 = []
        coefficients = []
        lags_used = np.zeros((2, len(lags_eye) * len(lags_acc)), dtype=np.int64)
        n_fit = 0
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
        # First loop over lags using quick_lag_step intervals
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

        if quick_lag_step > 1:
            # Do fine resolution loop
            max_ind = np.where(R2 == np.amax(R2))[0][0]
            best_eye_lag = lags_used[0, max_ind]
            # Make new lags_eye centered on this best_eye_lag
            lag_start_eye = max(lags_eye[0], best_eye_lag - quick_lag_step)
            lag_stop_eye = min(lags_eye[-1], best_eye_lag + quick_lag_step)
            lags_eye = np.arange(lag_start_eye, lag_stop_eye + 1, 1)
            best_acc_lag = lags_used[1, max_ind]
            # Make new lags_eye centered on this best_eye_lag
            lag_start_acc = max(lags_acc[0], best_acc_lag - quick_lag_step)
            lag_stop_acc = min(lags_acc[-1], best_acc_lag + quick_lag_step)
            lags_acc = np.arange(lag_start_acc, lag_stop_acc + 1, 1)
            # Reset fit measures
            R2 = []
            coefficients = []
            lags_used = np.zeros((2, len(lags_eye) * len(lags_acc)), dtype=np.int64)
            n_fit = 0
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


class FitNeuronPositionPlanes(FitNeuronToEye):
    """ Class that fits neuron firing rates to eye data using only the time
    the time points in time window where velocity < velocity_thresh. This will
    fit multiple planes to the position data at time_lag of zero to attempt
    to determine the location of any knees in position tuning. """

    def __init__(self, Neuron, time_window=[-200, 1200], blocks=None,
                    trial_sets=None, velocity_threshold=1.,
                    fixation_trial_sets=['fixation_trials'],
                    fix_trial_time_window=[200, 400], use_series=None):
        self.neuron = Neuron
        if use_series is not None:
            if use_series != Neuron.use_series:
                print("Input fit series {0} does not match Neuron's existing default series {1}. Resetting Neuron's series to {2}.".format(use_series, Neuron.use_series, use_series))
                Neuron.use_series = use_series
        self.time_window = np.array(time_window, dtype=np.int32)
        self.fix_trial_time_window = np.array(fix_trial_time_window, dtype=np.int32)
        self.blocks = blocks
        # Want to make sure we only pull eye data for trials where this neuron
        # was valid by adding its valid trial set to trial_sets
        if trial_sets is None:
            trial_sets = [Neuron.name]
        elif isinstance(trial_sets, list):
            trial_sets.append(Neuron.name)
        else:
            trial_sets = [trial_sets]
            trial_sets.append(Neuron.name)
        self.trial_sets = trial_sets
        self.fixation_trial_sets = np.zeros(len(Neuron.session), dtype='bool')
        if fixation_trial_sets is not None:
            if not isinstance(fixation_trial_sets, list):
                fixation_trial_sets = [fixation_trial_sets]
            for fts in fixation_trial_sets:
                self.fixation_trial_sets = self.fixation_trial_sets | Neuron.session._get_trial_set(fts)
        self.fit_results = {}

    def get_fix_only_eye_traces(self, return_t_inds=False):
        """ Gets eye position in array of trial x self.fix_trial_time_window
            3rd dimension of array is ordered as pursuit, learning position,
            then pursuit, learning velocity. Data are gathered for trials from
            fixation_trial_sets ONLY!
        """
        use_t_sets = [x for x in self.trial_sets] # Do not want to overwrite trial_sets
        use_t_sets.append(self.fixation_trial_sets)
        pos_p, pos_l, t_inds = self.neuron.session.get_xy_traces("eye position",
                                self.fix_trial_time_window, self.blocks,
                                use_t_sets, return_inds=True)
        eye_data = np.stack((pos_p, pos_l), axis=2)
        if return_t_inds:
            return eye_data, t_inds
        else:
            return eye_data

    def get_purs_only_eye_traces(self, return_t_inds=False):
        """ Gets eye position in array of trial x self.time_window
            3rd dimension of array is ordered as pursuit, learning position,
            then pursuit, learning velocity.
        """
        use_t_sets = [x for x in self.trial_sets] # Do not want to overwrite trial_sets
        use_t_sets.append(~self.fixation_trial_sets)
        pos_p, pos_l, t_inds = self.neuron.session.get_xy_traces("eye position",
                                self.time_window, self.blocks,
                                use_t_sets, return_inds=True)
        eye_data = np.stack((pos_p, pos_l), axis=2)
        if return_t_inds:
            return eye_data, t_inds
        else:
            return eye_data

    def get_fix_only_fr_traces(self, return_t_inds=False):
        """ Calls the neuron's get firing rate functions using the input blocks
        and time windows used for making the fit object, making this call
        cleaner when used below in other methods.
        """
        use_t_sets = [x for x in self.trial_sets] # Do not want to overwrite trial_sets
        use_t_sets.append(self.fixation_trial_sets)
        fr, t_inds = self.neuron.get_firing_traces(self.fix_trial_time_window,
                            self.blocks, use_t_sets, return_inds=True)
        if return_t_inds:
            return fr, t_inds
        else:
            return fr

    def get_purs_only_fr_traces(self, return_t_inds=False):
        """ Calls the neuron's get firing rate functions using the input blocks
        and time windows used for making the fit object, making this call
        cleaner when used below in other methods.
        """
        use_t_sets = [x for x in self.trial_sets] # Do not want to overwrite trial_sets
        use_t_sets.append(~self.fixation_trial_sets)
        fr, t_inds = self.neuron.get_firing_traces(self.time_window,
                            self.blocks, use_t_sets, return_inds=True)
        if return_t_inds:
            return fr, t_inds
        else:
            return fr

    def fit_4D_planes(self, knee_steps=[2.5, 0.25], bin_width=10, bin_threshold=1):
        """
        """



        print("time windows are hard coded here!")
        all_eye_data = np.empty((0, 2))
        all_fr_data = np.empty((0, ))
        fix_eye_data = np.nanmean(self.get_fix_only_eye_traces(), axis=1)
        fix_fr_data = np.nanmean(self.get_fix_only_fr_traces(), axis=1)
        all_eye_data = np.concatenate((all_eye_data, fix_eye_data), axis=0)
        all_fr_data = np.concatenate((all_fr_data, fix_fr_data), axis=0)
        purs_eye_data = np.nanmean(self.get_purs_only_eye_traces()[:, 0:200, :], axis=1)
        purs_fr_data = np.nanmean(self.get_purs_only_fr_traces()[:, 0:200], axis=1)
        all_eye_data = np.concatenate((all_eye_data, purs_eye_data), axis=0)
        all_fr_data = np.concatenate((all_fr_data, purs_fr_data), axis=0)
        purs_eye_data = np.nanmean(self.get_purs_only_eye_traces()[:, 1100:1300, :], axis=1)
        purs_fr_data = np.nanmean(self.get_purs_only_fr_traces()[:, 1100:1300], axis=1)
        all_eye_data = np.concatenate((all_eye_data, purs_eye_data), axis=0)
        all_fr_data = np.concatenate((all_fr_data, purs_fr_data), axis=0)

        # Remove any nans from data before fitting
        nan_select = np.any(np.isnan(all_eye_data), axis=1)
        all_eye_data = all_eye_data[~nan_select]
        all_fr_data = all_fr_data[~nan_select]

        knee_stop = np.ceil(np.amax(np.amax(all_eye_data, axis=0)))
        knee_start = np.floor(np.amin(np.amin(all_eye_data, axis=0)))
        if knee_steps[0] > ((knee_stop - knee_start)/2):
            raise ValueError("First knee step {0} is too large relative to the range of all eye positions {1}.".format(knee_steps[0], (knee_stop - knee_start)))
        half_knee_step = knee_steps[0] / 2
        steps = np.arange(knee_start, knee_stop + half_knee_step + 1, knee_steps[0])
        steps[-1] = knee_stop

        R2 = []
        coefficients = []
        steps_used = np.zeros((2, len(steps) * len(steps)))
        n_fit = 0
        eye_data = np.zeros((all_eye_data.shape[0], 5))
        eye_data[:, 4] = 1.
        # Loop over all potential knees in pursuit and learn axes
        for p_knee in steps:
            for l_knee in steps:
                # First 2 columns are positive/negative pursuit
                # Second 2 columns are postive/negative learning
                eye_data_select = all_eye_data[:, 0] >= p_knee
                eye_data[eye_data_select, 0] = all_eye_data[eye_data_select, 0]
                eye_data[~eye_data_select, 1] = all_eye_data[~eye_data_select, 0]
                eye_data_select = all_eye_data[:, 1] >= l_knee
                eye_data[eye_data_select, 2] = all_eye_data[eye_data_select, 1]
                eye_data[~eye_data_select, 3] = all_eye_data[~eye_data_select, 1]

                # Now fit and measure goodness
                coefficients.append(np.linalg.lstsq(eye_data, all_fr_data, rcond=None)[0])
                y_mean = np.mean(all_fr_data)
                y_predicted = np.matmul(eye_data, coefficients[-1])
                sum_squares_error = ((all_fr_data - y_predicted) ** 2).sum()
                sum_squares_total = ((all_fr_data - y_mean) ** 2).sum()
                R2.append(1 - sum_squares_error/(sum_squares_total))
                steps_used[0, n_fit] = p_knee
                steps_used[1, n_fit] = l_knee
                n_fit += 1

                # Need to reset eye_data so default is "0.0"
                eye_data[:, 0:4] = 0.0

        # Do fine resolution loop
        max_ind = np.where(R2 == np.amax(R2))[0][0]
        best_p_knee = steps_used[0, max_ind]
        # Make new steps centered on this best_p_knee
        step_start_p_knee = max(knee_start, best_p_knee - knee_steps[0])
        step_stop_p_knee = min(knee_stop, best_p_knee + knee_steps[0])
        steps_p = np.arange(step_start_p_knee, step_stop_p_knee + knee_steps[1], knee_steps[1])
        best_l_knee = steps_used[1, max_ind]
        # Make new steps centered on this best_l_knee
        step_start_l_knee = max(knee_start, best_l_knee - knee_steps[0])
        step_stop_l_knee = min(knee_stop, best_l_knee + knee_steps[0])
        steps_l = np.arange(step_start_l_knee, step_stop_l_knee + knee_steps[1], knee_steps[1])
        # Reset fit measures
        R2 = []
        coefficients = []
        steps_used = np.zeros((2, len(steps_p) * len(steps_l)))
        n_fit = 0
        # Loop over all potential knees in pursuit and learn axes
        for p_knee in steps_p:
            for l_knee in steps_l:
                # First 2 columns are positive/negative pursuit
                # Second 2 columns are postive/negative learning
                eye_data_select = all_eye_data[:, 0] >= p_knee
                eye_data[eye_data_select, 0] = all_eye_data[eye_data_select, 0]
                eye_data[~eye_data_select, 1] = all_eye_data[~eye_data_select, 0]
                eye_data_select = all_eye_data[:, 1] >= l_knee
                eye_data[eye_data_select, 2] = all_eye_data[eye_data_select, 1]
                eye_data[~eye_data_select, 3] = all_eye_data[~eye_data_select, 1]

                # Now fit and measure goodness
                coefficients.append(np.linalg.lstsq(eye_data, all_fr_data, rcond=None)[0])
                y_mean = np.mean(all_fr_data)
                y_predicted = np.matmul(eye_data, coefficients[-1])
                sum_squares_error = ((all_fr_data - y_predicted) ** 2).sum()
                sum_squares_total = ((all_fr_data - y_mean) ** 2).sum()
                R2.append(1 - sum_squares_error/(sum_squares_total))
                steps_used[0, n_fit] = p_knee
                steps_used[1, n_fit] = l_knee
                n_fit += 1

                # Need to reset eye_data so default is "0.0"
                eye_data[:, 0:4] = 0.0

        # Choose peak R2 value with minimum absolute value lag
        max_ind = np.where(R2 == np.amax(R2))[0][0]
        self.fit_results['4D_planes'] = {
                                'pursuit_knee': steps_used[0, max_ind],
                                'learning_knee': steps_used[1, max_ind],
                                'coeffs': coefficients[max_ind],
                                'R2': R2[max_ind],
                                'all_R2': R2,
                                'predict_fun': self.predict_4D_planes}

    def get_4D_planes_from_2D(self, eye_data):
        """ Converts input 2D eye data where the first column is the pursuit
        axis and second column is the learning axis into a 4D column of data
        where the first 2 columns are positive/negative pursuit relative to
        the fitted pursuit axis knee and the second 2 columns are
        positive/negative learning axis values relative to the learning knee.
        """
        X = np.zeros((eye_data.shape[0], 4))
        X_select = eye_data[:, 0] >= self.fit_results['4D_planes']['pursuit_knee']
        X[X_select, 0] = eye_data[X_select, 0]
        X[~X_select, 1] = eye_data[~X_select, 0]
        X_select = eye_data[:, 1] >= self.fit_results['4D_planes']['learning_knee']
        X[X_select, 2] = eye_data[X_select, 1]
        X[~X_select, 3] = eye_data[~X_select, 1]
        return X

    def get_4D_planes_predict_data(self, verbose=False):
        """ Gets behavioral data from blocks and trial sets and formats in a
        way that it can be used to predict firing rate according to the 4D
        planes model using predict_4D_planes. """
        print("time windows are hard coded here!")
        eye_data = np.empty((0, 2))
        fix_eye_data = np.nanmean(self.get_fix_only_eye_traces(), axis=1)
        eye_data = np.concatenate((eye_data, fix_eye_data), axis=0)
        purs_eye_data = np.nanmean(self.get_purs_only_eye_traces()[:, 0:200, :], axis=1)
        eye_data = np.concatenate((eye_data, purs_eye_data), axis=0)
        purs_eye_data = np.nanmean(self.get_purs_only_eye_traces()[:, 1100:1300, :], axis=1)
        eye_data = np.concatenate((eye_data, purs_eye_data), axis=0)
        # Remove any nans from data
        nan_select = np.any(np.isnan(eye_data), axis=1)
        eye_data = eye_data[~nan_select]
        X = np.zeros((eye_data.shape[0], 5))
        X[:, 4] = 1.
        X[:, 0:4] = self.get_4D_planes_from_2D(eye_data)
        return X

    def predict_4D_planes(self, X):
        """
        """
        if ~np.all(X[:, -1]):
            # Add column of 1's for constant
            X = np.hstack((X, np.ones((X.shape[0], 1))))
        if X.shape[1] != 5:
            raise ValueError("4D planes is fit with 4 non-constant coefficients but input data dimension is {0}.".format(X.shape[1]))
        y_hat = np.matmul(X, self.fit_results['4D_planes']['coeffs'])
        return y_hat
