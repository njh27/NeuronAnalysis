import numpy as np
from numpy import linalg as la
from scipy import signal
from scipy import stats
from scipy.optimize import minimize
import warnings



def firing_rate_window(maestro_PL2_data, time_window, neuron, rate_name):
    """ Will convert time_window to be integer values.
        Returns - time_window by len(maestro_PL2_data) array of firing rate. """

    if not isinstance(maestro_PL2_data, list):
        maestro_PL2_data = [maestro_PL2_data]
    time_window[0], time_window[1] = int(time_window[0]), int(time_window[1])
    if time_window[1] <= time_window[0]:
        raise ValueError("time_window[1] must be greater than time_window[0]")

    firing_rate = np.full((time_window[1] - time_window[0], len(maestro_PL2_data)), np.nan)
    for trial in range(0, len(maestro_PL2_data)):
        time_index = maestro_PL2_data[trial]['time_series'].find_index_range(time_window[0], time_window[1])
        if time_index is None:
            # Entire trial window is beyond available data
            continue
        if round(maestro_PL2_data[trial]['time_series'].start) - time_window[0] > 0:
            # Window start is beyond available data
            out_start = round(maestro_PL2_data[trial]['time_series'].start) - time_window[0]
        else:
            out_start = 0
        if round(maestro_PL2_data[trial]['time_series'].stop) - time_window[1] < 0:
            # Window stop is beyond available data
            out_stop = firing_rate.shape[0] - (time_window[1] - round(maestro_PL2_data[trial]['time_series'].stop))
        else:
            out_stop = firing_rate.shape[0]
        output_index = np.arange(out_start, out_stop, 1)
        firing_rate[output_index, trial] = maestro_PL2_data[trial][rate_name][neuron][time_index]

    firing_rate = np.squeeze(firing_rate)
    return firing_rate


def eye_data_window(maestro_PL2_data, time_window):
    """ Will convert time_window to be integer values.
        Returns - time_window by len(maestro_PL2_data) by 4 array of eye data.
                  3rd dimension of array is ordered as horizontal, vertical
                  position, then horizontal, vertical velocity. """

    if not isinstance(maestro_PL2_data, list):
        maestro_PL2_data = [maestro_PL2_data]
    time_window[0], time_window[1] = int(time_window[0]), int(time_window[1])
    if time_window[1] <= time_window[0]:
        raise ValueError("time_window[1] must be greater than time_window[0]")

    eye_data = np.full((time_window[1] - time_window[0], len(maestro_PL2_data), 4), np.nan)
    for trial in range(0, len(maestro_PL2_data)):
        time_index = maestro_PL2_data[trial]['time_series'].find_index_range(time_window[0], time_window[1])
        if time_index is None:
            # Entire trial window is beyond available data
            continue
        if round(maestro_PL2_data[trial]['time_series'].start) - time_window[0] > 0:
            # Window start is beyond available data
            out_start = round(maestro_PL2_data[trial]['time_series'].start) - time_window[0]
        else:
            out_start = 0
        if round(maestro_PL2_data[trial]['time_series'].stop) - time_window[1] < 0:
            # Window stop is beyond available data
            out_stop = eye_data.shape[0] - (time_window[1] - round(maestro_PL2_data[trial]['time_series'].stop))
        else:
            out_stop = eye_data.shape[0]
        output_index = np.arange(out_start, out_stop, 1)
        eye_data[output_index, trial, 0:2] = maestro_PL2_data[trial]['eye_position'][:, time_index].T
        eye_data[output_index, trial, 2:4] = maestro_PL2_data[trial]['eye_velocity'][:, time_index].T

    eye_data = np.squeeze(eye_data)
    return eye_data


def slip_data_window(maestro_PL2_data, time_window):
    """ Will convert time_window to be integer values.
        Returns - time_window by len(maestro_PL2_data) by 2 array of slip data.
                  3rd dimension of array is ordered as horizontal, vertical slip. """

    if not isinstance(maestro_PL2_data, list):
        maestro_PL2_data = [maestro_PL2_data]
    time_window[0], time_window[1] = int(time_window[0]), int(time_window[1])
    if time_window[1] <= time_window[0]:
        raise ValueError("time_window[1] must be greater than time_window[0]")

    slip_data = np.full((time_window[1] - time_window[0], len(maestro_PL2_data), 2), np.nan)
    for trial in range(0, len(maestro_PL2_data)):
        time_index = maestro_PL2_data[trial]['time_series'].find_index_range(time_window[0], time_window[1])
        if time_index is None:
            # Entire trial window is beyond available data
            continue
        if round(maestro_PL2_data[trial]['time_series'].start) - time_window[0] > 0:
            # Window start is beyond available data
            out_start = round(maestro_PL2_data[trial]['time_series'].start) - time_window[0]
        else:
            out_start = 0
        if round(maestro_PL2_data[trial]['time_series'].stop) - time_window[1] < 0:
            # Window stop is beyond available data
            out_stop = slip_data.shape[0] - (time_window[1] - round(maestro_PL2_data[trial]['time_series'].stop))
        else:
            out_stop = slip_data.shape[0]
        output_index = np.arange(out_start, out_stop, 1)
        slip_data[output_index, trial, :] = maestro_PL2_data[trial]['retinal_velocity'][:, time_index].T

    slip_data = np.squeeze(slip_data)
    return slip_data


def nan_sac_data_window(maestro_PL2_data, time_window, *data):
    """ Will convert time_window to be integer values.
        data is a numpy array as output from firing_rate_window, eye_data_window
        or slip_data_window.  This will Nan the saccade times corresponding to time_window
        for every point along the third dimension.  time_window[1] - time_window[0] must
        equal data_window.shape[0]
        Returns - Original input 'data' with saccade times in time_window marked as np.nan
                  along the entire 3rd dimension of 'data'.  Each element of data can be
                  returned separately in output tuple in the order entered. """
    if not isinstance(maestro_PL2_data, list):
        maestro_PL2_data = [maestro_PL2_data]
    time_window[0], time_window[1] = int(time_window[0]), int(time_window[1])
    if time_window[1] <= time_window[0]:
        raise ValueError("time_window[1] must be greater than time_window[0]")
    n_data_out = []
    data = list(data)
    for d in range(0, len(data)):
        if time_window[1] - time_window[0] != data[d].shape[0]:
            raise ValueError("Time window must correspond to data.shape[0]")
        if data[d].ndim > 1:
            if data[d].shape[1] != len(maestro_PL2_data):
                data[d] = data[d].reshape((data[d].shape[0], -1, data[d].shape[1]))
                if data[d].shape[1] != len(maestro_PL2_data):
                    raise ValueError("Number of trials in maestro_PL2_data must equal those in data.shape[1]")
        else:
            data[d] = data[d].reshape(data[d].shape[0], 1, 1)
        if data[d].ndim < 3:
            n_data_out.append(1)
        else:
            n_data_out.append(data[d].shape[2])

    all_data = np.dstack(data)
    nan_data = np.zeros(all_data.shape[0]).astype('bool')
    for trial in range(0, len(maestro_PL2_data)):
        time_index = maestro_PL2_data[trial]['time_series'].find_index_range(time_window[0], time_window[1])
        if time_index is None:
            # Entire trial window is beyond available data
            continue
        if round(maestro_PL2_data[trial]['time_series'].start) - time_window[0] > 0:
            # Window start is beyond available data
            out_start = round(maestro_PL2_data[trial]['time_series'].start) - time_window[0]
        else:
            out_start = 0
        if round(maestro_PL2_data[trial]['time_series'].stop) - time_window[1] < 0:
            # Window stop is beyond available data
            out_stop = nan_data.shape[0] - (time_window[1] - round(maestro_PL2_data[trial]['time_series'].stop))
        else:
            out_stop = nan_data.shape[0]
        output_index = np.arange(out_start, out_stop, 1)
        nan_data[output_index] = maestro_PL2_data[trial]['saccade_index'][time_index]
        all_data[nan_data, trial, :] = np.nan

    start_ind = 0
    out_data = []
    for d in range(0, len(n_data_out)):
        out_data.append(all_data[:, :, start_ind:start_ind+n_data_out[d]])
        out_data[d] = np.squeeze(out_data[d])
        start_ind += n_data_out[d]
    if len(n_data_out) == 1:
        out_data = out_data[0]
    else:
        out_data = tuple(out_data)
    return out_data


def bin_data(data, bin_width, bin_threshold=0):
    """ Gets the nan average of each bin in data for bins in which the number
        of non nan data points is greater than bin_threshold.  Bins less than
        bin threshold non nan data points are returned as nan. Data are binned
        from the first entries, so if the number of bins implied by binwidth
        exceeds data.shape[0] the last bin will be cut short. Input data is
        assumed to have the shape as output by the data_window functions. """

    if data.ndim == 1:
        out_shape = (data.shape[0] // bin_width, 1, 1)
        data = data.reshape(data.shape[0], 1, 1)
    elif data.ndim == 2:
        out_shape = (data.shape[0] // bin_width, data.shape[1], 1)
        data = data.reshape(data.shape[0], data.shape[1], 1)
    elif data.ndim == 3:
        out_shape = (data.shape[0] // bin_width, data.shape[1], data.shape[2])
        data = data.reshape(data.shape[0], data.shape[1], data.shape[2])
    else:
        raise ValueError("Unrecognized data input shape. Input data must be in the form as output by data functions.")

    binned_data = np.full(out_shape, np.nan)
    n = 0
    bin_start = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        while n < out_shape[0]:
            n_good = np.sum(~np.isnan(data[bin_start:bin_start+bin_width, :, :]), axis=0)
            binned_data[n, :, :] = np.nanmean(data[bin_start:bin_start+bin_width, :, :], axis=0)
            binned_data[n, n_good < bin_threshold] = np.nan
            n += 1
            bin_start += bin_width

    binned_data = np.squeeze(binned_data)
    return binned_data


def acc_from_vel(velocity, filter_win):
    """ Computes the acceleration from the velocity.  Velocity is assumed to be
        in the format as output by eye_data_window and acceleration will be
        computed along the columns. """
    velocity[np.isnan(velocity)] = 0
    return signal.savgol_filter(velocity, filter_win, 1, deriv=1, axis=0) * 1000


class FitNeuronToEye(object):
    """ Class that fits neuron firing rates to eye data and is capable of
        calculating and outputting some basic info and predictions. Time window
        indicates the FIRING RATE time window, other data will be lagged relative
        to the fixed firing rate window. """

    def __init__(self, maestro_PL2_data, neuron=0, FR_name='bin_FR', time_window=[0, 400], lag_range_eye=[-25, 25], lag_range_slip=[60, 120], slip_target_num=1):
        if not isinstance(maestro_PL2_data, list):
            maestro_PL2_data = [maestro_PL2_data]
        self.data = maestro_PL2_data
        self.neuron = neuron
        self.FR_name = FR_name
        self.time_window = np.array(time_window)
        self.lag_range_eye = lag_range_eye
        self.lag_range_slip = lag_range_slip
        self.slip_target_num = slip_target_num
        self.fit_result = {}
        self.fit_result['model_type'] = None
        self.FR = None
        self.eye = None
        self.slip = None

    def do_eye_lags(self, lag_range_eye):
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
        self.slip = nan_sac_data_window(self.data, self.time_window + self.fit_result['eye_lag'], self.slip)
        self.slip = bin_data(self.slip, bin_width, bin_threshold=0)
        self.slip = self.slip.reshape(self.slip.shape[0]*self.slip.shape[1], self.slip.shape[2], order='F')
        self.slip = self.slip[np.all(~np.isnan(self.slip), axis=1), :]

    def set_FR_fit_data(self, bin_width=1):
        self.FR = firing_rate_window(self.data, self.time_window, self.neuron, self.FR_name)
        self.FR = nan_sac_data_window(self.data, self.time_window + self.fit_result['eye_lag'], self.FR)
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

        self.fit_result['eye_lag'] = lags[max_ind]
        self.fit_result['slip_lag'] = None
        self.fit_result['coeffs'] = coefficients[max_ind]
        self.fit_result['R2'] = R2[max_ind]
        self.fit_result['all_R2'] = R2
        self.fit_result['model_type'] = 'piece_linear'
        self.fit_result['use_constant'] = constant
        self.set_FR_fit_data(bin_width)
        self.set_eye_fit_data(self.fit_result['eye_lag'], bin_width)

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
        if 'eye_lag' not in self.fit_result:
            eye_lag = 0
        else:
            eye_lag = self.fit_result['eye_lag']

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
        self.fit_result['slip_lag'] = slip_lags[max_ind]
        self.fit_result['model_type'] = 'slip_lag'
        self.fit_result['R2'] = R2
        self.fit_result['smooth_R2'] = smooth_R2

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
        self.fit_result['eye_lag'] = eye_lags[max_ind]
        self.fit_result['model_type'] = 'eye_lag'
        self.fit_result['R2'] = R2
        self.fit_result['smooth_R2'] = smooth_R2

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

        self.fit_result['eye_lag'] = eye_lag
        self.fit_result['slip_lag'] = slip_lag
        self.fit_result['coeffs'] = coefficients
        self.fit_result['R2'] = R2
        self.fit_result['model_type'] = 'piece_linear'
        self.fit_result['use_constant'] = constant
        self.set_FR_fit_data(bin_width)
        self.set_slip_fit_data(self.fit_result['slip_lag'], bin_width)

    def predict_piece_linear(self, x_predict):

        if self.fit_result['model_type'] != "piece_linear":
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

        if self.fit_result['use_constant']:
            piece_eye = np.zeros((x_predict.shape[0], 2 * x_predict.shape[1] + 1))
            piece_eye[:, -1] = 1
        else:
            piece_eye = np.zeros((x_predict.shape[0], 2 * x_predict.shape[1]))
        for column in range(0, x_predict.shape[1]):
            plus_index = x_predict[:, column] >= 0
            piece_eye[plus_index, column] = x_predict[plus_index, column]
            piece_eye[~plus_index, column + x_predict.shape[1]] = x_predict[~plus_index, column]

        y_hat = np.matmul(piece_eye, self.fit_result['coeffs'])

        return y_hat


    def predict_piece_linear_interaction(self, x_predict):

        if self.fit_result['model_type'] != "piece_linear":
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

        if self.fit_result['use_constant']:
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

        y_hat = np.matmul(piece_slip, self.fit_result['coeffs'])

        return y_hat


def get_eye_data(maestro_PL2_data, time_window, eye_lag, rate_name, neuron, bin_width=1, avg=None):

    time_window = np.array(time_window)

    firing_rate = firing_rate_window(maestro_PL2_data, time_window, neuron, rate_name)
    eye_data = eye_data_window(maestro_PL2_data, time_window + eye_lag)
    eye_data = np.dstack((eye_data, acc_from_vel(eye_data[:, :, 2:4], max(maestro_PL2_data[0]['saccade_time_cushion'] - 1, 9))))
    firing_rate, eye_data = nan_sac_data_window(maestro_PL2_data, time_window + eye_lag, firing_rate, eye_data)
    firing_rate = bin_data(firing_rate, bin_width, bin_threshold=0)
    eye_data = bin_data(eye_data, bin_width, bin_threshold=0)

    # return firing_rate, eye_data

    if avg is not None:
        if avg == 'trial':
            avg_axis = 1
        elif avg == 'time':
            avg_axis = 0
        else:
            raise ValueError("Unrecognized input for 'avg'.  Must be None, 'trial', or 'time'")
        # Supress the nanmean warning "RuntimeWarning: Mean of empty slice" that is generated if all values on axis=avg_axis are nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            position = np.nanmean(eye_data[:, :, 0:2], axis=avg_axis)
            velocity = np.nanmean(eye_data[:, :, 2:4], axis=avg_axis)
            acceleration = np.nanmean(eye_data[:, :, 4:6], axis=avg_axis)
            firing_rate = np.nanmean(firing_rate, axis=avg_axis)
    else:
        position = eye_data[:, :, 0:2]
        velocity = eye_data[:, :, 2:4]
        acceleration = eye_data[:, :, 4:6]

    bin_centers = np.arange(time_window[0], time_window[1], bin_width) + bin_width/2
    return position, velocity, acceleration, firing_rate, bin_centers


def get_slip_data(maestro_PL2_data, time_window, eye_lag, slip_lag, rate_name, neuron, bin_width, avg=None):

    time_window = np.array(time_window)

    firing_rate = firing_rate_window(maestro_PL2_data, time_window, neuron, rate_name)
    eye_data = eye_data_window(maestro_PL2_data, time_window + eye_lag)
    slip_data = slip_data_window(maestro_PL2_data, time_window + slip_lag)
    firing_rate, eye_data, slip_data = nan_sac_data_window(maestro_PL2_data, time_window + eye_lag, firing_rate, eye_data, slip_data)
    firing_rate = bin_data(firing_rate, bin_width, bin_threshold=0)
    eye_data = bin_data(eye_data, bin_width, bin_threshold=0)
    slip_data = bin_data(slip_data, bin_width, bin_threshold=0)

    if avg is not None:
        if avg == 'trial':
            avg_axis = 1
        elif avg == 'time':
            avg_axis = 0
        else:
            raise ValueError("Unrecognized input for 'avg'.  Must be None, 'trial', or 'time'")
        # Supress the nanmean warning "RuntimeWarning: Mean of empty slice" that is generated if all values on axis=avg_axis are nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            position = np.nanmean(eye_data[:, :, 0:2], axis=avg_axis)
            velocity = np.nanmean(eye_data[:, :, 2:4], axis=avg_axis)
            slip = np.nanmean(slip_data, axis=avg_axis)
            firing_rate = np.nanmean(firing_rate, axis=avg_axis)
    else:
        position = eye_data[:, :, 0:2]
        velocity = eye_data[:, :, 2:4]
        slip = slip_data

    bin_centers = np.arange(time_window[0], time_window[1], bin_width) + bin_width/2
    return position, velocity, slip, firing_rate, bin_centers


def subtract_trial_fix_FR(maestro_PL2_data, fix_win, eye_lag, rate_name, neuron):
    position, _, _, firing_rate, _ = get_eye_data(maestro_PL2_data, fix_win, eye_lag, rate_name, neuron, avg='time')
    pos_coeffs = np.linalg.lstsq(np.hstack((position, np.ones((position.shape[0], 1)))), firing_rate, rcond=None)[0]
    exp_fix_FR = np.matmul(np.hstack((position, np.ones((position.shape[0], 1)))), pos_coeffs)
    for trial in range(0, len(maestro_PL2_data)):
        maestro_PL2_data[trial][rate_name][neuron] = maestro_PL2_data[trial][rate_name][neuron] - exp_fix_FR[trial]

    return maestro_PL2_data


def adjust_trial_FR_range(maestro_PL2_data, fix_win, eye_lag, rate_name, neuron):
    _, _, _, fix_rates, _ = get_eye_data(maestro_PL2_data, fix_win, eye_lag, rate_name, neuron, avg='time')
    if np.any(np.isnan(fix_rates)):
        raise RuntimeError('At least one fix rate is nan in this window, I should FIX THIS')
    sigma = 10
    gauss_filter = signal.gaussian(sigma*3*2 + 1, sigma)
    gauss_filter = gauss_filter / np.sum(gauss_filter)
    pad_fix_rates = np.concatenate((np.repeat(np.mean(fix_rates[0:sigma*3]), sigma*3), fix_rates, np.repeat(np.mean(fix_rates[-sigma*3:]), sigma*3)))
    est_fix_rate = np.convolve(pad_fix_rates, gauss_filter, mode='valid')
    mean_fix_rate = np.mean(est_fix_rate)
    for trial in range(0, len(maestro_PL2_data)):
        maestro_PL2_data[trial][rate_name][neuron] = (maestro_PL2_data[trial][rate_name][neuron] - est_fix_rate[trial]) * (mean_fix_rate / est_fix_rate[trial])

    return maestro_PL2_data


def learning_actual_vs_expected_FR(maestro_PL2_data, neuron, rate_name, learn_trial_name, neuron_fit, time_window=[100, 300], n_bin_size=50):
    """ Returns the actual and expected firing rate in bins defined by the number of LEARNING TRIALS that have occurred.
        e.g. the first bin with n_bin_size=50 will contain firing rates for each trial condition that occurred within
        the first 50 learning trials, regardless of the total number of trials that had occurred. Data are returned
        as dictionaries with each trial name as a key, and each key contains a numpy array where axis=0 corresponds
        to timepoints and axis=1 corresponds to bin number.  Data in each element are the np.nanmean of all data
        in the corresponding trial bins. """

    time_window = np.array(time_window)

    firing_rate = firing_rate_window(maestro_PL2_data, time_window, neuron, rate_name)
    eye_data = eye_data_window(maestro_PL2_data, time_window + neuron_fit.fit_result['eye_lag'])
    slip_data = slip_data_window(maestro_PL2_data, time_window + neuron_fit.fit_result['slip_lag'])
    firing_rate, eye_data, slip_data = nan_sac_data_window(maestro_PL2_data, time_window + neuron_fit.fit_result['eye_lag'], firing_rate, eye_data, slip_data)

    n_learn = 0
    n_bin = 0
    act_rate = {}
    exp_rate = {}
    for trial in range(0, len(maestro_PL2_data)):
        if maestro_PL2_data[trial]['trial_name'] not in act_rate.keys():
            act_rate[maestro_PL2_data[trial]['trial_name']] = [[] for x in range(0, n_bin + 1)]
            exp_rate[maestro_PL2_data[trial]['trial_name']] = [[] for x in range(0, n_bin + 1)]

        if maestro_PL2_data[trial]['trial_name'] == learn_trial_name:
            n_learn += 1
            if n_learn % n_bin_size == 0:
                n_bin += 1
                for name in act_rate.keys():
                    act_rate[name].append([])
                    exp_rate[name].append([])

        position = eye_data[:, trial, 0:2]
        velocity = eye_data[:, trial, 2:4]
        slip = slip_data[:, trial, 0:2]
        act_rate[maestro_PL2_data[trial]['trial_name']][n_bin].append(firing_rate[:, trial])
        exp_rate[maestro_PL2_data[trial]['trial_name']][n_bin].append(neuron_fit.predict_piece_linear_interaction(np.hstack((position, velocity, slip))))

    # The following averages each of the bins above and transforms data from above lists to numpy arrays
    for trial_name in act_rate.keys():
        for bin in range(0, len(act_rate[trial_name])):
            if len(act_rate[trial_name][bin]) > 0:
                # Supress the nanmean warning "RuntimeWarning: Mean of empty slice" that is generated if all values on axis=0 are nan
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    act_rate[trial_name][bin] = np.nanmean(np.array(act_rate[trial_name][bin]), axis=0)
                    exp_rate[trial_name][bin] = np.nanmean(np.array(exp_rate[trial_name][bin]), axis=0)
            else:
                act_rate[trial_name][bin] = np.full(time_window[1] - time_window[0], np.nan)
                exp_rate[trial_name][bin] = np.full(time_window[1] - time_window[0], np.nan)

        act_rate[trial_name] = np.array(act_rate[trial_name]).T
        exp_rate[trial_name] = np.array(exp_rate[trial_name]).T

    return act_rate, exp_rate


def trial_actual_vs_expected_FR(maestro_PL2_data, neuron_fit, time_window=[100, 300], n_bin_size=25):
    """ Returns the actual and expected firing rate in bins defined by the number of trials, regardless of their type,
        that have occurred.  In contrast to "learning_actual_vs_expected_FR" above, this function bins by the number of
        trials without regard to the number of learning trials.  Although trials are binned by the total number of any
        trial name, THEY ARE BINNED BY EACH TRIAL NAME! """

    n_bin = 0
    act_rate = {}
    exp_rate = {}
    for trial in range(0, len(maestro_PL2_data)):
        if maestro_PL2_data[trial]['trial_name'] not in act_rate.keys():
            act_rate[maestro_PL2_data[trial]['trial_name']] = [[] for x in range(0, n_bin + 1)]
            exp_rate[maestro_PL2_data[trial]['trial_name']] = [[] for x in range(0, n_bin + 1)]

        if trial % n_bin_size == 0 and trial != 0:
            n_bin += 1
            for name in act_rate.keys():
                # if len(act_rate[name]) <= n_bin:
                act_rate[name].append([])
                exp_rate[name].append([])

        position, velocity, acceleration = get_average_eye([maestro_PL2_data[trial]], time_window=time_window)
        act_rate[maestro_PL2_data[trial]['trial_name']][n_bin].append(get_average_ISI_FR([maestro_PL2_data[trial]], time_window=time_window+neuron_fit.fit_result['lag']))
        exp_rate[maestro_PL2_data[trial]['trial_name']][n_bin].append(neuron_fit.predict_piece_linear(np.hstack((position, velocity, acceleration))))

    # The following averages each of the bins above and transforms data from above lists to numpy arrays
    for trial_name in act_rate.keys():
        for bin in range(0, len(act_rate[trial_name])):
            if len(act_rate[trial_name][bin]) > 0:
                # Supress the nanmean warning "RuntimeWarning: Mean of empty slice" that is generated if all values on axis=0 are nan
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    act_rate[trial_name][bin] = np.nanmean(np.array(act_rate[trial_name][bin]), axis=0)
                    exp_rate[trial_name][bin] = np.nanmean(np.array(exp_rate[trial_name][bin]), axis=0)
            else:
                act_rate[trial_name][bin] = np.full(time_window[1] - time_window[0], np.nan)
                exp_rate[trial_name][bin] = np.full(time_window[1] - time_window[0], np.nan)

        act_rate[trial_name] = np.array(act_rate[trial_name]).T
        exp_rate[trial_name] = np.array(exp_rate[trial_name]).T

    return act_rate, exp_rate


def any_actual_vs_expected_FR(maestro_PL2_data, neuron_fit, time_window=[100, 300], n_bin_size=25):
    """ Returns the actual and expected firing rate in bins defined by the number of trials, regardless of their type,
        that have occurred.  ALL TRIAL TYPES ARE BINNED TOGETHER AS ONE IN THE SAME! """

    n_bin = 0
    act_rate = [[]]
    exp_rate = [[]]
    for trial in range(0, len(maestro_PL2_data)):
        if trial % n_bin_size == 0 and trial != 0:
            n_bin += 1
            act_rate[name].append([])
            exp_rate[name].append([])

        position, velocity, acceleration = get_average_eye([maestro_PL2_data[trial]], time_window=time_window)
        act_rate[maestro_PL2_data[trial]['trial_name']][n_bin].append(get_average_ISI_FR([maestro_PL2_data[trial]], time_window=time_window+neuron_fit.fit_result['lag']))
        exp_rate[maestro_PL2_data[trial]['trial_name']][n_bin].append(neuron_fit.predict_piece_linear(np.hstack((position, velocity, acceleration))))

    return act_rate, exp_rate
