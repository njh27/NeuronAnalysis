import numpy as np
from numpy import linalg as la
from scipy.optimize import minimize
import warnings
from SessionAnalysis.utils import eye_data_series



def bin_data(data, bin_width, bin_threshold=0):
    """ Gets the nan average of each bin in data for bins in which the number
        of non nan data points is greater than bin_threshold.  Bins less than
        bin threshold non nan data points are returned as nan. Data are binned
        from the first entries, so if the number of bins implied by binwidth
        exceeds data.shape[0] the last bin will be cut short. Input data is
        assumed to have the shape as output by get_eye_data_traces,
        trial x time x variable and are binned along the time axis.
        bin_threshold must be <= bin_width. """
    if bin_width <= 1:
        # Nothing to bin just output
        return data
    if bin_threshold > bin_data:
        raise ValueError("bin_threshold cannot exceed the bin_width")

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

    binned_data = np.squeeze(binned_data)
    return binned_data


class FitNeuronToEye(object):
    """ Class that fits neuron firing rates to eye data and is capable of
        calculating and outputting some basic info and predictions. Time window
        indicates the FIRING RATE time window, other data will be lagged relative
        to the fixed firing rate window. """

    def __init__(self, Neuron, time_window=[0, 800], blocks=None, trial_sets=None,
                    lag_range_eye=[-25, 25], lag_range_slip=[60, 120],
                    use_series=None, slip_target_num=1):
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
        self.lag_range_eye = lag_range_eye
        self.lag_range_slip = lag_range_slip
        self.slip_target_num = slip_target_num
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
        """ Returns - time_window by len(maestro_PL2_data) by 4 array of eye data.
                      3rd dimension of array is ordered as pursuit, learning
                      position, then pursuit, learning velocity.
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
        eye_data = np.squeeze(eye_data)
        return eye_data

    def fit_lin_eye_kinematics(self, bin_width=10, bin_threshold=1,
                                fit_constant=True):
        """ Fits the input neuron eye data to position, velocity, acceleration
        linear model (in 2 dimensions -- one pursuit axis and one learing axis)
        for the blocks and trial_sets input.
        """
        lags = np.arange(self.lag_range_eye[0], self.lag_range_eye[1] + 1)
        R2 = []
        coefficients = []
        firing_rate = self.get_firing_traces()
        for lag in lags:
            eye_data = self.get_eye_data_traces(lag)
            acc_data = eye_data_series.acc_from_vel(eye_data[:, :, 2:4],
                            filter_win=self.neuron.session.saccade_ind_cushion)
            eye_data = np.concatenate((eye_data, acc_data), axis=2)
            # Use bin smoothing on data before fitting
            eye_data = bin_data(eye_data, bin_width, bin_threshold)
            temp_FR = bin_data(firing_rate, bin_width, bin_threshold)
            eye_data = eye_data.reshape(eye_data.shape[0]*eye_data.shape[1], eye_data.shape[2], order='C')
            temp_FR = temp_FR.reshape(temp_FR.shape[0]*temp_FR.shape[1], order='C')
            select_good = ~np.any(np.isnan(eye_data), axis=1)
            eye_data = eye_data[select_good, :]
            temp_FR = temp_FR[select_good]
            if fit_constant:
                # Add column of 1's for fitting constants
                eye_data = np.hstack((eye_data, np.ones((eye_data.shape[0], 1))))
            coefficients.append(np.linalg.lstsq(eye_data, temp_FR, rcond=None)[0])
            y_mean = np.mean(temp_FR)
            y_predicted = np.matmul(eye_data, coefficients[-1])
            sum_squares_error = ((temp_FR - y_predicted) ** 2).sum()
            sum_squares_total = ((temp_FR - y_mean) ** 2).sum()
            R2.append(1 - sum_squares_error/(sum_squares_total))

        # Choose peak R2 value with minimum absolute value lag
        max_ind = np.where(R2 == np.amax(R2))[0]
        max_ind = max_ind[np.argmin(np.abs(lags[max_ind]))]
        self.fit_results['lin_eye_kinematics'] = {
                                'eye_lag': lags[max_ind],
                                'slip_lag': None,
                                'coeffs': coefficients[max_ind],
                                'R2': R2[max_ind],
                                'all_R2': R2,
                                'use_constant': fit_constant,
                                'predict_fun': self.predict_lin_eye_kinematics}
        # self.set_FR_fit_data(bin_width)
        # self.set_eye_fit_data(self.fit_results['eye_lag'], bin_width)

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
