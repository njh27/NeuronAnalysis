import numpy as np
from numpy import linalg as la
import warnings
from NeuronAnalysis.general import bin_xy_func_z, bin_data
from SessionAnalysis.utils import eye_data_series



def piece_wise_eye_data(eye_data, add_constant=False):
    """ Given the n observations by dims input of eye data, the data are
    expanded into a piecewise/rectified version centered on 0.
    """
    # Initialize empty eye_data array that we can fill from slices of all data
    piece_dims = eye_data.shape[1] * 2
    if add_constant:
        piece_dims += 1
    eye_data_piece = np.ones((eye_data.shape[0], piece_dims))
    # Need to copy each dim of eye_data over twice so we can get +/- pieces
    in_dim = 0
    for dim in range(0, eye_data.shape[1] * 2):
        eye_data_piece[:, dim] = eye_data[:, in_dim]
        if dim % 2 == 1:
            # Need to rectify the pieces every other loop while we are at it
            select_pursuit = eye_data[:, in_dim] >= 0.0
            eye_data_piece[~select_pursuit, dim-1] = 0.0 # Less than zero dim = 0
            eye_data_piece[select_pursuit, dim] = 0.0 # Greater than zero dim = 0
            in_dim += 1 # Increment every other loop
    return eye_data_piece


def quick_fit_piecewise_acc(firing_rate, eye_data, fit_constant=True):
        """ A quick fitting function that does not require a class or input neuron
        object but rather directly takes the firing rate and eye data to be fit
        and performs the fit without the extra overhead/management.

        Input: firing_rate and corresponding eye_data samples where dim 0 is 
        samples and dim 1 is the eye data dimensions in the order:
            position pursuit, position learning, velocity pursuit,
            velocity learning, acceleration pursuit, acceleration learning
        This function will automatically split the function to "pieces" and
        remove any "nan" data points according to eye_data.
        Output: Fitted Coefficients in the order:

        """
        eye_data_piece = piece_wise_eye_data(eye_data, add_constant=fit_constant)
        select_good = ~np.any(np.isnan(eye_data_piece), axis=1)
        select_good = select_good & ~np.isnan(firing_rate)
        eye_data_piece = eye_data_piece[select_good, :]
        firing_rate_nonan = firing_rate[select_good] # This should generally return a copy
        coefficients = np.linalg.lstsq(eye_data_piece, firing_rate_nonan, rcond=None)[0]
        y_mean = np.mean(firing_rate_nonan)
        y_predicted = np.matmul(eye_data_piece, coefficients)
        sum_squares_error = np.nansum((firing_rate_nonan - y_predicted) ** 2)
        sum_squares_total = np.nansum((firing_rate_nonan - y_mean) ** 2)
        R2 = 1 - sum_squares_error/(sum_squares_total)

        return coefficients, R2


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
        self.trial_sets = Neuron.append_valid_trial_set(trial_sets)
        self.lag_range_eye = np.array(lag_range_eye, dtype=np.int32)
        self.lag_range_slip = np.array(lag_range_slip, dtype=np.int32)
        if self.lag_range_eye[1] <= self.lag_range_eye[0]:
            raise ValueError("lag_range_eye[1] must be greater than lag_range_eye[0]")
        if self.lag_range_slip[1] <= self.lag_range_slip[0]:
            raise ValueError("lag_range_slip[1] must be greater than lag_range_slip[0]")
        self.dc_inds = np.array([dc_win[0] - time_window[0], dc_win[1] - time_window[0]], dtype=np.int32)
        self.fit_results = {}

    def get_firing_traces(self, trial_sets=None):
        """ Calls the neuron's get firing rate functions using the input blocks
        and time windows used for making the fit object, making this call
        cleaner when used below in other methods.
        """
        if trial_sets is None:
            trial_sets = self.trial_sets
        fr = self.neuron.get_firing_traces(self.time_window, self.blocks,
                            trial_sets, return_inds=False)
        return fr
    
    def get_firing_traces_fix_adj(self, trial_sets=None):
        if trial_sets is None:
            trial_sets = self.trial_sets
        fr = self.neuron.get_firing_traces_fix_adj(self.time_window, 
                                                   self.blocks, trial_sets, 
                                                   fix_time_window=self.fix_adj['fix_win'], 
                                                   sigma=self.fix_adj['sigma'], 
                                                   cutoff_sigma=self.fix_adj['cutoff_sigma'], 
                                                   zscore_sigma=self.fix_adj['zscore_sigma'], 
                                                   rate_offset=self.fix_adj['rate_offset'], 
                                                   return_inds=False)
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

    def get_eye_data_traces_all_lags(self, trial_sets=None):
        """ Gets eye position and velocity in array of trial x
        self.time_window +/- self.lag_range_eye so that all lags of data are
        pulled at once and can later be taken as slices/views for much faster
        fitting over lags and single memory usage.
        3rd dimension of array is ordered as pursuit, learning position,
        then pursuit, learning velocity.
        """
        if trial_sets is None:
            trial_sets = self.trial_sets
        lag_time_window = [self.time_window[0] + self.lag_range_eye[0],
                            self.time_window[1] + self.lag_range_eye[1]]
        pos_p, pos_l, t_inds = self.neuron.session.get_xy_traces("eye position",
                                lag_time_window, self.blocks, trial_sets,
                                return_inds=True)
        vel_p, vel_l = self.neuron.session.get_xy_traces("eye velocity",
                                lag_time_window, self.blocks, trial_sets,
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

    def no_data_to_fit(self, fit_name, n_coeffs, fit_constant, pred_fun):
        """ Set fit values to zero if there is no data found to fit
        """
        print(f"No data available in blocks {self.blocks} and trial sets {self.trial_sets} for current neuron. Cannot fit, setting coefficients to zeros.")
        n_coeffs = n_coeffs + 1 if fit_constant else n_coeffs
        self.fit_results[fit_name] = {
                                        'eye_lag': 0,
                                        'slip_lag': None,
                                        'coeffs': np.zeros((n_coeffs)),
                                        'R2': 0.,
                                        'all_R2': [0],
                                        'use_constant': fit_constant,
                                        'dc_offset': 0.,
                                        'predict_fun': pred_fun}
        return

    def fit_pcwise_lin_eye_kinematics(self, bin_width=10, bin_threshold=1,
                                fit_constant=True, fit_avg_data=False,
                                quick_lag_step=10, fit_fix_adj_fr=False,
                                fix_adj_params={}, filter_win=None):
        """ Fits the input neuron eye data to position, velocity, acceleration
        linear model (in 2 dimensions -- one pursuit axis and one learing axis)
        for the blocks and trial_sets input.
        Output "coeffs" are in order: position pursuit, position learning
                                      velocity pursuit, velocity learning
                                      acceleration pursuit, acceleration learning
                                      constant offset
        """
        if filter_win is None:
            filter_win = self.neuron.session.saccade_ind_cushion
        self.filter_win = filter_win
        if fit_avg_data and (bin_width > 1):
            raise ValueError("Cannot run with average data and use binning.")
        if fit_avg_data:
            self.avg_trial_sets = ["pursuit", "anti_pursuit", "learning", "anti_learning", "instruction"]
        fr_get_fun = self.get_firing_traces_fix_adj if fit_fix_adj_fr else self.get_firing_traces
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
        fit_constant = False
        if len(fix_adj_params) == 0:
            self.fix_adj = {'fix_win': [-300, 0],
                            'sigma': 12.5,
                            'cutoff_sigma': 4.0,
                            'zscore_sigma': 3.0,
                            'rate_offset': 0.0,
                            }
        else:
            self.fix_adj = fix_adj_params
        if fit_avg_data:
            # Get data for each trial type in trial_sets
            firing_rate = []
            for t_set in self.avg_trial_sets:
                t_set_data = fr_get_fun(t_set)
                if t_set_data.size == 0:
                    # No data so skip
                    continue
                t_set_mean = np.nanmean(t_set_data, axis=0, keepdims=True)
                firing_rate.append(t_set_mean)
            firing_rate = np.vstack(firing_rate)
        else:
            firing_rate = fr_get_fun()
        if firing_rate.size == 0:
            # No data 
            self.no_data_to_fit("pcwise_lin_eye_kinematics", 12, fit_constant, self.predict_pcwise_lin_eye_kinematics)
            return
        if not fit_constant:
            dc_trial_rate = np.mean(firing_rate[:, self.dc_inds[0]:self.dc_inds[1]], axis=1)
            firing_rate = firing_rate - dc_trial_rate[:, None]
        binned_FR = bin_data(firing_rate, bin_width, bin_threshold)

        if fit_avg_data:
            # Get data for each trial type in trial_sets
            eye_data_all_lags = []
            for t_set in self.avg_trial_sets:
                t_set_data = self.get_eye_data_traces_all_lags(t_set)
                if t_set_data.size == 0:
                    # No data so skip
                    continue
                t_set_mean = np.nanmean(t_set_data, axis=0, keepdims=True)
                eye_data_all_lags.append(t_set_mean)
            eye_data_all_lags = np.vstack(eye_data_all_lags)
        else:
            eye_data_all_lags = self.get_eye_data_traces_all_lags()
        # Initialize empty eye_data array that we can fill from slices of all data
        eye_data = np.ones((eye_data_all_lags.shape[0], self.fit_dur, 6))
        # First loop over lags using quick_lag_step intervals
        for lag in lags:
            eye_data[:, :, 0:4] = self.get_eye_lag_slice(lag, eye_data_all_lags)
            eye_data[:, :, 4:6] = eye_data_series.acc_from_vel(eye_data[:, :, 2:4],
                            filter_win=self.filter_win, axis=1)
            # Use bin smoothing on data before fitting
            bin_eye_data = bin_data(eye_data, bin_width, bin_threshold)
            bin_eye_data = bin_eye_data.reshape(bin_eye_data.shape[0]*bin_eye_data.shape[1], bin_eye_data.shape[2], order='C')
            bin_eye_data = piece_wise_eye_data(bin_eye_data, add_constant=fit_constant)

            temp_FR = binned_FR.reshape(binned_FR.shape[0]*binned_FR.shape[1], order='C')
            select_good = ~np.any(np.isnan(bin_eye_data), axis=1) & ~np.isnan(temp_FR)
            bin_eye_data = bin_eye_data[select_good, :]
            temp_FR = temp_FR[select_good]
            if temp_FR.shape[0] == 0:
                # No data after removing NaNs
                self.no_data_to_fit("pcwise_lin_eye_kinematics", 12, fit_constant, self.predict_pcwise_lin_eye_kinematics)
                return
            coefficients.append(np.linalg.lstsq(bin_eye_data, temp_FR, rcond=None)[0])
            y_mean = np.mean(temp_FR)
            y_predicted = np.matmul(bin_eye_data, coefficients[-1])
            sum_squares_error = np.nansum((temp_FR - y_predicted) ** 2)
            sum_squares_total = np.nansum((temp_FR - y_mean) ** 2)
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
                            filter_win=self.filter_win, axis=1)
                # Use bin smoothing on data before fitting
                bin_eye_data = bin_data(eye_data, bin_width, bin_threshold)
                bin_eye_data = bin_eye_data.reshape(bin_eye_data.shape[0]*bin_eye_data.shape[1], bin_eye_data.shape[2], order='C')
                bin_eye_data = piece_wise_eye_data(bin_eye_data, add_constant=fit_constant)
                
                temp_FR = binned_FR.reshape(binned_FR.shape[0]*binned_FR.shape[1], order='C')
                select_good = ~np.any(np.isnan(bin_eye_data), axis=1) & ~np.isnan(temp_FR)
                bin_eye_data = bin_eye_data[select_good, :]
                temp_FR = temp_FR[select_good]
                coefficients.append(np.linalg.lstsq(bin_eye_data, temp_FR, rcond=None)[0])
                y_mean = np.mean(temp_FR)
                y_predicted = np.matmul(bin_eye_data, coefficients[-1])
                sum_squares_error = np.nansum((temp_FR - y_predicted) ** 2)
                sum_squares_total = np.nansum((temp_FR - y_mean) ** 2)
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
                                'predict_fun': self.predict_pcwise_lin_eye_kinematics}

    def get_predict_data(self, model, blocks, trial_sets, time_window=None, verbose=False):
        if model == "pcwise_lin_eye_kinematics":
            return self.get_pcwise_lin_eye_kin_predict_data(blocks, trial_sets, time_window, verbose)
        elif model == "pcwise_lin_eye_kinematics_acc_x_vel":
            return self.get_pcwise_lin_eye_kin_predict_data_acc_x_vel(blocks, trial_sets, time_window, verbose)
        else:
            raise ValueError(f"Unrecognized fit model {model}")
    def get_pcwise_lin_eye_kin_predict_data(self, blocks, trial_sets, time_window=None, verbose=False):
        """ Gets behavioral data from blocks and trial sets and formats in a
        way that it can be used to predict firing rate according to the linear
        eye kinematic model using predict_lin_eye_kinematics. """
        if time_window is None:
            time_window = self.time_window
        lagged_eye_win = [time_window[0] + self.fit_results['pcwise_lin_eye_kinematics']['eye_lag'],
                          time_window[1] + self.fit_results['pcwise_lin_eye_kinematics']['eye_lag']
                         ]
        if verbose: print("EYE lag:", self.fit_results['pcwise_lin_eye_kinematics']['eye_lag'])
        s_dim2 = 13 if self.fit_results['pcwise_lin_eye_kinematics']['use_constant'] else 12

        trial_sets = self.neuron.append_valid_trial_set(trial_sets)
        X = np.ones((time_window[1]-time_window[0], s_dim2))
        X[:, 0], X[:, 1] = self.neuron.session.get_mean_xy_traces(
                                                "eye position", lagged_eye_win,
                                                blocks=blocks,
                                                trial_sets=trial_sets)
        X[:, 2], X[:, 3] = self.neuron.session.get_mean_xy_traces(
                                                "eye velocity", lagged_eye_win,
                                                blocks=blocks,
                                                trial_sets=trial_sets)
        X[:, 4:6] = eye_data_series.acc_from_vel(X[:,2:4], filter_win=self.filter_win, axis=0)
        X = piece_wise_eye_data(X[:, 0:6], add_constant=self.fit_results['pcwise_lin_eye_kinematics']['use_constant'])
        return X

    def predict(self, model, X):
        if model == "pcwise_lin_eye_kinematics":
            return self.predict_pcwise_lin_eye_kinematics(X)
        elif model == "pcwise_lin_eye_kinematics_acc_x_vel":
            return self.predict_pcwise_lin_eye_kinematics_acc_x_vel(X)
        else:
            raise ValueError(f"Unrecognized fit model {model}")
    def predict_pcwise_lin_eye_kinematics(self, X):
        """
        """
        if self.fit_results['pcwise_lin_eye_kinematics']['use_constant']:
            if X.shape[1] != 13:
                raise ValueError(f"Piecewise linear eye kinematics is fit with 12 non-constant coefficients and a constant. Input X.shape[1] should be 13 but is {X.shape[1]}")
        else:
            if X.shape[1] != 12:
                raise ValueError(f"Piecewise linear eye kinematics is fit with 12 non-constant coefficients and no constant. Input X.shape[1] should be 12 but is {X.shape[1]}")
        y_hat = np.matmul(X, self.fit_results['pcwise_lin_eye_kinematics']['coeffs'])
        return y_hat

    def get_data_by_trial(self, model, blocks, trial_sets, return_shape=False, return_inds=False):
        if model == "pcwise_lin_eye_kinematics":
            return self.get_pcwise_lin_eye_kin_predict_data_by_trial(blocks, trial_sets, return_shape, return_inds)
        elif model == "pcwise_lin_eye_kinematics_acc_x_vel":
            return self.get_pcwise_lin_eye_kin_predict_data_by_trial_acc_x_vel(blocks, trial_sets, return_shape, return_inds)
        else:
            raise ValueError(f"Unrecognized fit model {model}")
    def get_pcwise_lin_eye_kin_predict_data_by_trial(self, blocks, trial_sets, return_shape=False, return_inds=False):
        """ Gets behavioral data from blocks and trial sets and formats in a
        way that it can be used to predict firing rate according to the linear
        eye kinematic model using predict_lin_eye_kinematics. """
        eye_data, t_inds = self.get_eye_data_traces(blocks, trial_sets, self.fit_results['pcwise_lin_eye_kinematics']['eye_lag'], return_inds=True)
        acc_data_p = eye_data_series.acc_from_vel(eye_data[:, :, 2], filter_win=self.filter_win, axis=1)
        acc_data_l = eye_data_series.acc_from_vel(eye_data[:, :, 3], filter_win=self.filter_win, axis=1)
        eye_data = np.concatenate((eye_data, np.expand_dims(acc_data_p, axis=2), np.expand_dims(acc_data_l, axis=2)), axis=2)

        initial_shape = eye_data.shape
        eye_data = eye_data.reshape(eye_data.shape[0]*eye_data.shape[1], eye_data.shape[2], order='C')
        eye_data = piece_wise_eye_data(eye_data, add_constant=self.fit_results['pcwise_lin_eye_kinematics']['use_constant'])
        if self.fit_results['pcwise_lin_eye_kinematics']['use_constant']:
            eye_data = np.hstack((eye_data, np.ones((eye_data.shape[0], 1))))
        if return_shape and return_inds:
            return eye_data, initial_shape, t_inds
        elif return_shape and not return_inds:
            return eye_data, initial_shape
        elif not return_shape and return_inds:
            return eye_data, t_inds
        else:
            return eye_data
    
    def predict_by_trial(self, model, X, x_reshape):
        if model == "pcwise_lin_eye_kinematics":
            return self.predict_pcwise_lin_eye_kinematics_by_trial(X, x_reshape)
        elif model == "pcwise_lin_eye_kinematics_acc_x_vel":
            return self.predict_pcwise_lin_eye_kinematics_by_trial_acc_x_vel(X, x_reshape)
        else:
            raise ValueError(f"Unrecognized fit model {model}")
    def predict_pcwise_lin_eye_kinematics_by_trial(self, X, x_reshape):
        """ Predicts the response from the model trial-wise instead of for average data
        """
        if self.fit_results['pcwise_lin_eye_kinematics']['use_constant']:
            if X.shape[1] != 13:
                raise ValueError(f"Piecewise linear eye kinematics is fit with 12 non-constant coefficients and a constant. Input X.shape[1] should be 13 but is {X.shape[1]}")
        else:
            if X.shape[1] != 12:
                raise ValueError(f"Piecewise linear eye kinematics is fit with 12 non-constant coefficients and no constant. Input X.shape[1] should be 12 but is {X.shape[1]}")
        y_hat = np.matmul(X, self.fit_results['pcwise_lin_eye_kinematics']['coeffs'])
        y_hat = y_hat.reshape(x_reshape[0], x_reshape[1], order='C')
        return y_hat
    
    
    def fit_pcwise_lin_eye_kinematics_acc_x_vel(self, bin_width=10, bin_threshold=1,
                                fit_constant=True, fit_avg_data=False,
                                quick_lag_step=10, fit_fix_adj_fr=False,
                                fix_adj_params={}, filter_win=None):
        """ Fits the input neuron eye data to position, velocity, acceleration
        linear model (in 2 dimensions -- one pursuit axis and one learing axis)
        for the blocks and trial_sets input. Adds the acceleration and velocity
        interaction terms to the linear model.
        Output "coeffs" are in order: position pursuit, position learning
                                      velocity pursuit, velocity learning
                                      acceleration pursuit, acceleration learning
                                      constant offset
        """
        if filter_win is None:
            filter_win = self.neuron.session.saccade_ind_cushion
        self.filter_win = filter_win
        if fit_avg_data:
            self.avg_trial_sets = ["pursuit", "anti_pursuit", "learning", "anti_learning", "instruction"]
        fr_get_fun = self.get_firing_traces_fix_adj if fit_fix_adj_fr else self.get_firing_traces
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
        fit_constant = False
        if len(fix_adj_params) == 0:
            self.fix_adj = {'fix_win': [-300, 0],
                            'sigma': 12.5,
                            'cutoff_sigma': 4.0,
                            'zscore_sigma': 3.0,
                            'rate_offset': 0.0,
                            }
        else:
            self.fix_adj = fix_adj_params
        if fit_avg_data:
            # Get data for each trial type in trial_sets
            firing_rate = []
            for t_set in self.avg_trial_sets:
                t_set_data = fr_get_fun(t_set)
                if t_set_data.size == 0:
                    # No data so skip
                    continue
                t_set_mean = np.nanmedian(t_set_data, axis=0, keepdims=True)
                firing_rate.append(t_set_mean)
            firing_rate = np.vstack(firing_rate)
        else:
            firing_rate = fr_get_fun()
        if firing_rate.size == 0:
            # No data 
            self.no_data_to_fit("pcwise_lin_eye_kinematics_acc_x_vel", 12, fit_constant, self.predict_pcwise_lin_eye_kinematics_acc_x_vel)
            return
        if not fit_constant:
            dc_trial_rate = np.mean(firing_rate[:, self.dc_inds[0]:self.dc_inds[1]], axis=1)
            firing_rate = firing_rate - dc_trial_rate[:, None]
        binned_FR = bin_data(firing_rate, bin_width, bin_threshold)

        if fit_avg_data:
            # Get data for each trial type in trial_sets
            eye_data_all_lags = []
            for t_set in self.avg_trial_sets:
                t_set_data = self.get_eye_data_traces_all_lags(t_set)
                if t_set_data.size == 0:
                    # No data so skip
                    continue
                t_set_mean = np.nanmedian(t_set_data, axis=0, keepdims=True)
                eye_data_all_lags.append(t_set_mean)
            eye_data_all_lags = np.vstack(eye_data_all_lags)
        else:
            eye_data_all_lags = self.get_eye_data_traces_all_lags()
        # Initialize empty eye_data array that we can fill from slices of all data
        eye_data = np.ones((eye_data_all_lags.shape[0], self.fit_dur, 6))
        # First loop over lags using quick_lag_step intervals
        for lag in lags:
            eye_data[:, :, 0:4] = self.get_eye_lag_slice(lag, eye_data_all_lags)
            eye_data[:, :, 4:6] = eye_data_series.acc_from_vel(eye_data[:, :, 2:4],
                            filter_win=self.filter_win, axis=1)
            # Use bin smoothing on data before fitting
            bin_eye_data = bin_data(eye_data, bin_width, bin_threshold)
            bin_eye_data = bin_eye_data.reshape(bin_eye_data.shape[0]*bin_eye_data.shape[1], bin_eye_data.shape[2], order='C')
            bin_eye_data = piece_wise_eye_data(bin_eye_data, add_constant=fit_constant)

            # Now compute the interaction terms
            bin_eye_data = np.hstack((bin_eye_data, np.zeros((bin_eye_data.shape[0], 8))))
            bin_eye_data[:, 12] = bin_eye_data[:, 4] * bin_eye_data[:, 8]
            bin_eye_data[:, 13] = bin_eye_data[:, 4] * bin_eye_data[:, 9]
            bin_eye_data[:, 14] = bin_eye_data[:, 5] * bin_eye_data[:, 8]
            bin_eye_data[:, 15] = bin_eye_data[:, 5] * bin_eye_data[:, 9]
            bin_eye_data[:, 16] = bin_eye_data[:, 6] * bin_eye_data[:, 10]
            bin_eye_data[:, 17] = bin_eye_data[:, 6] * bin_eye_data[:, 11]
            bin_eye_data[:, 18] = bin_eye_data[:, 7] * bin_eye_data[:, 10]
            bin_eye_data[:, 19] = bin_eye_data[:, 7] * bin_eye_data[:, 11]

            temp_FR = binned_FR.reshape(binned_FR.shape[0]*binned_FR.shape[1], order='C')
            select_good = ~np.any(np.isnan(bin_eye_data), axis=1) & ~np.isnan(temp_FR)
            bin_eye_data = bin_eye_data[select_good, :]
            temp_FR = temp_FR[select_good]
            if temp_FR.shape[0] == 0:
                # No data after removing NaNs
                self.no_data_to_fit("pcwise_lin_eye_kinematics_acc_x_vel", 12, fit_constant, self.predict_pcwise_lin_eye_kinematics_acc_x_vel)
                return
            coefficients.append(np.linalg.lstsq(bin_eye_data, temp_FR, rcond=None)[0])
            y_mean = np.mean(temp_FR)
            y_predicted = np.matmul(bin_eye_data, coefficients[-1])
            sum_squares_error = np.nansum((temp_FR - y_predicted) ** 2)
            sum_squares_total = np.nansum((temp_FR - y_mean) ** 2)
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
                            filter_win=self.filter_win, axis=1)
                # Use bin smoothing on data before fitting
                bin_eye_data = bin_data(eye_data, bin_width, bin_threshold)
                bin_eye_data = bin_eye_data.reshape(bin_eye_data.shape[0]*bin_eye_data.shape[1], bin_eye_data.shape[2], order='C')
                bin_eye_data = piece_wise_eye_data(bin_eye_data, add_constant=fit_constant)

                # Now compute the interaction terms
                bin_eye_data = np.hstack((bin_eye_data, np.zeros((bin_eye_data.shape[0], 8))))
                bin_eye_data[:, 12] = bin_eye_data[:, 4] * bin_eye_data[:, 8]
                bin_eye_data[:, 13] = bin_eye_data[:, 4] * bin_eye_data[:, 9]
                bin_eye_data[:, 14] = bin_eye_data[:, 5] * bin_eye_data[:, 8]
                bin_eye_data[:, 15] = bin_eye_data[:, 5] * bin_eye_data[:, 9]
                bin_eye_data[:, 16] = bin_eye_data[:, 6] * bin_eye_data[:, 10]
                bin_eye_data[:, 17] = bin_eye_data[:, 6] * bin_eye_data[:, 11]
                bin_eye_data[:, 18] = bin_eye_data[:, 7] * bin_eye_data[:, 10]
                bin_eye_data[:, 19] = bin_eye_data[:, 7] * bin_eye_data[:, 11]
                
                temp_FR = binned_FR.reshape(binned_FR.shape[0]*binned_FR.shape[1], order='C')
                select_good = ~np.any(np.isnan(bin_eye_data), axis=1) & ~np.isnan(temp_FR)
                bin_eye_data = bin_eye_data[select_good, :]
                temp_FR = temp_FR[select_good]
                coefficients.append(np.linalg.lstsq(bin_eye_data, temp_FR, rcond=None)[0])
                y_mean = np.mean(temp_FR)
                y_predicted = np.matmul(bin_eye_data, coefficients[-1])
                sum_squares_error = np.nansum((temp_FR - y_predicted) ** 2)
                sum_squares_total = np.nansum((temp_FR - y_mean) ** 2)
                R2.append(1 - sum_squares_error/(sum_squares_total))

        # Choose peak R2 value with minimum absolute value lag
        max_ind = np.where(R2 == np.amax(R2))[0]
        max_ind = max_ind[np.argmin(np.abs(lags[max_ind]))]
        dc_offset = coefficients[max_ind][-1] if fit_constant else 0.
        self.fit_results['pcwise_lin_eye_kinematics_acc_x_vel'] = {
                                'eye_lag': lags[max_ind],
                                'slip_lag': None,
                                'coeffs': coefficients[max_ind],
                                'R2': R2[max_ind],
                                'all_R2': R2,
                                'use_constant': fit_constant,
                                'dc_offset': dc_offset,
                                'predict_fun': self.predict_pcwise_lin_eye_kinematics_acc_x_vel}

    def get_pcwise_lin_eye_kin_predict_data_acc_x_vel(self, blocks, trial_sets, time_window=None, verbose=False):
        """ Gets behavioral data from blocks and trial sets and formats in a
        way that it can be used to predict firing rate according to the linear
        eye kinematic model using predict_lin_eye_kinematics. """
        if time_window is None:
            time_window = self.time_window
        lagged_eye_win = [time_window[0] + self.fit_results['pcwise_lin_eye_kinematics_acc_x_vel']['eye_lag'],
                          time_window[1] + self.fit_results['pcwise_lin_eye_kinematics_acc_x_vel']['eye_lag']
                         ]
        if verbose: print("EYE lag:", self.fit_results['pcwise_lin_eye_kinematics_acc_x_vel']['eye_lag'])
        s_dim2 = 21 if self.fit_results['pcwise_lin_eye_kinematics_acc_x_vel']['use_constant'] else 20

        trial_sets = self.neuron.append_valid_trial_set(trial_sets)
        X = np.ones((time_window[1]-time_window[0], s_dim2))
        X[:, 0], X[:, 1] = self.neuron.session.get_mean_xy_traces(
                                                "eye position", lagged_eye_win,
                                                blocks=blocks,
                                                trial_sets=trial_sets)
        X[:, 2], X[:, 3] = self.neuron.session.get_mean_xy_traces(
                                                "eye velocity", lagged_eye_win,
                                                blocks=blocks,
                                                trial_sets=trial_sets)
        X[:, 4:6] = eye_data_series.acc_from_vel(X[:,2:4], filter_win=self.filter_win, axis=0)
        X = piece_wise_eye_data(X[:, 0:6], add_constant=self.fit_results['pcwise_lin_eye_kinematics_acc_x_vel']['use_constant'])
        # Now compute the interaction terms
        X = np.hstack((X, np.zeros((X.shape[0], 8))))
        X[:, 12] = X[:, 4] * X[:, 8]
        X[:, 13] = X[:, 4] * X[:, 9]
        X[:, 14] = X[:, 5] * X[:, 8]
        X[:, 15] = X[:, 5] * X[:, 9]
        X[:, 16] = X[:, 6] * X[:, 10]
        X[:, 17] = X[:, 6] * X[:, 11]
        X[:, 18] = X[:, 7] * X[:, 10]
        X[:, 19] = X[:, 7] * X[:, 11]
        return X

    def predict_pcwise_lin_eye_kinematics_acc_x_vel(self, X):
        """
        """
        if self.fit_results['pcwise_lin_eye_kinematics_acc_x_vel']['use_constant']:
            if X.shape[1] != 21:
                raise ValueError(f"Piecewise linear eye kinematics is fit with 12 non-constant coefficients and a constant. Input X.shape[1] should be 13 but is {X.shape[1]}")
        else:
            if X.shape[1] != 20:
                raise ValueError(f"Piecewise linear eye kinematics is fit with 12 non-constant coefficients and no constant. Input X.shape[1] should be 12 but is {X.shape[1]}")
        y_hat = np.matmul(X, self.fit_results['pcwise_lin_eye_kinematics_acc_x_vel']['coeffs'])
        return y_hat

    def get_pcwise_lin_eye_kin_predict_data_by_trial_acc_x_vel(self, blocks, trial_sets, return_shape=False, return_inds=False):
        """ Gets behavioral data from blocks and trial sets and formats in a
        way that it can be used to predict firing rate according to the linear
        eye kinematic model using predict_lin_eye_kinematics. """
        eye_data, t_inds = self.get_eye_data_traces(blocks, trial_sets, self.fit_results['pcwise_lin_eye_kinematics_acc_x_vel']['eye_lag'], return_inds=True)
        acc_data_p = eye_data_series.acc_from_vel(eye_data[:, :, 2], filter_win=self.filter_win, axis=1)
        acc_data_l = eye_data_series.acc_from_vel(eye_data[:, :, 3], filter_win=self.filter_win, axis=1)
        eye_data = np.concatenate((eye_data, np.expand_dims(acc_data_p, axis=2), np.expand_dims(acc_data_l, axis=2)), axis=2)

        initial_shape = eye_data.shape
        eye_data = eye_data.reshape(eye_data.shape[0]*eye_data.shape[1], eye_data.shape[2], order='C')
        eye_data = piece_wise_eye_data(eye_data, add_constant=self.fit_results['pcwise_lin_eye_kinematics_acc_x_vel']['use_constant'])
        # Now compute the interaction terms
        eye_data = np.hstack((eye_data, np.zeros((eye_data.shape[0], 8))))
        eye_data[:, 12] = eye_data[:, 4] * eye_data[:, 8]
        eye_data[:, 13] = eye_data[:, 4] * eye_data[:, 9]
        eye_data[:, 14] = eye_data[:, 5] * eye_data[:, 8]
        eye_data[:, 15] = eye_data[:, 5] * eye_data[:, 9]
        eye_data[:, 16] = eye_data[:, 6] * eye_data[:, 10]
        eye_data[:, 17] = eye_data[:, 6] * eye_data[:, 11]
        eye_data[:, 18] = eye_data[:, 7] * eye_data[:, 10]
        eye_data[:, 19] = eye_data[:, 7] * eye_data[:, 11]
        if self.fit_results['pcwise_lin_eye_kinematics_acc_x_vel']['use_constant']:
            eye_data = np.hstack((eye_data, np.ones((eye_data.shape[0], 1))))
        if return_shape and return_inds:
            return eye_data, initial_shape, t_inds
        elif return_shape and not return_inds:
            return eye_data, initial_shape
        elif not return_shape and return_inds:
            return eye_data, t_inds
        else:
            return eye_data
    
    def predict_pcwise_lin_eye_kinematics_by_trial_acc_x_vel(self, X, x_reshape):
        """ Predicts the response from the model trial-wise instead of for average data
        """
        if self.fit_results['pcwise_lin_eye_kinematics_acc_x_vel']['use_constant']:
            if X.shape[1] != 21:
                raise ValueError(f"Piecewise linear eye kinematics is fit with 12 non-constant coefficients and a constant. Input X.shape[1] should be 13 but is {X.shape[1]}")
        else:
            if X.shape[1] != 20:
                raise ValueError(f"Piecewise linear eye kinematics is fit with 12 non-constant coefficients and no constant. Input X.shape[1] should be 12 but is {X.shape[1]}")
        y_hat = np.matmul(X, self.fit_results['pcwise_lin_eye_kinematics_acc_x_vel']['coeffs'])
        y_hat = y_hat.reshape(x_reshape[0], x_reshape[1], order='C')
        return y_hat



    def fit_lin_eye_kinematics(self, bin_width=10, bin_threshold=1,
                                fit_constant=True, fit_avg_data=False,
                                quick_lag_step=10, ignore_acc=False):
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
            if not ignore_acc:
                eye_data[:, :, 4:6] = eye_data_series.acc_from_vel(eye_data[:, :, 2:4],
                                filter_win=self.neuron.session.saccade_ind_cushion)
            else:
                eye_data[:, :, 4:6] = 0.0
            # Use bin smoothing on data before fitting
            bin_eye_data = bin_data(eye_data, bin_width, bin_threshold)
            if fit_avg_data:
                bin_eye_data = np.nanmean(bin_eye_data, axis=0, keepdims=True)
            bin_eye_data = bin_eye_data.reshape(bin_eye_data.shape[0]*bin_eye_data.shape[1], bin_eye_data.shape[2], order='C')
            temp_FR = binned_FR.reshape(binned_FR.shape[0]*binned_FR.shape[1], order='C')
            select_good = ~np.any(np.isnan(bin_eye_data), axis=1) & ~np.isnan(temp_FR)
            bin_eye_data = bin_eye_data[select_good, :]
            temp_FR = temp_FR[select_good]
            coefficients.append(np.linalg.lstsq(bin_eye_data, temp_FR, rcond=None)[0])
            y_mean = np.mean(temp_FR)
            y_predicted = np.matmul(bin_eye_data, coefficients[-1])
            sum_squares_error = np.nansum((temp_FR - y_predicted) ** 2)
            sum_squares_total = np.nansum((temp_FR - y_mean) ** 2)
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
                if not ignore_acc:
                    eye_data[:, :, 4:6] = eye_data_series.acc_from_vel(eye_data[:, :, 2:4],
                                    filter_win=self.neuron.session.saccade_ind_cushion)
                else:
                    eye_data[:, :, 4:6] = 0.0
                # Use bin smoothing on data before fitting
                bin_eye_data = bin_data(eye_data, bin_width, bin_threshold)
                if fit_avg_data:
                    bin_eye_data = np.nanmean(bin_eye_data, axis=0, keepdims=True)
                bin_eye_data = bin_eye_data.reshape(bin_eye_data.shape[0]*bin_eye_data.shape[1], bin_eye_data.shape[2], order='C')
                temp_FR = binned_FR.reshape(binned_FR.shape[0]*binned_FR.shape[1], order='C')
                select_good = ~np.any(np.isnan(bin_eye_data), axis=1) & ~np.isnan(temp_FR)
                bin_eye_data = bin_eye_data[select_good, :]
                temp_FR = temp_FR[select_good]
                coefficients.append(np.linalg.lstsq(bin_eye_data, temp_FR, rcond=None)[0])
                y_mean = np.mean(temp_FR)
                y_predicted = np.matmul(bin_eye_data, coefficients[-1])
                sum_squares_error = np.nansum((temp_FR - y_predicted) ** 2)
                sum_squares_total = np.nansum((temp_FR - y_mean) ** 2)
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
                                'ignore_acc': ignore_acc,
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

        trial_sets = self.neuron.append_valid_trial_set(trial_sets)
        X = np.ones((self.time_window[1]-self.time_window[0], s_dim2))
        X[:, 0], X[:, 1] = self.neuron.session.get_mean_xy_traces(
                                                "eye position", lagged_eye_win,
                                                blocks=blocks,
                                                trial_sets=trial_sets)
        X[:, 2], X[:, 3] = self.neuron.session.get_mean_xy_traces(
                                                "eye velocity", lagged_eye_win,
                                                blocks=blocks,
                                                trial_sets=trial_sets)
        if not self.fit_results['lin_eye_kinematics']['ignore_acc']:
            X[:, 4:6] = eye_data_series.acc_from_vel(X[:, 2:4], filter_win=29, axis=0)
        else:
            X[:, 4:6] = 0.0
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

    def fit_pcwise_eye_slip_interaction(self, bin_width=10, bin_threshold=1,
                                    fit_constant=True, fit_avg_data=False,
                                    quick_lag_step=10, knees=[0., 0.]):
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
        s_dim2 = 17 if fit_constant else 16
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
                # Copy over velocity data to make room for positions
                eye_data[:, :, 4:6] = eye_data[:, :, 2:4]
                eye_data[:, :, 6:8] = eye_data[:, :, 2:4]
                eye_data[:, :, 2:4] = eye_data[:, :, 0:2]
                # Need copy of slip for each interaction
                eye_data[:, :, 8:10] = self.get_slip_lag_slice(slag, slip_data_all_lags)
                eye_data[:, :, 10:12] = eye_data[:, :, 8:10]
                eye_data[:, :, 12:14] = eye_data[:, :, 8:10]
                eye_data[:, :, 14:16] = eye_data[:, :, 8:10]

                # Use bin smoothing on data before fitting
                bin_eye_data = bin_data(eye_data, bin_width, bin_threshold)
                if fit_avg_data:
                    bin_eye_data = np.nanmean(eye_data, axis=0, keepdims=True)

                # Need to get the +/- position data separate AFTER BINNING AND MEAN!
                select_pursuit = bin_eye_data[:, :, 0] >= knees[0]
                bin_eye_data[~select_pursuit, 0] = 0.0 # Less than knee dim0 = 0
                bin_eye_data[select_pursuit, 2] = 0.0 # Less than knee dim2 = 0
                select_learning = bin_eye_data[:, :, 1] >= knees[1]
                bin_eye_data[~select_learning, 1] = 0.0 # Less than knee dim1 = 0
                bin_eye_data[select_learning, 3] = 0.0 # Less than knee dim3 = 0
                # Now velocity...
                select_pursuit = bin_eye_data[:, :, 4] >= 0.0
                bin_eye_data[~select_pursuit, 4] = 0.0 # Less than knee dim0 = 0
                bin_eye_data[select_pursuit, 6] = 0.0 # Less than knee dim2 = 0
                select_learning = bin_eye_data[:, :, 5] >= 0.0
                bin_eye_data[~select_learning, 5] = 0.0 # Less than knee dim1 = 0
                bin_eye_data[select_learning, 7] = 0.0 # Less than knee dim3 = 0
                # Convert slip terms to position and velocity interactions
                bin_eye_data[:, :, 8:16] *= bin_eye_data[:, :, 0:8]



                bin_eye_data = bin_eye_data.reshape(bin_eye_data.shape[0]*bin_eye_data.shape[1], bin_eye_data.shape[2], order='C')
                temp_FR = binned_FR.reshape(binned_FR.shape[0]*binned_FR.shape[1], order='C')
                select_good = ~np.any(np.isnan(bin_eye_data), axis=1) & ~np.isnan(temp_FR)
                bin_eye_data = bin_eye_data[select_good, :]
                temp_FR = temp_FR[select_good]

                coefficients.append(np.linalg.lstsq(bin_eye_data, temp_FR, rcond=None)[0])
                y_mean = np.mean(temp_FR)
                y_predicted = np.matmul(bin_eye_data, coefficients[-1])
                sum_squares_error = np.nansum((temp_FR - y_predicted) ** 2)
                sum_squares_total = np.nansum((temp_FR - y_mean) ** 2)
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
                    # Copy over velocity data to make room for positions
                    eye_data[:, :, 4:6] = eye_data[:, :, 2:4]
                    eye_data[:, :, 6:8] = eye_data[:, :, 2:4]
                    eye_data[:, :, 2:4] = eye_data[:, :, 0:2]
                    # Need copy of slip for each interaction
                    eye_data[:, :, 8:10] = self.get_slip_lag_slice(slag, slip_data_all_lags)
                    eye_data[:, :, 10:12] = eye_data[:, :, 8:10]
                    eye_data[:, :, 12:14] = eye_data[:, :, 8:10]
                    eye_data[:, :, 14:16] = eye_data[:, :, 8:10]

                    # Use bin smoothing on data before fitting
                    bin_eye_data = bin_data(eye_data, bin_width, bin_threshold)
                    if fit_avg_data:
                        bin_eye_data = np.nanmean(eye_data, axis=0, keepdims=True)

                    # Need to get the +/- position data separate AFTER BINNING AND MEAN!
                    select_pursuit = bin_eye_data[:, :, 0] >= knees[0]
                    bin_eye_data[~select_pursuit, 0] = 0.0 # Less than knee dim0 = 0
                    bin_eye_data[select_pursuit, 2] = 0.0 # Less than knee dim2 = 0
                    select_learning = bin_eye_data[:, :, 1] >= knees[1]
                    bin_eye_data[~select_learning, 1] = 0.0 # Less than knee dim1 = 0
                    bin_eye_data[select_learning, 3] = 0.0 # Less than knee dim3 = 0
                    # Now velocity...
                    select_pursuit = bin_eye_data[:, :, 4] >= 0.0
                    bin_eye_data[~select_pursuit, 4] = 0.0 # Less than knee dim0 = 0
                    bin_eye_data[select_pursuit, 6] = 0.0 # Less than knee dim2 = 0
                    select_learning = bin_eye_data[:, :, 5] >= 0.0
                    bin_eye_data[~select_learning, 5] = 0.0 # Less than knee dim1 = 0
                    bin_eye_data[select_learning, 7] = 0.0 # Less than knee dim3 = 0
                    # Convert slip terms to position and velocity interactions
                    bin_eye_data[:, :, 8:16] *= bin_eye_data[:, :, 0:8]

                    bin_eye_data = bin_eye_data.reshape(bin_eye_data.shape[0]*bin_eye_data.shape[1], bin_eye_data.shape[2], order='C')
                    temp_FR = binned_FR.reshape(binned_FR.shape[0]*binned_FR.shape[1], order='C')
                    select_good = ~np.any(np.isnan(bin_eye_data), axis=1) & ~np.isnan(temp_FR)
                    bin_eye_data = bin_eye_data[select_good, :]
                    temp_FR = temp_FR[select_good]

                    coefficients.append(np.linalg.lstsq(bin_eye_data, temp_FR, rcond=None)[0])
                    y_mean = np.mean(temp_FR)
                    y_predicted = np.matmul(bin_eye_data, coefficients[-1])
                    sum_squares_error = np.nansum((temp_FR - y_predicted) ** 2)
                    sum_squares_total = np.nansum((temp_FR - y_mean) ** 2)
                    R2.append(1 - sum_squares_error/(sum_squares_total))
                    lags_used[0, n_fit] = elag
                    lags_used[1, n_fit] = slag
                    n_fit += 1

        # Choose peak R2 value with minimum absolute value lag
        max_ind = np.where(R2 == np.amax(R2))[0][0]
        dc_offset = coefficients[max_ind][-1] if fit_constant else 0.
        self.fit_results['pcwise_eye_slip_interaction'] = {
                                'eye_lag': lags_used[0, max_ind],
                                'slip_lag': lags_used[1, max_ind],
                                'coeffs': coefficients[max_ind],
                                'R2': R2[max_ind],
                                'all_R2': R2,
                                'use_constant': fit_constant,
                                'dc_offset': dc_offset,
                                'predict_fun': self.predict_pcwise_eye_slip_interaction,
                                'knees': knees}

    def get_pcwise_eye_slip_inter_predict_data(self, blocks, trial_sets,
                                            get_avg_data=False, verbose=False):
        """ Gets behavioral data from blocks and trial sets and formats in a
        way that it can be used to predict firing rate according to the eye
        slip interaction model using predict_eye_slip_interaction. """
        slip_lagged_eye_win = [self.time_window[0] + self.fit_results['pcwise_eye_slip_interaction']['eye_lag'],
                               self.time_window[1] + self.fit_results['pcwise_eye_slip_interaction']['eye_lag']
                              ]
        slip_lagged_slip_win = [self.time_window[0] + self.fit_results['pcwise_eye_slip_interaction']['slip_lag'],
                                self.time_window[1] + self.fit_results['pcwise_eye_slip_interaction']['slip_lag']
                               ]
        if verbose: print("EYE lag:", self.fit_results['pcwise_eye_slip_interaction']['eye_lag'])
        if verbose: print("SLIP lag:", self.fit_results['pcwise_eye_slip_interaction']['slip_lag'])
        s_dim2 = 17 if self.fit_results['pcwise_eye_slip_interaction']['use_constant'] else 16

        trial_sets = self.neuron.append_valid_trial_set(trial_sets)
        if get_avg_data:
            X = np.ones((self.time_window[1]-self.time_window[0], s_dim2))
            X[:, 0], X[:, 1] = self.neuron.session.get_mean_xy_traces(
                                                    "eye position",
                                                    slip_lagged_eye_win,
                                                    blocks=blocks,
                                                    trial_sets=trial_sets)
        else:
            # We don't know how big X will be until we grab some data and find out
            x, y = self.neuron.session.get_xy_traces("eye position",
                                                        slip_lagged_eye_win,
                                                        blocks=blocks,
                                                        trial_sets=trial_sets,
                                                        return_inds=False)
            # x,y output as n_trials by time array
            X = np.ones((x.size, s_dim2))
            X[:, 0] = np.ravel(x, order='C')
            X[:, 1] = np.ravel(y, order='C')
        # Copy position for +/-
        X[:, 2:4] = X[:, 0:2]
        # Need to get the +/- position data separate
        X_select = X[:, 0] >= self.fit_results['pcwise_eye_slip_interaction']['knees'][0]
        X[~X_select, 0] = 0.0 # Less than knee dim0 = 0
        X[X_select, 2] = 0.0 # Less than knee dim2 = 0
        X_select = X[:, 1] >= self.fit_results['pcwise_eye_slip_interaction']['knees'][1]
        X[~X_select, 1] = 0.0 # Less than knee dim1 = 0
        X[X_select, 3] = 0.0 # Less than knee dim3 = 0

        if get_avg_data:
            X[:, 4], X[:, 5] = self.neuron.session.get_mean_xy_traces(
                                                    "eye velocity",
                                                    slip_lagged_eye_win,
                                                    blocks=blocks,
                                                    trial_sets=trial_sets)
        else:
            x, y = self.neuron.session.get_xy_traces("eye velocity",
                                                        slip_lagged_eye_win,
                                                        blocks=blocks,
                                                        trial_sets=trial_sets,
                                                        return_inds=False)
            X[:, 4] = np.ravel(x, order='C')
            X[:, 5] = np.ravel(y, order='C')
        # Copy velocity for +/-
        X[:, 6:8] = X[:, 4:6]
        # Need to get the +/- position data separate
        X_select = X[:, 4] >= 0.0
        X[~X_select, 4] = 0.0 # Less than knee dim0 = 0
        X[X_select, 6] = 0.0 # Less than knee dim2 = 0
        X_select = X[:, 5] >= 0.0
        X[~X_select, 5] = 0.0 # Less than knee dim1 = 0
        X[X_select, 7] = 0.0 # Less than knee dim3 = 0

        if get_avg_data:
            X[:, 8], X[:, 9] = self.neuron.session.get_mean_xy_traces(
                                                    "slip", slip_lagged_slip_win,
                                                    blocks=blocks,
                                                    trial_sets=trial_sets)
        else:
            x, y = self.neuron.session.get_xy_traces("slip",
                                                        slip_lagged_slip_win,
                                                        blocks=blocks,
                                                        trial_sets=trial_sets,
                                                        return_inds=False)
            X[:, 8] = np.ravel(x, order='C')
            X[:, 9] = np.ravel(y, order='C')
        X[:, 10:12] = X[:, 8:10]
        X[:, 12:14] = X[:, 8:10]
        X[:, 14:16] = X[:, 8:10]
        X[:, 8:16] *= X[:, 0:8]
        return X

    def predict_pcwise_eye_slip_interaction(self, X):
        """
        """
        if self.fit_results['pcwise_eye_slip_interaction']['use_constant']:
            if ~np.all(X[:, -1]):
                # Add column of 1's for constant
                X = np.hstack((X, np.ones((X.shape[0], 1))))
        if X.shape[1] != self.fit_results['pcwise_eye_slip_interaction']['coeffs'].shape[0]:
            raise ValueError("Piecewise eye slip interaction is fit with 16 non-constant coefficients but input data dimension is {0}.".format(X.shape[1]))
        y_hat = np.matmul(X, self.fit_results['pcwise_eye_slip_interaction']['coeffs'])
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
                select_good = ~np.any(np.isnan(bin_eye_data), axis=1) & ~np.isnan(temp_FR)
                bin_eye_data = bin_eye_data[select_good, :]
                temp_FR = temp_FR[select_good]

                coefficients.append(np.linalg.lstsq(bin_eye_data, temp_FR, rcond=None)[0])
                y_mean = np.mean(temp_FR)
                y_predicted = np.matmul(bin_eye_data, coefficients[-1])
                sum_squares_error = np.nansum((temp_FR - y_predicted) ** 2)
                sum_squares_total = np.nansum((temp_FR - y_mean) ** 2)
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
                    select_good = ~np.any(np.isnan(bin_eye_data), axis=1) & ~np.isnan(temp_FR)
                    bin_eye_data = bin_eye_data[select_good, :]
                    temp_FR = temp_FR[select_good]

                    coefficients.append(np.linalg.lstsq(bin_eye_data, temp_FR, rcond=None)[0])
                    y_mean = np.mean(temp_FR)
                    y_predicted = np.matmul(bin_eye_data, coefficients[-1])
                    sum_squares_error = np.nansum((temp_FR - y_predicted) ** 2)
                    sum_squares_total = np.nansum((temp_FR - y_mean) ** 2)
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

        trial_sets = self.neuron.append_valid_trial_set(trial_sets)
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
                select_good = ~np.any(np.isnan(bin_eye_data), axis=1) & ~np.isnan(temp_FR)
                bin_eye_data = bin_eye_data[select_good, :]
                temp_FR = temp_FR[select_good]
                coefficients.append(np.linalg.lstsq(bin_eye_data, temp_FR, rcond=None)[0])
                y_mean = np.mean(temp_FR)
                y_predicted = np.matmul(bin_eye_data, coefficients[-1])
                sum_squares_error = np.nansum((temp_FR - y_predicted) ** 2)
                sum_squares_total = np.nansum((temp_FR - y_mean) ** 2)
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
                    select_good = ~np.any(np.isnan(bin_eye_data), axis=1) & ~np.isnan(temp_FR)
                    bin_eye_data = bin_eye_data[select_good, :]
                    temp_FR = temp_FR[select_good]
                    coefficients.append(np.linalg.lstsq(bin_eye_data, temp_FR, rcond=None)[0])
                    y_mean = np.mean(temp_FR)
                    y_predicted = np.matmul(bin_eye_data, coefficients[-1])
                    sum_squares_error = np.nansum((temp_FR - y_predicted) ** 2)
                    sum_squares_total = np.nansum((temp_FR - y_mean) ** 2)
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

        trial_sets = self.neuron.append_valid_trial_set(trial_sets)
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

    def fit_eye_CS_interaction(self, bin_width=10, bin_threshold=1,
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
                select_good = ~np.any(np.isnan(bin_eye_data), axis=1) & ~np.isnan(temp_FR)
                bin_eye_data = bin_eye_data[select_good, :]
                temp_FR = temp_FR[select_good]

                coefficients.append(np.linalg.lstsq(bin_eye_data, temp_FR, rcond=None)[0])
                y_mean = np.mean(temp_FR)
                y_predicted = np.matmul(bin_eye_data, coefficients[-1])
                sum_squares_error = np.nansum((temp_FR - y_predicted) ** 2)
                sum_squares_total = np.nansum((temp_FR - y_mean) ** 2)
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
                    select_good = ~np.any(np.isnan(bin_eye_data), axis=1) & ~np.isnan(temp_FR)
                    bin_eye_data = bin_eye_data[select_good, :]
                    temp_FR = temp_FR[select_good]

                    coefficients.append(np.linalg.lstsq(bin_eye_data, temp_FR, rcond=None)[0])
                    y_mean = np.mean(temp_FR)
                    y_predicted = np.matmul(bin_eye_data, coefficients[-1])
                    sum_squares_error = np.nansum((temp_FR - y_predicted) ** 2)
                    sum_squares_total = np.nansum((temp_FR - y_mean) ** 2)
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

    def get_eye_CS_inter_predict_data(self, blocks, trial_sets, verbose=False):
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

        trial_sets = self.neuron.append_valid_trial_set(trial_sets)
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

    def predict_eye_CS_interaction(self, X):
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
        self.trial_sets = Neuron.append_valid_trial_set(trial_sets)
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

    def fit_4D_planes(self, knee_steps=[2.5, 0.25], bin_width=10,
                        bin_threshold=1, min_fr_step=.5):
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

        fr_edges = np.concatenate(([knee_start - knee_steps[0]], steps, [knee_stop + knee_steps[0]]))
        _, _, z_out = bin_xy_func_z(all_eye_data[:, 0], all_eye_data[:, 1],
                                all_fr_data, fr_edges, fr_edges, np.nanmedian)
        # Use gaps between firing rate bins to choose a range and step for FR
        row_diffs = np.abs(np.diff(z_out.ravel(order="C")))
        col_diffs = np.abs(np.diff(z_out.ravel(order="F")))
        fr_range = np.ceil(np.nanmax(np.concatenate((row_diffs, col_diffs))))
        fr_range = max(fr_range, 1.) # Range at least 1 Hz
        fr_step = np.floor(np.nanmin(np.concatenate((row_diffs, col_diffs))))
        fr_step = max(fr_step, 0.1) # Step at least 0.1 Hz

        R2 = []
        coefficients = []
        steps_used = np.zeros((3, len(steps) * len(steps)))
        n_fit = 0
        eye_data = np.zeros((all_eye_data.shape[0], 4))
        # eye_data = np.zeros((all_eye_data.shape[0], 5))
        # eye_data[:, 4] = 1.
        # Loop over all potential knees in pursuit and learn axes
        for p_knee in steps:
            for l_knee in steps:
                # Determine the firing rate range needed for current knees
                p_edges = [p_knee - half_knee_step, p_knee + half_knee_step]
                l_edges = [l_knee - half_knee_step, l_knee + half_knee_step]
                _, _, z_out = bin_xy_func_z(all_eye_data[:, 0], all_eye_data[:, 1],
                                        all_fr_data, p_edges, l_edges, np.nanmedian)
                if np.isnan(z_out):
                    # No firing rate data near this point so don't make it a knee
                    coefficients.append([])
                    R2.append(-np.inf)
                    steps_used[0, n_fit] = p_knee
                    steps_used[1, n_fit] = l_knee
                    n_fit += 1
                    continue
                fr_start = max(0., z_out - fr_range)
                fr_stop = z_out + fr_range + fr_step/2
                fr_steps = np.arange(fr_start, fr_stop, fr_step)
                for fr_knee in fr_steps:
                    # First 2 columns are positive/negative pursuit
                    # Second 2 columns are postive/negative learning
                    eye_data_select = all_eye_data[:, 0] >= p_knee
                    eye_data[eye_data_select, 0] = all_eye_data[eye_data_select, 0] - p_knee
                    eye_data[~eye_data_select, 1] = all_eye_data[~eye_data_select, 0] - p_knee
                    eye_data_select = all_eye_data[:, 1] >= l_knee
                    eye_data[eye_data_select, 2] = all_eye_data[eye_data_select, 1] - l_knee
                    eye_data[~eye_data_select, 3] = all_eye_data[~eye_data_select, 1] - l_knee

                    # Now fit and measure goodness
                    coefficients.append(np.linalg.lstsq(eye_data, all_fr_data - fr_knee, rcond=None)[0])
                    y_mean = np.mean(all_fr_data)
                    y_predicted = np.matmul(eye_data, coefficients[-1])
                    sum_squares_error = np.nansum((all_fr_data - y_predicted) ** 2)
                    sum_squares_total = np.nansum((all_fr_data - y_mean) ** 2)
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
        half_knee_step = knee_steps[1] / 2

        fr_edges_p = np.concatenate(([step_start_p_knee - knee_steps[1]], steps_p, [step_stop_p_knee + knee_steps[1]]))
        fr_edges_l = np.concatenate(([step_start_l_knee - knee_steps[1]], steps_l, [step_stop_l_knee + knee_steps[1]]))
        _, _, z_out = bin_xy_func_z(all_eye_data[:, 0], all_eye_data[:, 1],
                                all_fr_data, fr_edges_p, fr_edges_l, np.nanmedian)
        # Use gaps between firing rate bins to choose a range and step for FR
        row_diffs = np.abs(np.diff(z_out.ravel(order="C")))
        col_diffs = np.abs(np.diff(z_out.ravel(order="F")))
        fr_range = np.ceil(np.amax(np.concatenate((row_diffs, col_diffs))))
        fr_range = max(fr_range, 1.) # Range at least 1 Hz
        fr_step = np.floor(np.amin(np.concatenate((row_diffs, col_diffs))))
        fr_step = max(fr_step, 0.1) # Step at least 0.1 Hz

        # Reset fit measures
        R2 = []
        coefficients = []
        steps_used = np.zeros((2, len(steps_p) * len(steps_l)))
        n_fit = 0
        # Loop over all potential knees in pursuit and learn axes
        for p_knee in steps_p:
            for l_knee in steps_l:
                # Determine the firing rate range needed for current knees
                p_edges = [p_knee - half_knee_step, p_knee + half_knee_step]
                l_edges = [l_knee - half_knee_step, l_knee + half_knee_step]
                _, _, z_out = bin_xy_func_z(all_eye_data[:, 0], all_eye_data[:, 1],
                                        all_fr_data, p_edges, l_edges, np.nanmedian)
                if np.isnan(z_out):
                    # No firing rate data near this point so don't make it a knee
                    coefficients.append([])
                    R2.append(-np.inf)
                    steps_used[0, n_fit] = p_knee
                    steps_used[1, n_fit] = l_knee
                    n_fit += 1
                    continue
                fr_start = max(0., z_out - fr_range)
                fr_stop = z_out + fr_range + fr_step/2
                fr_steps = np.arange(fr_start, fr_stop, fr_step)
                for fr_knee in fr_steps:
                    # First 2 columns are positive/negative pursuit
                    # Second 2 columns are postive/negative learning
                    eye_data_select = all_eye_data[:, 0] >= p_knee
                    eye_data[eye_data_select, 0] = all_eye_data[eye_data_select, 0] - p_knee
                    eye_data[~eye_data_select, 1] = all_eye_data[~eye_data_select, 0] - p_knee
                    eye_data_select = all_eye_data[:, 1] >= l_knee
                    eye_data[eye_data_select, 2] = all_eye_data[eye_data_select, 1] - l_knee
                    eye_data[~eye_data_select, 3] = all_eye_data[~eye_data_select, 1] - l_knee

                    # Now fit and measure goodness
                    coefficients.append(np.linalg.lstsq(eye_data, all_fr_data - fr_knee, rcond=None)[0])
                y_mean = np.mean(all_fr_data)
                y_predicted = np.matmul(eye_data, coefficients[-1])
                sum_squares_error = np.nansum((all_fr_data - y_predicted) ** 2)
                sum_squares_total = np.nansum((all_fr_data - y_mean) ** 2)
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
