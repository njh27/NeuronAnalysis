import numpy as np
from numpy import linalg as la
from scipy.optimize import curve_fit
import warnings
from NeuronAnalysis.general import bin_xy_func_z
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


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def binary_param(p):
    return np.around(sigmoid(p))

# Define Gaussian function
def gaussian(x, mu, sigma, scale):
    return scale * np.exp(-( ((x - mu) ** 2) / (2*(sigma**2))) )


# Define the model function as a linear combination of Gaussian functions
def gaussian_basis_set(x, scales, fixed_means, fixed_sigma):
    num_gaussians = len(fixed_means)
    result = np.zeros_like(x)
    for i in range(num_gaussians):
        result += gaussian(x, fixed_means[i], fixed_sigma, scales[i])
    return result

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



class FitCSLearningFun(object):
    """ Class that fits neuron firing rates to eye data and is capable of
        calculating and outputting some basic info and predictions. Time window
        indicates the FIRING RATE time window, other data will be lagged relative
        to the fixed firing rate window. """

    def __init__(self, Neuron, time_window=[0, 800], blocks=None, trial_sets=None,
                    lag_range_pf=[-25, 25], lag_range_slip=[60, 120],
                    dc_win=[0, 100], use_series=None):
        print("CURRENTLY DOES NOT USE BEHAVIOR DATA FROM VALID NEURON TRIALS WHEN GETTING DATA FOR FIT PREDICTIONS")
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
        self.lag_range_pf = np.array(lag_range_pf, dtype=np.int32)
        self.lag_range_slip = np.array(lag_range_slip, dtype=np.int32)
        if self.lag_range_pf[1] <= self.lag_range_pf[0]:
            raise ValueError("lag_range_pf[1] must be greater than lag_range_pf[0]")
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

    def get_eye_data_traces(self, blocks, trial_sets, lag=0):
        """ Gets eye position and velocity in array of trial x self.time_window
            3rd dimension of array is ordered as pursuit, learning position,
            then pursuit, learning velocity.
        """
        lag_time_window = self.time_window + np.int32(lag)
        if lag_time_window[1] <= lag_time_window[0]:
            raise ValueError("time_window[1] must be greater than time_window[0]")

        pos_p, pos_l = self.neuron.session.get_xy_traces("eye position",
                                lag_time_window, blocks, trial_sets,
                                return_inds=False)
        vel_p, vel_l = self.neuron.session.get_xy_traces("eye velocity",
                                lag_time_window, blocks, trial_sets,
                                return_inds=False)
        eye_data = np.stack((pos_p, pos_l, vel_p, vel_l), axis=2)
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

    def fit_gauss_basis_kinematics(self, n_gaussians, std_gaussians, pos_range,
                                    vel_range, bin_width=10, bin_threshold=1,
                                    fit_avg_data=False, p0=None,
                                    quick_lag_step=10):
        """ Fits the input neuron eye data to position and velocity using a
        basis set of Gaussians according to the input number of Gaussians over
        the state space ranges specified by pos/vel _range.
        Output "coeffs" are in order the order of the n_gaussians for position
        followed by the n_gaussians for velocity.
        """
        do_fine = True
        ftol=1e-3
        xtol=1e-2
        gtol=1e-4
        max_nfev=10000
        loss='linear'

        if n_gaussians % 2 == 0:
            print("Adding a gaussian to make an odd number of Gaussians with 1 centered at zero.")
            n_gaussians += 1

        # if n_gaussians % 2 == 0:
        #     print("Adding a gaussian to make an odd number of Gaussians with 1 centered at zero.")
        #     n_gaussians += 1
        #
        # pos_fixed_means = np.linspace(pos_range[0], pos_range[1], n_gaussians)
        # print(pos_fixed_means)
        #
        # pos_range = [-10, 10]
        # pos_fixed_means = np.concatenate( (np.linspace(pos_range[0], 0, n_gaussians), np.linspace(0, pos_range[1], n_gaussians)) )
        # print(pos_fixed_means)

        quick_lag_step = int(np.around(quick_lag_step))
        if quick_lag_step < 1:
            raise ValueError("quick_lag_step must be positive integer")
        if quick_lag_step > (self.lag_range_pf[1] - self.lag_range_pf[0]):
            raise ValueError("quick_lag_step is too large relative to lag_range_pf")
        half_lag_step = np.int32(np.around(quick_lag_step / 2))
        lags_pf = np.arange(self.lag_range_pf[0], self.lag_range_pf[1] + half_lag_step + 1, quick_lag_step)
        lags_pf[-1] = self.lag_range_pf[1]
        lags_mli = np.copy(lags_pf)

        R2 = []
        coefficients = []
        lags_used = np.zeros((2, len(lags_pf) * len(lags_mli)), dtype=np.int64)
        n_fit = 0
        firing_rate = self.get_firing_traces()
        if fit_avg_data:
            firing_rate = np.nanmean(firing_rate, axis=0, keepdims=True)
        binned_FR = bin_data(firing_rate, bin_width, bin_threshold)
        FR_select = ~np.isnan(binned_FR)
        FR_select = FR_select.reshape(FR_select.shape[0]*FR_select.shape[1], order='C')
        eye_data_all_lags = self.get_eye_data_traces_all_lags()
        # Initialize empty eye_data array that we can fill from slices of all data
        eye_data = np.ones((eye_data_all_lags.shape[0], self.fit_dur, 8))

        # Set up the basic values and fit function for basis set
        # Use inputs to set these variables and keep in scope for wrapper
        pos_fixed_means = np.linspace(pos_range[0], pos_range[1], n_gaussians)
        vel_fixed_means = np.linspace(vel_range[0], vel_range[1], n_gaussians)
        # all_fixed_means = np.vstack((pos_fixed_means, vel_fixed_means))
        pos_fixed_std = std_gaussians
        vel_fixed_std = std_gaussians
        # Wrapper function for curve_fit using our fixed gaussians, means, sigmas....
        def pc_model_response_fun(x, *params):
            """ Defines the model we are fitting to the data """
            # First add the intrinsic rate/offset parameter
            result = np.zeros(x.shape[0]) + params[-1]
            # Then sum over the basis set predictors
            for k in range(4):
                use_means = pos_fixed_means if k < 2 else vel_fixed_means
                use_std = pos_fixed_std if k < 2 else vel_fixed_std
                result += gaussian_basis_set(x[:, k], params[k * n_gaussians:(k + 1) * n_gaussians],
                                                                    use_means, use_std)
            # Finally add in the relu predictors
            for k_ind, k in enumerate(range(4, 8)):
                a = params[4 * n_gaussians + k_ind * 2]
                b = params[4 * n_gaussians + k_ind * 2 + 1]
                result += downward_relu(x[:, k], a, b, c=0.)
            return result

        if p0 is None:
            # curve_fit seems unable to figure out how many parameters without setting this
            p0 = np.ones(4*n_gaussians + 4*2 + 1)
            p0[-1] = 75
        # Set lower and upper bounds for each parameter
        lower_bounds = np.zeros(p0.shape)
        upper_bounds = np.inf * np.ones(p0.shape)
        lower_bounds[0:4*n_gaussians] = 0.
        upper_bounds[0:4*n_gaussians] = 500.
        lower_bounds[4*n_gaussians:4*n_gaussians + 4*2] = np.array([0, -np.inf, 0, -np.inf, 0, -np.inf, 0, -np.inf])
        upper_bounds[4*n_gaussians:4*n_gaussians + 4*2] = np.array([1, np.inf, 1, np.inf, 1, np.inf, 1, np.inf])
        lower_bounds[-1] = 10
        upper_bounds[-1] = 200

        # First loop over lags_pf using quick_lag_step intervals
        for plag in lags_pf:
            for mlag in lags_mli:
                # if ( (plag % 5 == 0) and (mlag % 5 == 0) ):
                # if (mlag % 5 == 0):
                #     print("current pf lag and mli lag:", plag, mlag)


                eye_data[:, :, 0:4] = self.get_eye_lag_slice(plag, eye_data_all_lags)
                eye_data[:, :, 4:8] = self.get_eye_lag_slice(mlag, eye_data_all_lags)
                # Use bin smoothing on data before fitting
                bin_eye_data = bin_data(eye_data, bin_width, bin_threshold)
                if fit_avg_data:
                    bin_eye_data = np.nanmean(bin_eye_data, axis=0, keepdims=True)
                # Reshape to 2D matrices and remove nans
                bin_eye_data = bin_eye_data.reshape(bin_eye_data.shape[0]*bin_eye_data.shape[1], bin_eye_data.shape[2], order='C')
                temp_FR = binned_FR.reshape(binned_FR.shape[0]*binned_FR.shape[1], order='C')
                select_good = np.logical_and(~np.any(np.isnan(bin_eye_data), axis=1), FR_select)
                bin_eye_data = bin_eye_data[select_good, :]
                temp_FR = temp_FR[select_good]

                # Fit the Gaussian basis set to the data
                popt, pcov = curve_fit(pc_model_response_fun, bin_eye_data,
                                        temp_FR, p0=p0,
                                        bounds=(lower_bounds, upper_bounds),
                                        ftol=ftol,
                                        xtol=xtol,
                                        gtol=gtol,
                                        max_nfev=max_nfev,
                                        loss=loss)

                # Store this for now so we can call predict_gauss_basis_kinematics
                # below for computing R2. This will be overwritten with optimal
                # values at the end when we are done
                self.fit_results['gauss_basis_kinematics'] = {
                                        'pf_lag': plag,
                                        'mli_lag': mlag,
                                        'coeffs': popt,
                                        'n_gaussians': n_gaussians,
                                        'pos_means': pos_fixed_means,
                                        'pos_stds': pos_fixed_std,
                                        'vel_means': vel_fixed_means,
                                        'vel_stds': vel_fixed_std,
                                        'R2': None,
                                        'predict_fun': self.predict_gauss_basis_kinematics}
                # Compute R2 and save with coefficients for selecting best later
                coefficients.append(popt)
                y_mean = np.mean(temp_FR)
                y_predicted = self.predict_gauss_basis_kinematics(bin_eye_data)
                sum_squares_error = np.nansum((temp_FR - y_predicted) ** 2)
                sum_squares_total = np.nansum((temp_FR - y_mean) ** 2)
                R2.append(1 - sum_squares_error/(sum_squares_total))
                lags_used[0, n_fit] = plag
                lags_used[1, n_fit] = mlag
                n_fit += 1

        if ( (quick_lag_step > 1) and (do_fine) ):
            # Do fine resolution loop
            max_ind = np.where(R2 == np.amax(R2))[0][0]
            best_pf_lag = lags_used[0, max_ind]
            # Make new lags_pf centered on this best_pf_lag
            lag_start_pf = max(lags_pf[0], best_pf_lag - quick_lag_step)
            lag_stop_pf = min(lags_pf[-1], best_pf_lag + quick_lag_step)
            lags_pf = np.arange(lag_start_pf, lag_stop_pf + 1, 1)
            best_mli_lag = lags_used[1, max_ind]
            # Make new lags_mli centered on this best_mli_lag
            lag_start_mli = max(lags_mli[0], best_mli_lag - quick_lag_step)
            lag_stop_mli = min(lags_mli[-1], best_mli_lag + quick_lag_step)
            lags_mli = np.arange(lag_start_mli, lag_stop_mli + 1, 1)
            # Reset fit measures
            R2 = []
            coefficients = []
            lags_used = np.zeros((2, len(lags_pf) * len(lags_mli)), dtype=np.int64)
            n_fit = 0
            for plag in lags_pf:
                for mlag in lags_mli:
                    eye_data[:, :, 0:4] = self.get_eye_lag_slice(plag, eye_data_all_lags)
                    eye_data[:, :, 4:8] = self.get_eye_lag_slice(mlag, eye_data_all_lags)
                    # Use bin smoothing on data before fitting
                    bin_eye_data = bin_data(eye_data, bin_width, bin_threshold)
                    if fit_avg_data:
                        bin_eye_data = np.nanmean(bin_eye_data, axis=0, keepdims=True)
                    # Reshape to 2D matrices and remove nans
                    bin_eye_data = bin_eye_data.reshape(bin_eye_data.shape[0]*bin_eye_data.shape[1], bin_eye_data.shape[2], order='C')
                    temp_FR = binned_FR.reshape(binned_FR.shape[0]*binned_FR.shape[1], order='C')
                    select_good = ~np.any(np.isnan(bin_eye_data), axis=1)
                    bin_eye_data = bin_eye_data[select_good, :]
                    temp_FR = temp_FR[select_good]

                    # Fit the Gaussian basis set to the data
                    popt, pcov = curve_fit(pc_model_response_fun, bin_eye_data,
                                            temp_FR, p0=p0,
                                            bounds=(lower_bounds, upper_bounds),
                                            ftol=ftol,
                                            xtol=xtol,
                                            gtol=gtol,
                                            max_nfev=max_nfev,
                                            loss=loss)

                    # Store this for now so we can call predict_gauss_basis_kinematics
                    # below for computing R2. This will be overwritten with optimal
                    # values at the end when we are done
                    self.fit_results['gauss_basis_kinematics'] = {
                                            'pf_lag': plag,
                                            'mli_lag': mlag,
                                            'coeffs': popt,
                                            'n_gaussians': n_gaussians,
                                            'pos_means': pos_fixed_means,
                                            'pos_stds': pos_fixed_std,
                                            'vel_means': vel_fixed_means,
                                            'vel_stds': vel_fixed_std,
                                            'R2': None,
                                            'predict_fun': self.predict_gauss_basis_kinematics}
                    # Compute R2 and save with coefficients for selecting best later
                    coefficients.append(popt)
                    y_mean = np.mean(temp_FR)
                    y_predicted = self.predict_gauss_basis_kinematics(bin_eye_data)
                    sum_squares_error = np.nansum((temp_FR - y_predicted) ** 2)
                    sum_squares_total = np.nansum((temp_FR - y_mean) ** 2)
                    R2.append(1 - sum_squares_error/(sum_squares_total))
                    lags_used[0, n_fit] = plag
                    lags_used[1, n_fit] = mlag
                    n_fit += 1


        # Choose peak R2 value with minimum absolute value lag
        max_ind = np.where(R2 == np.amax(R2))[0][0]
        # Get output in fit_results dict so we can use the model later
        self.fit_results['gauss_basis_kinematics'] = {
                                'pf_lag': lags_used[0, max_ind],
                                'mli_lag': lags_used[1, max_ind],
                                'coeffs': coefficients[max_ind],
                                'n_gaussians': n_gaussians,
                                'pos_means': pos_fixed_means,
                                'pos_stds': pos_fixed_std,
                                'vel_means': vel_fixed_means,
                                'vel_stds': vel_fixed_std,
                                'R2': R2[max_ind],
                                'predict_fun': self.predict_gauss_basis_kinematics}
        return

    def get_gauss_basis_kinematics_predict_data_trial(self, blocks, trial_sets,
                                                      verbose=False,
                                                      return_shape=False):
        """ Gets behavioral data from blocks and trial sets and formats in a
        way that it can be used to predict firing rate according to the linear
        eye kinematic model using predict_lin_eye_kinematics. """
        eye_data_pf = self.get_eye_data_traces(blocks, trial_sets,
                            self.fit_results['gauss_basis_kinematics']['pf_lag'])
        eye_data_mli = self.get_eye_data_traces(blocks, trial_sets,
                            self.fit_results['gauss_basis_kinematics']['mli_lag'])
        if verbose: print("PF lag:", self.fit_results['gauss_basis_kinematics']['pf_lag'])
        if verbose: print("MLI lag:", self.fit_results['gauss_basis_kinematics']['mli_lag'])
        eye_data = np.stack((eye_data_pf, eye_data_mli), axis=2)
        initial_shape = eye_data.shape
        eye_data = eye_data.reshape(eye_data.shape[0]*eye_data.shape[1], eye_data.shape[2], order='C')
        if return_shape:
            return eye_data, initial_shape
        else:
            return eye_data

    def get_gauss_basis_kinematics_predict_data_mean(self, blocks, trial_sets, verbose=False):
        """ Gets behavioral data from blocks and trial sets and formats in a
        way that it can be used to predict firing rate according to the linear
        eye kinematic model using predict_lin_eye_kinematics. """
        lagged_pf_win = [self.time_window[0] + self.fit_results['gauss_basis_kinematics']['pf_lag'],
                          self.time_window[1] + self.fit_results['gauss_basis_kinematics']['pf_lag']
                         ]
        lagged_mli_win = [self.time_window[0] + self.fit_results['gauss_basis_kinematics']['mli_lag'],
                          self.time_window[1] + self.fit_results['gauss_basis_kinematics']['mli_lag']
                         ]
        if verbose: print("PF lag:", self.fit_results['gauss_basis_kinematics']['pf_lag'])
        if verbose: print("MLI lag:", self.fit_results['gauss_basis_kinematics']['mli_lag'])
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

    def predict_gauss_basis_kinematics_by_trial(self, blocks, trial_sets, verbose=False):
        """
        """
        X, init_shape = self.get_gauss_basis_kinematics_predict_data_trial(
                                blocks, trial_sets, verbose, return_shape=True)
        y_hat = self.predict_gauss_basis_kinematics(X)
        y_hat = y_hat.reshape(init_shape[0], init_shape[1], order='C')
        return y_hat



    def stuff(self):
        pass
