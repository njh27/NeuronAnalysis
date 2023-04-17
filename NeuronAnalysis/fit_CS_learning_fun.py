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



class FitCSLearningFun(object):
    """ Class that fits neuron firing rates to eye data and is capable of
        calculating and outputting some basic info and predictions. Time window
        indicates the FIRING RATE time window, other data will be lagged relative
        to the fixed firing rate window. """

    def __init__(self, Neuron, time_window=[0, 800], blocks=None, trial_sets=None,
                    lag_range_eye=[-25, 25], lag_range_slip=[60, 120],
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

    def get_eye_data_traces(self):
        """ Gets eye position and velocity in array of trial x self.time_window
            3rd dimension of array is ordered as pursuit, learning position,
            then pursuit, learning velocity.
        """
        pos_p, pos_l, t_inds = self.neuron.session.get_xy_traces("eye position",
                                self.time_window, self.blocks, self.trial_sets,
                                return_inds=True)
        vel_p, vel_l = self.neuron.session.get_xy_traces("eye velocity",
                                self.time_window, self.blocks, self.trial_sets,
                                return_inds=False)
        eye_data = np.stack((pos_p, pos_l, vel_p, vel_l), axis=2)
        return eye_data

    def get_slip_data_traces(self):
        """ Returns - time_window by len(maestro_PL2_data) by 2 array of
        retinal slip data. 3rd dimension of array is ordered as pursuit,
        learning slip.
        """
        slip_p, slip_l, t_inds = self.neuron.session.get_xy_traces("slip",
                                self.time_window, self.blocks, self.trial_sets,
                                return_inds=True)
        slip_data = np.stack((slip_p, slip_l), axis=2)
        return slip_data

    def fit_gauss_basis_kinematics(self, n_gaussians, std_gaussians, pos_range,
                                    vel_range, bin_width=10, bin_threshold=1,
                                    fit_avg_data=False, p0=None):
        """ Fits the input neuron eye data to position and velocity using a
        basis set of Gaussians according to the input number of Gaussians over
        the state space ranges specified by pos/vel _range.
        Output "coeffs" are in order the order of the n_gaussians for position
        followed by the n_gaussians for velocity.
        """
        firing_rate = self.get_firing_traces()
        if fit_avg_data:
            firing_rate = np.nanmean(firing_rate, axis=0, keepdims=True)
        binned_FR = bin_data(firing_rate, bin_width, bin_threshold)
        eye_data = self.get_eye_data_traces()
        # Use bin smoothing on data before fitting
        bin_eye_data = bin_data(eye_data, bin_width, bin_threshold)
        if fit_avg_data:
            bin_eye_data = np.nanmean(bin_eye_data, axis=0, keepdims=True)

        # Reshape to 2D matrices and remove nans
        bin_eye_data = bin_eye_data.reshape(bin_eye_data.shape[0]*bin_eye_data.shape[1], bin_eye_data.shape[2], order='C')
        binned_FR = binned_FR.reshape(binned_FR.shape[0]*binned_FR.shape[1], order='C')
        select_good = ~np.any(np.isnan(bin_eye_data), axis=1)
        bin_eye_data = bin_eye_data[select_good, :]
        binned_FR = binned_FR[select_good]

        # Use inputs to set these variables and keep in scope for wrapper
        pos_fixed_means = np.linspace(pos_range[0], pos_range[1], n_gaussians)
        vel_fixed_means = np.linspace(vel_range[0], vel_range[1], n_gaussians)
        # all_fixed_means = np.vstack((pos_fixed_means, vel_fixed_means))
        pos_fixed_std = std_gaussians
        vel_fixed_std = std_gaussians
        # Wrapper function for curve_fit using our fixed gaussians, means, sigmas....
        def wrapper_gaussian_basis_set(x, *scales):
            result = np.zeros(x.shape[1])
            for k in range(4):
                use_means = pos_fixed_means if k < 2 else vel_fixed_means
                use_std = pos_fixed_std if k < 2 else vel_fixed_std
                result += gaussian_basis_set(x[k, :], scales[k * n_gaussians:(k + 1) * n_gaussians],
                                                                    use_means, use_std)
            return result

        if p0 is None:
            # curve_fit seems unable to figure out how many parameters without setting this
            p0 = np.ones(4*n_gaussians)
        # Fit the Gaussian basis set to the data
        popt, pcov = curve_fit(wrapper_gaussian_basis_set, bin_eye_data.T, binned_FR, p0=p0)

        # Get output in fit_results dict so we can use the model later
        self.fit_results['gauss_basis_kinematics'] = {
                                'coeffs': popt,
                                'n_gaussians': n_gaussians,
                                'pos_means': pos_fixed_means,
                                'pos_stds': pos_fixed_std,
                                'vel_means': vel_fixed_means,
                                'vel_stds': vel_fixed_std,
                                'R2': None,
                                'predict_fun': self.predict_gauss_basis_kinematics}
        y_mean = np.mean(binned_FR)
        y_predicted = self.predict_gauss_basis_kinematics(bin_eye_data)
        sum_squares_error = ((binned_FR - y_predicted) ** 2).sum()
        sum_squares_total = ((binned_FR - y_mean) ** 2).sum()
        R2 = 1 - sum_squares_error/(sum_squares_total)
        self.fit_results['gauss_basis_kinematics']['R2'] = R2
        return


    def get_gauss_basis_kinematics_predict_data(self, blocks, trial_sets, verbose=False):
        """ Gets behavioral data from blocks and trial sets and formats in a
        way that it can be used to predict firing rate according to the linear
        eye kinematic model using predict_lin_eye_kinematics. """
        X = np.ones((self.time_window[1]-self.time_window[0], 4))
        X[:, 0], X[:, 1] = self.neuron.session.get_mean_xy_traces(
                                                "eye position", self.time_window,
                                                blocks=blocks,
                                                trial_sets=trial_sets)
        X[:, 2], X[:, 3] = self.neuron.session.get_mean_xy_traces(
                                                "eye velocity", self.time_window,
                                                blocks=blocks,
                                                trial_sets=trial_sets)
        return X

    def predict_gauss_basis_kinematics(self, X):
        """
        """
        if X.shape[1] != 4:
            raise ValueError("Gaussian basis kinematics model is fit for 4 data dimensions but input data dimension is {0}.".format(X.shape[1]))
        scales = self.fit_results['gauss_basis_kinematics']['coeffs']
        n_gaussians = self.fit_results['gauss_basis_kinematics']['n_gaussians']
        y_hat = np.zeros(X.shape[0])
        for k in range(4):
            use_means = self.fit_results['gauss_basis_kinematics']['pos_means'] if k < 2 else self.fit_results['gauss_basis_kinematics']['vel_means']
            use_std = self.fit_results['gauss_basis_kinematics']['pos_stds'] if k < 2 else self.fit_results['gauss_basis_kinematics']['vel_stds']
            y_hat += gaussian_basis_set(X[:, k], scales[k * n_gaussians:(k + 1) * n_gaussians],
                                                                use_means, use_std)
        return y_hat




    def stuff(self):
        pass
