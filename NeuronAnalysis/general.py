import numpy as np
import operator
from scipy.optimize import leastsq



class Indexer(object):
    """ Class that allows you to move an index along points in a 1D vector
    of numpy or list values according to a specified relationship between
    the values in the array so you can track values as needed without having to
    search an entire array each time, but rather start and stop at points of
    interest. Current index of None results in searching from scratch and finding
    the absolute first value satisfying criteria.
    """
    def __init__(self, search_array, current_index=None):
        self.search_array = search_array
        self.search_len = len(search_array)
        if self.search_len < 1:
            # Only one item, cannot be a next index
            raise ValueError("Cannot perform index searches on less than 1 value")
        self.set_current_index(current_index)
        # Make dictionary to find functions for input relation operator
        self.ops = {'>': operator.gt,
               '<': operator.lt,
               '>=': operator.ge,
               '<=': operator.le,
               '=': operator.eq}

    def set_current_index(self, index):
        if index is None:
            self.current_index = None
            return
        index = int(index)
        if np.abs(index) >= self.search_len:
            raise ValueError("Current index is out of bounds of search array of len {0}.".format(self.search_len))
        self.current_index = index
        # Adjust negative index to positive
        if self.current_index < 0:
            self.current_index = self.current_index + self.search_len

    """ This is to find an index in a numpy array or Python list search item to
        index the NEXT element in the array that holds the input "relation" with
        the input "value".  It will NOT output the value of current_index even if
        it matches value!  This search is done starting at "current_index" and
        outputs an absolute index adjusted for this starting point. The search
        will not wrap around from the end to the beginning, but instead will
        terminate and return None if the end is reached without finding the
        correct value. If a negative current_index is input, the function will
        attempt it to a positive index and output a positive index."""
    def find_index_next(self, value, relation='='):
        if self.current_index is None:
            return self.find_first_value(value, relation)
        next_index = None
        # Check for index of matching value in search_item
        for index, vals in zip(range(self.current_index + 1, self.search_len, 1), self.search_array[self.current_index + 1:]):
            try:
                if self.ops[relation](vals, value):
                    next_index = index
                    break
            except KeyError:
                raise ValueError("Input 'relation' must be '>', '<', '>=', '<=' or '=', but {0} was given.".format(relation))
        return(next_index)

    """ This is to move an index in a numpy array or Python list search item to
        index the PREVIOUS element in the array that holds the input "relation" with
        the input "value".  It will NOT output the value of current_index even if
        it matches value!  This search is done starting at "current_index" and
        outputs an absolute index adjusted for this starting point.  The search
        will not wrap around from the beginning to the end, but instead will
        terminate and return None if the beginning is reached without finding the
        correct value. If a negative current_index is input, the function will
        attempt it to a positive index and output a positive index."""
    def find_index_previous(self, value, relation='='):
        if self.current_index is None:
            return self.find_first_value(value, relation)
        previous_index = None
        # Check for index of matching value in search_item
        for index, vals in zip(range(self.current_index - 1, -1, -1), self.search_array[self.current_index - 1::-1]):
            try:
                if self.ops[relation](vals, value):
                    previous_index = index
                    break
            except KeyError:
                raise ValueError("Input 'relation' must be '>', '<', '>=', '<=' or '=', but {0} was given.".format(relation))
        return(previous_index)

    """ This is to find the index of first occurence of some value in a numpy array
        or python list that satisfies the input relation with the input value.
        Returns None if value isn't found, else returns it's index. """
    def find_first_value(self, value, relation='='):
        index_out = None
        # Check if search_array is iterable, and if not assume it is scalar and check equality
        try:
            for index, vals in enumerate(self.search_array):
                try:
                    if self.ops[relation](vals, value):
                        index_out = index
                        break
                except KeyError:
                    raise ValueError("Input 'relation' must be '>', '<', '>=', '<=' or '=', but {0} was given.".format(relation))
        except TypeError:
            try:
                if self.ops[relation](self.search_array, value):
                    index_out = 0
            except KeyError:
                raise ValueError("Input 'relation' must be '>', '<', '>=', '<=' or '=', but {0} was given.".format(relation))
        return(index_out)

    """ This is to move the current index to the index returned by
    find_next_index. Does nothing if None is found."""
    def move_index_next(self, value, relation='='):
        next_index = self.find_index_next(value, relation)
        self.set_current_index(next_index)
        return next_index

    """ This is to move the current index to the index returned by
    find_prevous_index. Does nothing if None is found."""
    def move_index_previous(self, value, relation='='):
        previous_index = self.find_index_previous(value, relation)
        self.set_current_index(previous_index)
        return previous_index

    """ This is to move the current index to the index returned by
    find_first_value. Does nothing if None is found."""
    def move_index_first_value(self, value, relation='='):
        index_out = self.find_first_value(value, relation)
        self.set_current_index(index_out)
        return index_out


def zero_phase_kernel(x, x_center):
    """ Zero pads the 1D kernel x, so that it is aligned with the current element
        of x located at x_center.  This ensures that convolution with the kernel
        x will be zero phase with respect to x_center.
    """

    kernel_offset = x.size - 2 * x_center - 1 # -1 To center ON the x_center index
    kernel_size = np.abs(kernel_offset) + x.size
    if kernel_size // 2 == 0: # Kernel must be odd
        kernel_size -= 1
        kernel_offset -= 1
    kernel = np.zeros(kernel_size)
    if kernel_offset > 0:
        kernel_slice = slice(kernel_offset, kernel.size)
    elif kernel_offset < 0:
        kernel_slice = slice(0, kernel.size + kernel_offset)
    else:
        kernel_slice = slice(0, kernel.size)
    kernel[kernel_slice] = x

    return kernel


def gauss_convolve(data, sigma, cutoff_sigma=4, pad_data=True):
    """ Uses Gaussian kernel to smooth "data" with width cutoff_sigma"""
    if cutoff_sigma > 0.5*len(data):
        raise ValueError("{0} data points is not enough for cutoff sigma of {1}.".format(len(data), cutoff_sigma))
    x_win = int(np.around(sigma * cutoff_sigma))
    xvals = np.arange(-1 * x_win, x_win + 1)
    kernel = np.exp(-.5 * (xvals / sigma) ** 2)
    kernel = kernel / np.sum(kernel)
    kernel = zero_phase_kernel(kernel, x_win)
    if pad_data:
        padded = np.hstack([[data[0]]*int(np.ceil(cutoff_sigma)), data, [data[-1]]*int(np.ceil(cutoff_sigma))])
        convolved_data = np.convolve(padded, kernel, mode='same')
        convolved_data = convolved_data[cutoff_sigma:-cutoff_sigma]
    else:
        convolved_data = np.convolve(data, kernel, mode='same')

    return convolved_data


def postsynaptic_decay_FR(spike_train, tau_rise=1., tau_decay=2.5,
                            kernel_area=1.0, min_val=0.0, reverse=False):
    """Convolves binned spike train values using the postsynaptic exponential
        rise and decay method with a CAUSAL kernel.
        Can be scaled to have have larger integral under kernel by
        "kernel_area" or to add a constant value of minimal decay using
        "min_val". """

    # Build kernel over 'xvals'
    xvals = np.arange(0, len(spike_train))
    kernel = np.exp(- 1 * xvals / tau_decay) - np.exp(- 1 * xvals / tau_rise)
    kernel = kernel_area * kernel / np.sum(kernel)

    # Pad the input data with zeros
    pad_size = len(kernel) - 1
    if reverse:
        kernel = kernel[-1::-1]
        padded_x = np.pad(spike_train, (0, pad_size), mode='constant')
    else:
        padded_x = np.pad(spike_train, (pad_size, 0), mode='constant')

    psp_decay_FR = np.convolve(padded_x, kernel, mode='valid') + min_val

    return psp_decay_FR


def gaussian(x, mu, sigma):
    return (1 / (np.sqrt(2 * np.pi * sigma ** 2))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def assymetric_CS_LTD(spike_train, sigma_rise, sigma_decay, kernel_area=1.0, min_val=0.0):
   # Parameters for the asymmetric Gaussian-like kernel
    peak_position = len(spike_train) // 2

    # Generate x-values
    xvals_left = np.arange(0, peak_position)
    xvals_right = np.arange(peak_position, len(spike_train))

    # Calculate the Gaussian functions for each side of the peak
    gaussian_left = gaussian(xvals_left, peak_position, sigma_rise)
    gaussian_right = gaussian(xvals_right, peak_position, sigma_decay)

    # Normalize the gaussians to be the same at peak_position
    gaussian_left /= gaussian_left[-1]
    gaussian_right /= gaussian_right[0]

    # Combine the left and right Gaussian functions into a single asymmetric kernel
    kernel = np.concatenate((gaussian_left[:-1], gaussian_right))

    # Normalize the kernel
    kernel = kernel_area * kernel / np.sum(kernel)

    # Convolve the asymmetric kernel with the input data
    filtered_x = np.convolve(spike_train, kernel, mode='same') + min_val

    return filtered_x


def cart2pol(x, y):
    theta = np.arctan2(y, x)
    rho = np.sqrt(x**2 + y**2)

    return theta, rho


def compute_empirical_cdf(x):

    x = np.array(x)
    cdf = [0, 0]
    unique_vals, unique_counts = np.unique(x, return_counts=True)
    cdf[0] = unique_vals
    cdf[1] = np.cumsum(unique_counts) / x.size

    return cdf


def central_difference(x, y):

    cdiffs = np.zeros(x.size)
    cdiffs[1:-1] = (y[2:] - y[0:-2]) / 2.
    cdiffs[0] = cdiffs[1]
    cdiffs[-1] = cdiffs[-2]

    return cdiffs


def bin_x_func_y(x, y, bin_edges, y_func=np.mean, y_func_args=[], y_func_kwargs={}):
    """ Bins data in y according to values in x as in a histogram, but instead
        of counts this function calls the function 'y_func' on the binned values
        of y. Returns the center point of the binned x values and the result of
        the function y_func(binned y data). """
    if len(bin_edges) < 2:
        raise ValueError("Must specify at least 2 edges to define at least 1 bin.")
    n_bins = len(bin_edges) - 1
    x_out = np.empty(n_bins)
    y_out = np.empty(n_bins)
    for edge in range(0, n_bins):
        x_out[edge] = bin_edges[edge] + (bin_edges[edge+1] - bin_edges[edge]) / 2
        x_index = np.logical_and(x >= bin_edges[edge], x < bin_edges[edge+1])
        y_out[edge] = y_func(y[x_index], *y_func_args, **y_func_kwargs)

    return x_out, y_out


def bin_xy_func_z(x, y, z, bin_edges_x, bin_edges_y, z_func=np.mean,
                    z_func_args=[], z_func_kwargs={}, empty_val=np.nan):
    """ Bins data in z according to values in (x,y) as in a 2D histogram, and
        calls the function 'z_func' on the binned values of z. Returns the
        center points of the binned (x, y) values and the result of
        the function z_func(binned z data). 2D version of bin_x_func_y above. """
    if (len(bin_edges_x) < 2) or (len(bin_edges_y) < 2):
        raise ValueError("Must specify at least 2 edges to define at least 1 bin.")
    if (len(x) != len(z)) or (len(y) != len(z)):
        raise ValueError("x, y, and z must all be the arrays of the same length indicating the 3D point for all observations.")
    n_bins_x = len(bin_edges_x) - 1
    n_bins_y = len(bin_edges_y) - 1
    x_out = np.empty(n_bins_x, dtype=np.int64)
    y_out = np.empty(n_bins_y, dtype=np.int64)
    z_out = np.zeros((n_bins_x, n_bins_y))
    for edge_x in range(0, n_bins_x):
        x_out[edge_x] = bin_edges_x[edge_x] + (bin_edges_x[edge_x+1] - bin_edges_x[edge_x]) / 2
        x_index = np.logical_and(x >= bin_edges_x[edge_x], x < bin_edges_x[edge_x+1])
        for edge_y in range(0, n_bins_y):
            if edge_x == 0:
                # We only need to do this once for y edges
                y_out[edge_y] = bin_edges_y[edge_y] + (bin_edges_y[edge_y+1] - bin_edges_y[edge_y]) / 2
            y_index = np.logical_and(y >= bin_edges_y[edge_y], y < bin_edges_y[edge_y+1])
            xy_index = x_index & y_index
            if ~np.any(xy_index):
                z_out[edge_x, edge_y] = empty_val
            else:
                z_out[edge_x, edge_y] = z_func(z[xy_index], *z_func_args, **z_func_kwargs)

    return x_out, y_out, z_out


def fit_cos_fixed_freq(x_val, y_val):
    """ Fit cosine function to x and y data assuming fixed frequency of 1. """
    y_mean = np.mean(y_val)
    y_amp = (np.amax(y_val) - np.amin(y_val)) / 2
    optimize_func = lambda x: (x[0] * (np.cos(x_val + x[1])) + x[2]) - y_val
    amp, phase, offset = leastsq(optimize_func, [y_amp, 0, y_mean])[0]

    if amp < 0:
        phase += np.pi
        amp *= -1

    return amp, phase, offset
