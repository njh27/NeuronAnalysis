""" Defines a bunch of useful functions for creating unit layer activations
or basis sets like gaussian, sigmoid, cos.. """

import numpy as np



# Weird special case rectified linear functions
def negative_relu(x, c=0.):
    """ Basic relu function but returns negative result. """
    return -1*np.maximum(0., x-c)

def reflected_negative_relu(x, c=0.):
    """ Basic relu function but returns negative result, reflected about y axis. """
    return np.minimum(0., x-c)


# Define cosine functions
def cosine(x, freq, phase, amp=1.0, offset=0.0):
    return amp * np.cos(freq * (x - phase)) + offset

def cosine_activation(x, fixed_freqs, fixed_phases):
    num_cos = len(fixed_freqs)
    x_transform = np.zeros((x.size, num_cos))
    for k in range(num_cos):
        x_transform[:, k] = cosine(x, fixed_freqs[k], fixed_phases[k], amp=1.0, offset=1.0)
    return x_transform

def gen_linspace_cos(max_min_freq, n_cos, fixed_phase=True):
    """ Returns n_cos freq values spanning the range max_min_freq. If fixed_phase
    is true then all have phase 0. If fixed phase is false, then phases
    linearly spaced from 0 to pi are used such that the total number of
    cosine parameters returned is n_cos^2."""
    try:
        if len(max_min_freq) > 2:
            raise ValueError("max_min_freq must be 1 or 2 elements specifing the max and min frequency for the cosines.")
    except TypeError:
        # Happens if max_min_freq does not have "len" method, usually because it's a singe number
        max_min_freq = [0.25, max_min_freq]
    # Frequencey must be > 0
    max_min_freq[0] = max(0.25, max_min_freq[0])
    if fixed_phase:
        freq_cos = np.linspace(max_min_freq[0], max_min_freq[1], n_cos)
        phase_cos = np.full(freq_cos.shape, 0.0)
    else:
        freq_steps = np.linspace(max_min_freq[0], max_min_freq[1], n_cos)
        phase_steps = np.linspace(0, np.pi, n_cos)
        freq_cos = np.zeros(n_cos ** 2)
        phase_cos = np.zeros(n_cos **2)
        n_ind = 0
        for fs in freq_steps:
            for ps in phase_steps:
                freq_cos[n_ind] = fs
                phase_cos[n_ind] = ps
                n_ind += 1

    return freq_cos, phase_cos

def gen_randuniform_cosines(max_min_freq, n_cos, fixed_phase=False):
    """ Generates n_cos cosine functions with frequencies uniform random over
    max_min_freq and phases uniform random over [0, pi) if fixed_phase=False."""
    try:
        if len(max_min_freq) > 2:
            raise ValueError("max_min_freq must be 1 or 2 elements specifing the max and min frequency for the cosines.")
    except TypeError:
        # Happens if max_min_freq does not have "len" method, usually because it's a singe number
        max_min_freq = [0.25, max_min_freq]
    # Frequencey must be > 0
    max_min_freq[0] = max(0.25, max_min_freq[0])
    freq_cos = np.random.uniform(max_min_freq[0], max_min_freq[1], n_cos)
    phase_cos = np.random.uniform(0, np.pi, n_cos)

    return freq_cos, phase_cos

def eye_input_to_PC_cosine_relu(eye_data, freq_cos, phase_cos,
                                    n_cos_per_dim=None):
    """ Takes the total 8 dimensional eye data input (x,y position, and
    velocity times 2 lags) and converts it into the n_cosines by 4 + 8 relu
    function input model of PC input. Done point by point for n x 4
    input "eye_data". """
    # Currently hard coded but could change in future
    n_eye_dims = 4
    n_eye_lags = 2
    n_total_eye_dims = n_eye_dims * n_eye_lags
    if n_cos_per_dim is None:
        n_cos_per_dim = int(len(freq_cos) / n_eye_dims)
        if n_cos_per_dim < 1:
            raise ValueError("Not enough cosines input to cover {0} dimensions of eye data.".format(n_eye_dims))
        n_cos_per_dim = np.full((n_eye_dims, ), n_cos_per_dim)
    else:
        if len(n_cos_per_dim) != n_eye_dims:
            raise ValueError("Must specify the number of cosines representing dimensions for each of {0} eye dimensions.".format(n_eye_dims))
        for nd in n_cos_per_dim:
            if nd < 1:
                raise ValueError("Not enough cosines input to cover {0} dimensions of eye data.".format(n_eye_dims))

    if len(freq_cos) != len(phase_cos):
        raise ValueError("Must input the same number of frequencies and phases but got {0} means and {1} slopes.".format(len(freq_cos), len(phase_cos)))
    n_features = len(freq_cos) + 8 # Total input feature to PC is cosines + relus
    first_relu_ind = len(freq_cos)

    # Transform data into "input" n_cosines dimensional format
    # This is effectively like taking our 4 input data features and passing
    # them through n_guassians number of hidden layer units using a
    # cosine activation function and fixed weights plus some relu units
    eye_transform = np.zeros((eye_data.shape[0], n_features))
    dim_start = 0
    dim_stop = 0
    for k in range(0, n_eye_dims):
        dim_stop += n_cos_per_dim[k]
        # First do Cosine activation on first 4 eye dims
        dim_freqs = freq_cos[dim_start:dim_stop]
        dim_phases = phase_cos[dim_start:dim_stop]
        eye_transform[:, dim_start:dim_stop] = cosine_activation(eye_data[:, k],
                                                                    dim_freqs,
                                                                    dim_phases)
        dim_start = dim_stop
        # Then relu activation on second 4 eye dims
        eye_transform[:, (first_relu_ind + 2 * k)] = negative_relu(
                                                            eye_data[:, n_eye_dims + k],
                                                            c=0.0)
        eye_transform[:, (first_relu_ind + (2 * k + 1))] = reflected_negative_relu(
                                                            eye_data[:, n_eye_dims + k],
                                                            c=0.0)
    return eye_transform


# Define sigmoid functions
def sigmoid(x, a=1, b=0, c=1, d=0):
    return c / (1 + np.exp(-(x-b)/a)) + d

def sigmoid_activation(x, fixed_slopes, fixed_centers, fixed_asymptote=1, fixed_bias=0.0):
    num_sigmoids = len(fixed_centers)
    x_transform = np.zeros((x.size, num_sigmoids))
    for k in range(num_sigmoids):
        x_transform[:, k] = sigmoid(x, fixed_slopes[k], fixed_centers[k], fixed_asymptote, fixed_bias)
    return x_transform

def gen_linspace_sigmoids(max_min, n_sigmoids, slope_sigmoids):
    """ Returns means and slopes for the DOUBLE the number of sigmoids input
    in "n_sigmoids" with centers linearly spaced over max_min. Number of output
    sigmoids is doubled such that the first half uses "slope_sigmoids" and
    the second half contains the same means but negative slopes
    "-1*slope_sigmoids." """
    if isinstance(slope_sigmoids, np.ndarray):
        if len(slope_sigmoids) == 1:
            slope_sigmoids = np.full((n_sigmoids, ), slope_sigmoids[0])
        if len(slope_sigmoids) != n_sigmoids:
            raise ValueError("Input slopes must be same size as number of sigmoids or 1")
    elif isinstance(slope_sigmoids, list):
        if len(slope_sigmoids) == 1:
            slope_sigmoids = np.full((n_sigmoids, ), slope_sigmoids[0])
        if len(slope_sigmoids) != n_sigmoids:
            raise ValueError("Input slopes must be same size as number of sigmoids or 1")
    else:
        # We suppose slope_sigmoids is a single numeric value
        slope_sigmoids = np.full((n_sigmoids, ), slope_sigmoids)
    try:
        if len(max_min) > 2:
            raise ValueError("max_min must be 1 or 2 elements specifing the max and min mean for the sigmoids.")
    except TypeError:
        # Happens if max_min does not have "len" method, usually because it's a singe number
        max_min = [-1 * max_min, max_min]
    means_sigmoids = np.linspace(max_min[0], max_min[1], n_sigmoids)
    means_sigmoids = np.hstack((means_sigmoids, means_sigmoids))
    slope_sigmoids = np.hstack((slope_sigmoids, -1*slope_sigmoids))

    return means_sigmoids, slope_sigmoids

def gen_randuniform_sigmoids(mean_max_min, slope_max_min, n_sigmoids,
                                symmetric_output=False):
    """ Returns means and slopes for DOUBLE the number of sigmoids input
    in "n_sigmoids" with centers chosen uniform randomly over mean_max_min and
    slopes selected uniform randomly over slope_max_min. Number of output
    sigmoids is doubled such that the first half uses "slope_sigmoids" and
    the second half contains the same means but negative slopes
    "-1*slope_sigmoids." The 'symmetric_output' flag will output half of the
    sigmoids with exactly the same mean as the other half and with the same
    slope magnitude but of negative value. If this flag is false, half the
    outputs will have positive slope, and half negative, but the means and
    slope magnitudes of all the sigmoids will be chosen randomly. """
    try:
        if len(mean_max_min) > 2:
            raise ValueError("mean_max_min must be 1 or 2 elements specifing the max and min mean for the sigmoid means.")
    except TypeError:
        # Happens if mean_max_min does not have "len" method, usually because it's a singe number
        mean_max_min = np.abs(mean_max_min)
        mean_max_min = [-1 * mean_max_min, mean_max_min]
    try:
        if len(slope_max_min) > 2:
            raise ValueError("slope_max_min must be 1 or 2 elements specifing the max and min slope for the sigmoids.")
    except TypeError:
        # Happens if mean_max_min does not have "len" method, usually because it's a singe number
        slope_max_min = np.abs(slope_max_min)
        slope_max_min = [0.5, slope_max_min]
    # slope must be > 0
    slope_max_min[0] = max(0.1, slope_max_min[0])
    means_sigmoids = np.random.uniform(mean_max_min[0], mean_max_min[1], n_sigmoids)
    slope_sigmoids = np.random.uniform(slope_max_min[0], slope_max_min[1], n_sigmoids)
    if symmetric_output:
        means_sigmoids = np.hstack((means_sigmoids, means_sigmoids))
        slope_sigmoids = np.hstack((slope_sigmoids, -1*slope_sigmoids))
    else:
        means_sigmoids = np.hstack((means_sigmoids, np.random.uniform(mean_max_min[0], mean_max_min[1], n_sigmoids)))
        slope_sigmoids = np.hstack((slope_sigmoids, -1*np.random.uniform(slope_max_min[0], slope_max_min[1], n_sigmoids)))

    return means_sigmoids, slope_sigmoids

def eye_input_to_PC_sigmoid_relu(eye_data, means_sigmoids, slope_sigmoids,
                                    n_sigmoids_per_dim=None):
    """ Takes the total 8 dimensional eye data input (x,y position, and
    velocity times 2 lags) and converts it into the n_sigmoids by 4 + 8 relu
    function input model of PC input. Done point by point for n x 4
    input "eye_data". """
    # Currently hard coded but could change in future
    n_eye_dims = 4
    n_eye_lags = 2
    n_total_eye_dims = n_eye_dims * n_eye_lags
    if n_sigmoids_per_dim is None:
        n_sigmoids_per_dim = int(len(means_sigmoids) / n_eye_dims)
        if n_sigmoids_per_dim < 1:
            raise ValueError("Not enough sigmoid means input to cover {0} dimensions of eye data.".format(n_eye_dims))
        n_sigmoids_per_dim = np.full((n_eye_dims, ), n_sigmoids_per_dim)
    else:
        if len(n_sigmoids_per_dim) != n_eye_dims:
            raise ValueError("Must specify the number of sigmoids representing dimensions for each of {0} eye dimensions.".format(n_eye_dims))
        for nd in n_sigmoids_per_dim:
            if nd < 1:
                raise ValueError("Not enough sigmoid means input to cover {0} dimensions of eye data.".format(n_eye_dims))

    if len(means_sigmoids) != len(slope_sigmoids):
        raise ValueError("Must input the same number of means and slopes but got {0} means and {1} slopes.".format(len(means_sigmoids), len(slope_sigmoids)))
    n_features = len(means_sigmoids) + 8 # Total input feature to PC is sigmoids + relus
    first_relu_ind = len(means_sigmoids)

    # Transform data into "input" n_sigmoids dimensional format
    # This is effectively like taking our 4 input data features and passing
    # them through n_guassians number of hidden layer units using a
    # sigmoidal activation function and fixed weights plus some relu units
    eye_transform = np.zeros((eye_data.shape[0], n_features))
    dim_start = 0
    dim_stop = 0
    for k in range(0, n_eye_dims):
        dim_stop += n_sigmoids_per_dim[k]
        # First do Gaussian activation on first 4 eye dims
        dim_means = means_sigmoids[dim_start:dim_stop]
        dim_slopes = slope_sigmoids[dim_start:dim_stop]
        eye_transform[:, dim_start:dim_stop] = sigmoid_activation(eye_data[:, k],
                                                                    dim_slopes,
                                                                    dim_means)
        dim_start = dim_stop
        # Then relu activation on second 4 eye dims
        eye_transform[:, (first_relu_ind + 2 * k)] = negative_relu(
                                                            eye_data[:, n_eye_dims + k],
                                                            c=0.0)
        eye_transform[:, (first_relu_ind + (2 * k + 1))] = reflected_negative_relu(
                                                            eye_data[:, n_eye_dims + k],
                                                            c=0.0)
    return eye_transform


# Define Gaussian function
def gaussian(x, mu, sigma, scale):
    return scale * np.exp(-( ((x - mu) ** 2) / (2*(sigma**2))) )

# Define the model function as a linear combination of Gaussian functions
def gaussian_activation(x, fixed_means, fixed_sigmas):
    num_gaussians = len(fixed_means)
    x_transform = np.zeros((x.size, num_gaussians))
    for k in range(num_gaussians):
        x_transform[:, k] = gaussian(x, fixed_means[k], fixed_sigmas[k], scale=1.0)
    return x_transform

def gen_linspace_gaussians(max_min, n_gaussians, stds_gaussians):
    """ Returns means and standard deviations for the number of gaussians input
    in "n_gaussians" with centers linearly spaced over max_min. """
    if isinstance(stds_gaussians, np.ndarray):
        if len(stds_gaussians) == 1:
            stds_gaussians = np.full((n_gaussians, ), stds_gaussians[0])
        if len(stds_gaussians) != n_gaussians:
            raise ValueError("Input standard deviations must be same size as means or 1")
    elif isinstance(stds_gaussians, list):
        if len(stds_gaussians) == 1:
            stds_gaussians = np.full((n_gaussians, ), stds_gaussians[0])
        if len(stds_gaussians) != n_gaussians:
            raise ValueError("Input standard deviations must be same size as means or 1")
    else:
        # We suppose gauss stds is a single numeric value
        stds_gaussians = np.full((n_gaussians, ), stds_gaussians)
    try:
        if len(max_min) > 2:
            raise ValueError("max_min must be 1 or 2 elements specifing the max and min mean for the gaussians.")
    except TypeError:
        # Happens if max_min does not have "len" method, usually because it's a singe number
        max_min = [-1 * max_min, max_min]
    means_gaussians = np.linspace(max_min[0], max_min[1], n_gaussians)

    return means_gaussians, stds_gaussians

def gen_randuniform_gaussians(mean_max_min, std_max_min, n_gaussians):
    """ Returns means and standard deviations for the number of gaussians input
    in "n_gaussians" with centers chosen uniform randomly over mean_max_min and
    standard deviations selected uniform randomly over std_max_min. """
    try:
        if len(mean_max_min) > 2:
            raise ValueError("mean_max_min must be 1 or 2 elements specifing the max and min mean for the gaussian means.")
    except TypeError:
        # Happens if mean_max_min does not have "len" method, usually because it's a singe number
        mean_max_min = np.abs(mean_max_min)
        mean_max_min = [-1 * mean_max_min, mean_max_min]
    try:
        if len(std_max_min) > 2:
            raise ValueError("std_max_min must be 1 or 2 elements specifing the max and min mean for the gaussian STDs.")
    except TypeError:
        # Happens if mean_max_min does not have "len" method, usually because it's a singe number
        std_max_min = np.abs(std_max_min)
        std_max_min = [1, std_max_min]
    # STD must be > 0
    std_max_min[0] = max(0.01, std_max_min[0])
    means_gaussians = np.random.uniform(mean_max_min[0], mean_max_min[1], n_gaussians)
    stds_gaussians = np.random.uniform(std_max_min[0], std_max_min[1], n_gaussians)

    return means_gaussians, stds_gaussians

def eye_input_to_PC_gauss_relu(eye_data, gauss_means, gauss_stds,
                                n_gaussians_per_dim, eye_transform=None):
    """ Now modifies the input eye_transform IN PLACE! if not None
    Takes the total 8 dimensional eye data input (x,y position, and
    velocity times 2 lags) and converts it into the n_gaussians by 4 + 8 relu
    function input model of PC input. Done point by point for n x 4
    input "eye_data". n_gaussians_per_dim is a list/array of how many
    gaussians are used to represent each dim so it must either match dims
    or be equal to 1 in which case the same number of gaussians is assumed
    for each dimension. """
    # Currently hard coded but could change in future
    n_eye_dims = 4
    if len(gauss_means) != len(gauss_stds):
        raise ValueError("Must input the same number of means and standard deviations but got {0} means and {1} standard deviations.".format(len(gauss_means), len(gauss_stds)))
    n_features = len(gauss_means) + 8 # Total input featur to PC is gaussians + relus
    if eye_transform is None:
        eye_transform = np.zeros((eye_data.shape[0], n_features))
    if (eye_transform.shape[0] != eye_data.shape[0]) or (eye_transform.shape[1] != n_features):
        raise ValueError("eye_transform is not the correct shape!")
    first_relu_ind = len(gauss_means)

    # Transform data into "input" n_gaussians dimensional format
    # This is effectively like taking our 4 input data features and passing
    # them through n_guassians number of hidden layer units using a
    # Gaussian activation function and fixed weights plus some relu units
    dim_start = 0
    dim_stop = 0
    for k in range(0, n_eye_dims):
        dim_stop += n_gaussians_per_dim[k]
        # First do Gaussian activation on first 4 eye dims
        dim_means = gauss_means[dim_start:dim_stop]
        dim_stds = gauss_stds[dim_start:dim_stop]
        eye_transform[:, dim_start:dim_stop] = gaussian_activation(
                                                                    eye_data[:, k],
                                                                    dim_means,
                                                                    dim_stds)
        dim_start = dim_stop
        # Then relu activation on second 4 eye dims
        eye_transform[:, (first_relu_ind + 2 * k)] = negative_relu(
                                                            eye_data[:, n_eye_dims + k],
                                                            c=0.0)
        eye_transform[:, (first_relu_ind + (2 * k + 1))] = reflected_negative_relu(
                                                            eye_data[:, n_eye_dims + k],
                                                            c=0.0)
    return eye_transform

""" Compute the activation of each unit from the 4D input x. The input
    "proj_gaussians" is a list or tuple full of tuples, each of which defines a
    Gaussian activation function by specifying its unit vector direction, mean
    and STD as follows:
        proj_gaussians[n] = (np.array[xp, yp, xv, yv], gauss_mean, gauss_std)
        where np.array([x, y]) is a 4D unit vector spefifying the angle of
        projection that defines the gaussian over the x and y position/velocity
        and the remaining values specify the mean and standard deviation.
    x must be n observations x 4 eye dimensions for each of the dimensions
    defining the Gaussian activation functions.
"""
def proj_gaussian_activation(x, proj_gaussians):
    num_gaussians = len(proj_gaussians)
    x_transform = np.zeros((x.shape[0], num_gaussians))
    for k in range(num_gaussians):
        # we must first compute the projection of x onto the unit vector
        # defining activation function k
        proj_x = x @ proj_gaussians[k][0]
        x_transform[:, k] = gaussian(proj_x, proj_gaussians[k][1],
                                     proj_gaussians[k][2], scale=1.0)
    return x_transform

def proj_gen_linspace_gaussians(max_min, n_gaussians, n_vectors, stds_gaussians,
                                data_type):
    """ Creates the gaussian activation units for the projection gaussians for
    a total of "n_gaussians" by "n_vectors" activation units. Each of the
    n_vectors will be given n_gaussians to span it with means evenly spaced
    from "max_min" and fixed standard deviations "stds_gaussians.
    This simple implementation keeps position and velocity inputs SEPARATE! and
    so even though each Gaussian has a 4D vector defining it they contain zeros
    in either position or velocity as specified. The two cardinal axes are
    always included to ensure we can span the space by n_vectors must be even
    number to ensure 0 and pi/2 angles are covered. """
    if isinstance(stds_gaussians, np.ndarray):
        if len(stds_gaussians) == 1:
            stds_gaussians = np.full((n_gaussians, ), stds_gaussians[0])
        if len(stds_gaussians) != n_gaussians:
            raise ValueError("Input standard deviations must be same size as means or 1")
    elif isinstance(stds_gaussians, list):
        if len(stds_gaussians) == 1:
            stds_gaussians = np.full((n_gaussians, ), stds_gaussians[0])
        if len(stds_gaussians) != n_gaussians:
            raise ValueError("Input standard deviations must be same size as means or 1")
    else:
        # We suppose gauss stds is a single numeric value
        stds_gaussians = np.full((n_gaussians, ), stds_gaussians)
    try:
        if len(max_min) > 2:
            raise ValueError("max_min must be 1 or 2 elements specifing the max and min mean for the gaussians.")
    except TypeError:
        # Happens if max_min does not have "len" method, usually because it's a singe number
        max_min = [-1 * max_min, max_min]
    if "pos" in data_type.lower():
        v_dims = slice(0, 2)
    elif "vel" in data_type.lower():
        v_dims = slice(2, 4)
    else:
        raise ValueError("Unrecognized data type. Must be position or velocity.")
    if n_vectors < 2:
        raise ValueError("Need at least 2 vectors to span 2D eye data")
    if n_vectors % 2 == 1:
        # Odd number won't include pi/2 so add it
        n_vectors += 1
    means_gaussians = np.linspace(max_min[0], max_min[1], n_gaussians)
    angles = np.linspace(0, np.pi, n_vectors, endpoint=False)
    proj_gaussians = []
    for ang in angles:
        # Create the vector that defines all units on this vector
        vector = np.zeros(4)
        vector[v_dims] = np.array([np.cos(ang), np.sin(ang)])
        # Remove any numerical error from sin and cos
        vector[np.isclose(vector, 0, atol=1e-10)] = 0
        vector[np.isclose(vector, 1, atol=1e-10)] = 1
        for gauss in range(0, n_gaussians):
            proj_gaussians.append( (vector, means_gaussians[gauss], stds_gaussians[gauss]) )

    return proj_gaussians

def proj_gen_randuniform_gaussians(mean_max_min, std_max_min, n_gaussians, data_type):
    """ Returns means and standard deviations and preferred vectors that define a Gaussian input
    unit where all values are selected random uniformly from the ranges input. Unlike the 
    linspace version above, the random Gaussinas returns a total of n_gaussians gaussian units.
    All of them have random mean, std, and vectors instead of spanning a given vector evenly."""
    try:
        if len(mean_max_min) > 2:
            raise ValueError("mean_max_min must be 1 or 2 elements specifing the max and min mean for the gaussians.")
    except TypeError:
        # Happens if mean_max_min does not have "len" method, usually because it's a singe number
        mean_max_min = [-1 * mean_max_min, mean_max_min]
    try:
        if len(std_max_min) > 2:
            raise ValueError("std_max_min must be 1 or 2 elements specifing the max and min STD for the gaussians.")
    except TypeError:
        # Happens if std_max_min does not have "len" method, usually because it's a singe number
        std_max_min = [0.01, std_max_min]
    # STD can't be less than zero
    std_max_min[0] = max(std_max_min[0], 0.01)

    if "pos" in data_type.lower():
        v_dims = slice(0, 2)
    elif "vel" in data_type.lower():
        v_dims = slice(2, 4)
    else:
        raise ValueError("Unrecognized data type. Must be position or velocity.")

    means_gaussians = np.random.uniform(mean_max_min[0], mean_max_min[1], n_gaussians)
    stds_gaussians = np.random.uniform(std_max_min[0], std_max_min[1], n_gaussians)
    angles = np.random.uniform(0, np.pi, n_gaussians)
    proj_gaussians = []
    for ang_i, ang in enumerate(angles):
        # Create the vector that defines all units on this vector
        vector = np.zeros(4)
        vector[v_dims] = np.array([np.cos(ang), np.sin(ang)])
        # Remove any numerical error from sin and cos
        vector[np.isclose(vector, 0, atol=1e-10)] = 0
        vector[np.isclose(vector, 1, atol=1e-10)] = 1
        proj_gaussians.append( (vector, means_gaussians[ang_i], stds_gaussians[ang_i]) )

    return proj_gaussians

def proj_eye_input_to_PC_gauss_relu(eye_data, proj_gaussians,
                                    eye_transform=None):
    """ Modifies the input eye_transform IN PLACE! if not None
    Takes the total 8 dimensional eye data input (x,y position, and
    velocity times 2 lags) and converts it into the n_gaussians by 8 relu
    function input model of PC input. The input "proj_gaussians" is a list of
    tuple where each tuple defines the a single gaussian of the basis set
    as follows:
        proj_gaussians[n] = (np.array[xp, yp, xv, yv], gauss_mean, gauss_std)
        where np.array([x, y]) is a 4D unit vector spefifying the angle of
        projection that defines the gaussian over the x and y position/velocity
        and the remaining values specify the mean and standard deviation.
    """
    # Currently hard coded but could change in future
    n_eye_dims = 4
    n_features = len(proj_gaussians) + 8 # Total input featur to PC is gaussians + relus
    if eye_transform is None:
        eye_transform = np.zeros((eye_data.shape[0], n_features))
    if (eye_transform.shape[0] != eye_data.shape[0]) or (eye_transform.shape[1] != n_features):
        raise ValueError("eye_transform is not the correct shape!")
    first_relu_ind = len(proj_gaussians)

    # Transform data into "input" n_gaussians dimensional format
    # This is effectively like taking our input data features and passing
    # them through n_guassians number of hidden layer units using a
    # Gaussian activation function plus some relu units
    eye_transform[:, 0:first_relu_ind] = proj_gaussian_activation(
                                                    eye_data[:, 0:n_eye_dims],
                                                    proj_gaussians)
    # Then relu activation on second 4 eye dims
    for k in range(0, n_eye_dims):
        eye_transform[:, (first_relu_ind + 2 * k)] = negative_relu(
                                                            eye_data[:, n_eye_dims + k],
                                                            c=0.0)
        eye_transform[:, (first_relu_ind + (2 * k + 1))] = reflected_negative_relu(
                                                            eye_data[:, n_eye_dims + k],
                                                            c=0.0)
    return eye_transform
