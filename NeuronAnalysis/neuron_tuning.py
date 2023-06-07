import numpy as np
from NeuronAnalysis.fit_neuron_to_eye import bin_data, quick_fit_piecewise_acc, piece_wise_eye_data
from SessionAnalysis.utils import eye_data_series



def get_fr_eye_data(neuron, blocks, trial_sets, bin_width, bin_threshold, time_window, 
                    lag=0, acc_filter_win=31, return_inds=False, fr_offsets_by_trial=None):
    """ Helper function that returns the firing rate and eye data as requested in the format
    needed to fit in linear regression with acceleration terms and binning.
    """
    # Get firing rate in standard window
    firing_rate, fr_inds = neuron.get_firing_traces(time_window, blocks,
                                                    trial_sets, return_inds=True)
    if fr_offsets_by_trial is not None:
        assert len(fr_offsets_by_trial) == len(fr_inds)
        firing_rate -= fr_offsets_by_trial[:, None]
    firing_rate = bin_data(firing_rate, bin_width, bin_threshold)
    firing_rate = firing_rate.reshape(firing_rate.shape[0]*firing_rate.shape[1], order="C")
    # Get eye data in LAG window
    lagged_t_window = [time_window[0] + lag, time_window[1] + lag]
    pos_p, pos_l = neuron.session.get_xy_traces("eye position",
                            lagged_t_window, blocks, fr_inds,
                            return_inds=False)
    vel_p, vel_l = neuron.session.get_xy_traces("eye velocity",
                            lagged_t_window, blocks, fr_inds,
                            return_inds=False)
    # Stack with exra vel slots for acceleration
    eye_data = np.stack((pos_p, pos_l, vel_p, vel_l, vel_p, vel_l), axis=2)
    eye_data[:, :, 4:6] = eye_data_series.acc_from_vel(eye_data[:, :, 2:4],
                                                        filter_win=acc_filter_win)
    # Now bin and reshape to 2D
    eye_data = bin_data(eye_data, bin_width, bin_threshold)
    eye_data = eye_data.reshape(eye_data.shape[0]*eye_data.shape[1], eye_data.shape[2], order="C")
    if return_inds:
        return firing_rate, eye_data, fr_inds
    return firing_rate, eye_data


def nan_lstsq(x, y):
    """ Quick linear regression that removes nans from x and y.
    """
    if x.ndim == 1:
        x = np.expand_dims(x, axis=1)
    select_good = ~np.isnan(y)
    select_good = select_good & ~np.any(np.isnan(x), axis=1)
    x_nonan = x[select_good, :]
    y_nonan = y[select_good] # This should generally return a copy
    coefficients = np.linalg.lstsq(x_nonan, y_nonan, rcond=None)[0]
    y_mean = np.mean(y_nonan)
    y_predicted = np.matmul(x_nonan, coefficients)
    sum_squares_error = np.nansum((y_nonan - y_predicted) ** 2)
    sum_squares_total = np.nansum((y_nonan - y_mean) ** 2)
    R2 = 1 - sum_squares_error/(sum_squares_total)

    return coefficients, R2


def comp_block_scaling_factors(primary_blocks, adj_blocks, neuron, time_window=[-100, 900], 
                               fix_time_window=[-300, 0], lag_range_eye=[-50, 150], 
                               trial_sets=None, bin_width=10, bin_threshold=5, quick_lag_step=10):
    """ Takes a given neuron and performs a linear fit on the primary block.
    Then computes the optimum scaling factor between the primary block
    fit and a linear fit on each block input in "scaled_blocks" such that
    the responses of the scaled blocks are DC shifted and multiplicatively
    scaled to "match" the primary block in a linear least squares sense.
    """
    # Make sure input blocks are lists and primary blocks are not in blocks to be scaled
    if not isinstance(primary_blocks, list):
        primary_blocks = [primary_blocks]
    if not isinstance(adj_blocks, list):
        raise ValueError("adj_blocks must be a list of block names")
    # Make a new list instead of changing the input, ensuring it does not contain primary block
    scaled_blocks = []
    for block in adj_blocks:
        add_block = True
        for pblock in primary_blocks:
            if pblock == block:
                add_block = False
                break
        if add_block:
            scaled_blocks.append(block)    
    quick_lag_step = int(round(quick_lag_step))
    if quick_lag_step < 1:
        raise ValueError("quick_lag_step must be positive integer")
    if quick_lag_step > (lag_range_eye[1] - lag_range_eye[0]):
        raise ValueError("quick_lag_step is too large relative to lag_range_eye")
    half_lag_step = int(round(quick_lag_step / 2))
    lags = np.arange(lag_range_eye[0], lag_range_eye[1] + half_lag_step + 1, quick_lag_step)
    lags[-1] = lag_range_eye[1]

    # Get fixation data for all trials and smoothed value for adjusting Offset drift
    all_blocks = primary_blocks + scaled_blocks
    fr_fix, fr_inds_all = neuron.get_fix_by_block_gauss(all_blocks, fix_time_window, sigma=12.5, 
                                                        cutoff_sigma=4, zscore_sigma=3.0)

    # Get the fixation offset for the primary trials
    fr_inds_prim = neuron.session._parse_blocks_trial_sets(primary_blocks, trial_sets)
    _, _, fix_inds = np.intersect1d(fr_inds_prim, fr_inds_all, return_indices=True)
    prim_offsets = fr_fix[fix_inds]
    # Single value fixation average for primary block is used as constant for all
    bias_constant = np.nanmean(prim_offsets)

    R2 = []
    coefficients = []
    # First loop over lags using quick_lag_step intervals
    for lag in lags:
        firing_rate, eye_data = get_fr_eye_data(neuron, primary_blocks, trial_sets, bin_width, 
                                                bin_threshold, time_window, lag=lag,
                                                fr_offsets_by_trial=prim_offsets)
        coeffs, r2 = quick_fit_piecewise_acc(firing_rate, eye_data, fit_constant=True)
        coefficients.append(coeffs)
        R2.append(r2)
    if quick_lag_step > 1:
        # Do fine resolution loop
        max_ind = np.where(R2 == np.amax(R2))[0][0]
        # max_ind = max_ind[np.argmin(np.abs(lags[max_ind]))]
        best_lag = lags[max_ind]
        # Make new lags centered on this best_lag
        lag_start = max(lags[0], best_lag - quick_lag_step)
        lag_stop = min(lags[-1], best_lag + quick_lag_step)
        lags = np.arange(lag_start, lag_stop + 1, 1)
        # Reset fit measures
        R2 = []
        coefficients = []
        for lag in lags:
            firing_rate, eye_data = get_fr_eye_data(neuron, primary_blocks, trial_sets, bin_width, 
                                                    bin_threshold, time_window, lag=lag,
                                                    fr_offsets_by_trial=prim_offsets)
            coeffs, r2 = quick_fit_piecewise_acc(firing_rate, eye_data, fit_constant=True)
            coefficients.append(coeffs)
            R2.append(r2)
    # Choose peak R2 value with minimum absolute value lag and get best params
    max_ind = np.where(R2 == np.amax(R2))[0][0]
    # max_ind = max_ind[np.argmin(np.abs(lags[max_ind]))]
    coefficients = coefficients[max_ind]
    eye_lag = lags[max_ind]
    R2 = R2[max_ind]

    # Now that we have a fit for the primary block, get scaling factors for other blocks
    block_scaling_factors = {}
    for block_name in primary_blocks:
        block_scaling_factors[block_name] = (np.array([[1.0]]), 1.0, bias_constant)
    for block_name in scaled_blocks:
        # Get predicted value for these trials given the primary fit (not using constant term here)
        cur_t_inds = neuron.session._parse_blocks_trial_sets(block_name, trial_sets)
        _, _, fix_inds = np.intersect1d(cur_t_inds, fr_inds_all, return_indices=True)
        curr_offsets = fr_fix[fix_inds]
        firing_rate, eye_data = get_fr_eye_data(neuron, [block_name], trial_sets, bin_width, 
                                                bin_threshold, time_window, lag=eye_lag, return_inds=False,
                                                fr_offsets_by_trial=curr_offsets)
        eye_data_piece = piece_wise_eye_data(eye_data, add_constant=False)
        fr_y_hat = np.dot(eye_data_piece, coefficients[0:-1])
        coeffs, r2 = nan_lstsq(fr_y_hat, firing_rate)
        block_scaling_factors[block_name] = (coeffs, r2, bias_constant)

    return block_scaling_factors, fr_fix, fr_inds_all