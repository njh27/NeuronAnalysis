import numpy as np



class Neuron(object):
    """ Class that defines a neuron...
    """

    def __init__(self, neuron_dict, session, name, cell_type=None):
        """ Neuron dict is a dictionary of neuron information as contained in each
            list element output by neuroviz. Creates an object that can store
            cell type class, tuning properties, and valid trials for a neuron. """
        # Properties that should minimially define a neuron
        self.spike_indices = neuron_dict['spike_indices__'].sorted()
        self.channel = neuron_dict['channel_id__']
        self.sampling_rate = neuron_dict['sampling_rate__']
        self.cell_type = cell_type

        # Properties for linking with session behavior and trial data
        self.session = session
        self.name = name # The name of this neuron and its dataseries in the session
        self.valid_trials = np.zeros(len(session), dtype='bool')
        self.check_stability()

    def check_stability(self):
        stable_wins = find_stable_ranges(self.ms_spikes(), win_size, duration, tol_percent=25)

    def fit_FR_model(self, blocks, trials, dataseries):
        pass


    def compute_tuning_by_condition(self, time_window, target_number, target_data_type='position',
                                    block_name=None, trial_index=None, n_block_occurence=0,
                                    subtract_fixation_win=None):

        trial_trains, trial_inds = self.condition_rate_by_trial(None, time_window,
                            block_name=block_name, n_block_occurence=n_block_occurence)

        if subtract_fixation_win is not None:
            fixation_trains, _ = self.condition_rate_by_trial(None, subtract_fixation_win,
                            block_name=block_name, n_block_occurence=n_block_occurence)

        target_data, trial_inds_t = self.session.target_condition_data_by_trial(None, time_window, target_data_type,
                target_number, block_name=block_name, n_block_occurence=n_block_occurence)

        if trial_index is not None:
            # Remove any trials not in trial_index from consideration and their
            # corresponding target_data and trial_trains
            data_mask = np.ones(target_data.shape[0], dtype='bool')
            data_index = 0
            for i in range(0, trial_inds_t.size):
                if trial_inds_t[i]:
                    if not trial_index[i]:
                        # Input trial index says skip this trial
                        data_mask[data_index] = False
                        trial_inds_t[i] = False
                    data_index += 1
            trial_trains = trial_trains[data_mask, :]
            target_data = target_data[data_mask, :]
            if subtract_fixation_win is not None: fixation_trains = fixation_trains[data_mask, :]

        tune_spikes = {}
        tune_trials = {}
        # for t_key in self.session.blocks[block_name]['trial_names']:
        #     tune_spikes[t_key] = []
        #     tune_trials[t_key] = []
        row_ind = 0
        for t in range(self.session.blocks[block_name]['trial_windows'][0][0], self.session.blocks[block_name]['trial_windows'][0][1]):
            if trial_inds_t[t]:
                if self.session.trials[t]['trial_name'] not in tune_spikes:
                    tune_spikes[self.session.trials[t]['trial_name']] = []
                    tune_trials[self.session.trials[t]['trial_name']] = []
                if subtract_fixation_win is None:
                    tune_spikes[self.session.trials[t]['trial_name']].append(np.nanmean(trial_trains[row_ind, :]))
                else:
                    tune_spikes[self.session.trials[t]['trial_name']].append(
                                np.nanmean(trial_trains[row_ind, :]) - np.nanmean(fixation_trains[row_ind, :]))
                tune_trials[self.session.trials[t]['trial_name']].append(np.nanmean(target_data[row_ind, :, :], axis=0))
                row_ind += 1
        theta = np.zeros(len(tune_trials))
        rho = np.zeros(len(tune_trials))
        for ind, key in enumerate(tune_trials):
            rho[ind] = np.nanmean(tune_spikes[key])
            tune_trials[key] = np.nanmean(np.vstack(tune_trials[key]), axis=0)
            theta[ind] = np.arctan2(tune_trials[key][1], tune_trials[key][0])

        theta_order = np.argsort(theta)
        theta = theta[theta_order]
        rho = rho[theta_order]

        return theta, rho

    def set_optimal_pursuit_vector(self, time_window, target_number, target_data_type='position',
                                   block_name=None, n_block_occurence=0, subtract_fixation_win=None):

        theta, rho = self.compute_tuning_by_condition(time_window, target_number,
                        target_data_type=target_data_type, block_name=block_name,
                        n_block_occurence=n_block_occurence, subtract_fixation_win=subtract_fixation_win)

        amp, phase, offset = fit_cos_fixed_freq(theta, rho)
        self.cos_fit_fun = lambda x: (amp * (np.cos(x + phase)) + offset)
        self.optimal_pursuit_vector = -1 * phase

    def compute_ACG(self, time_window, lag_window, dt, trial_names, block_name, n_block_occurence=0):
        raise RuntimeError("this is still old code not updated")
        if self.nan_saccades:
            print("NANING SACCADES NOT IMPLEMENTED IN CCG CALCULATION YET !!!")

        if time_window is None:
            # Do ACG over all spikes from all time points (ignores trials)
            counts, time_axis = cross_correlogram(self.spike_times , self.spike_times , lag_window, dt)
            return counts, time_axis

        if trial_names is None:
            trial_names = self.session.get_trial_names()
        trial_index = self.session.get_trial_index(trial_names, block_name, n_block_occurence)
        spikes_1 = []
        n_total_spikes = 0
        for t in range(0, len(self.trial_spikes)):
            if not trial_index[t]:
                continue
            t_spk_ind = np.logical_and(self.trial_spikes[t]['spike_times'] >= time_window[0],
                                       self.trial_spikes[t]['spike_times'] < time_window[1])
            if self.nan_saccades:
                pass
            spikes_1.append(self.trial_spikes[t]['spike_times'][t_spk_ind])
            n_total_spikes += spikes_1[-1].size

        counts, time_axis = trial_wise_cross_correlogram(spikes_1, spikes_1, time_window, lag_window=lag_window, dt=dt)
        counts = counts / n_total_spikes

        return counts, time_axis

    def compute_CCG(self, Neuron):
        pass

    def get_ISIs(self):
        pass

    def compute_CV2(self):
        ISIs = np.diff(self.spike_indices)
        diff_ISI = np.abs(np.diff(ISIs))
        sum_ISI = ISIs[0:-1] + ISIs[1:]
        sum_ISI = sum_ISI[diff_ISI != 0]
        diff_ISI = diff_ISI[diff_ISI != 0]
        CV2s = 2 * diff_ISI / sum_ISI
        return CV2s

    def ms_spikes(self):
        """ Convert spike times in units of indices to units of milliseconds. """
        ms_spikes = self.spike_indices / (Neuron.sampling_rate / 1000)
        return ms_spikes

def find_stable_ranges(spikes_ms, win_size, duration, tol_percent=25):
    """ Duration is in units of 'win_size', i.e. for a duration of N win_sizes. """
    if tol_percent > 1:
        tol_percent = tol_percent / 100
    stable_wins = []

    return stable_wins
