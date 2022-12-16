import numpy as np



class Neuron(object):
    """ Class that defines a neuron...
    """

    def __init__(self, neuron_dict, session, name, class=None):
        """ Neuron dict is a dictionary of neuron information as contained in each
            list element output by neuroviz. Creates an object that can store
            cell type class, tuning properties, and valid trials for a neuron. """
        # Properties that should minimially define a neuron
        self.spike_times = ms_spikes(neuron_dict)
        self.spike_waves = neuron_dict['waveforms']
        self.avg_waveform = neuron_dict['template']

        # Properties that define a neuron's origin and sorting
        self.reader = pl2_reader
        self.recorded_chan = neuron_dict['channel']
        self.neighbor_chans = neuron_dict['neighbors']
        self.sort_quality = neuron_dict['sort_quality']

        # Properties for linking with session behavior and trial data
        self.trial_spikes = None
        self.session = Session
        if Session is not None:
            self.join_session(Session)
        self.nan_saccades = nan_saccades
        self.nan_saccade_lag = nan_saccade_lag
        self.default_spike_train = default_spike_train

    def join_session(self, Session):
        """. """
        self.trial_spikes = ConjoinedList([{} for x in range(0, len(Session.trials))])
        for trial in range(0, len(Session.trials)):
            # Since XS2 start pulse was screwey, use XS2 END pulse for beginning and end, find beginning by subtracting duration
            trial_index = np.all([self.spike_times >= (Session.trials[trial]['plexon_start_stop'][1] - Session.trials[trial]['duration_ms']),
                                  self.spike_times <= Session.trials[trial]['plexon_start_stop'][1]], axis=0)
            self.trial_spikes[trial]['spike_times'] = self.spike_times[trial_index]
            # Align spikes to trial time zero
            self.trial_spikes[trial]['spike_times'] -= (Session.trials[trial]['plexon_start_stop'][1] - Session.trials[trial]['duration_ms'])
            self.trial_spikes[trial]['spike_alignment'] = 0
        self.session = Session
        Session.neurons.append(self)
        self.trial_spikes.conjoin_list(Session.trials)
        self.align_spikes_to_session()

    def align_spikes_to_session(self):
        """. """
        for t in range(0, len(self.trial_spikes)):
            # Un-do old time series alignment, do new alignment and save it's time
            self.trial_spikes[t]['spike_times'] += self.trial_spikes[t]['spike_alignment']
            self.trial_spikes[t]['spike_times'] -= self.session.trials[t]['time_series_alignment']
            self.trial_spikes[t]['spike_alignment'] = self.session.trials[t]['time_series_alignment']

    def ISI_FR(self, low_filt=25):
        """ Add per trial firing rate using the inverse inter spike interval method. """
        if low_filt is not None:
            b_filt, a_filt = signal.butter(2, low_filt/500)
        for t in range(0, len(self.trial_spikes)):
            self.trial_spikes[t]['ISI_FR'] = inverse_ISI_FR(self.trial_spikes[t]['spike_times'],
                [self.session.trials[t]['time_series'][0], self.session.trials[t]['time_series'][-1]], time_step=1)
            if low_filt is not None:
                self.trial_spikes[t]['ISI_FR'] = signal.filtfilt(b_filt, a_filt, self.trial_spikes[t]['ISI_FR'], axis=0,
                    padlen=np.amin((self.session.trials[t]['duration_ms']/2, 100)).astype('int'))

    def bin_FR(self, binwidth=1):
        """ Add per trial firing rate using binned spike count method. """
        if binwidth != 1:
            # This probably needs some modification to accomodate binwidths that != 1.
            print("WARNING THIS HAS NOT BEEN MADE TO WORK WITH BINWIDTH != 1!")
        half_bin = binwidth / 2
        for t in range(0, len(self.trial_spikes)):
            hist_bins = np.arange(self.session.trials[t]['time_series'][0] - half_bin,
                                  self.session.trials[t]['time_series'][-1] + half_bin, binwidth)
            self.trial_spikes[t]['bin_FR'] = 1000 * np.histogram(self.trial_spikes[t]['spike_times'], bins=hist_bins)[0].astype('int')

    def bin_gauss_convolved_FR(self, sigma, cutoff_sigma=4):
        """. """
        if 'bin_FR' not in self.trial_spikes[0]:
            # THIS WILL REQUIRE MORE ADJUSTMENT IF BINWIDTH != 1
            self.bin_FR(1)

        x_win = int(np.around(sigma * cutoff_sigma))
        xvals = np.arange(-1 * x_win, x_win + 1)
        kernel = np.exp(-.5 * (xvals / sigma) ** 2)
        kernel = kernel / np.sum(kernel)
        for t in range(0, len(self.trial_spikes)):
            self.trial_spikes[t]['GC_FR'] = np.convolve(self.trial_spikes[t]['bin_FR'], kernel, mode='same')

    def postsynaptic_decay_FR(self, tau_rise=1., tau_decay=2.5):
        """ Add per trial firing rate using the postsynaptic exponential rise
            and decay method. This is computed on the basis of binned firing rate,
            so bin_FR will be called if it hasn't already. """
        if 'bin_FR' not in self.trial_spikes[0]:
            # THIS WILL REQUIRE MORE ADJUSTMENT IF BINWIDTH != 1
            self.bin_FR(1)

        xvals = np.arange(0, np.ceil(5 * max(tau_rise, tau_decay)) + 1)
        kernel = np.exp(- 1 * xvals / tau_decay) - np.exp(- 1 * xvals / tau_rise)
        kernel = zero_phase_kernel(kernel, 0) # Shift kernel to be causal at t = 0
        kernel = kernel / np.sum(kernel)
        for t in range(0, len(self.trial_spikes)):
            self.trial_spikes[t]['PSP_FR'] = np.convolve(self.trial_spikes[t]['bin_FR'], kernel, mode='same')

    def create_default_train(self, **kwargs):
        if self.default_spike_train == 'ISI_FR':
            self.ISI_FR(**kwargs)
        elif self.default_spike_train == 'bin_FR':
            self.bin_FR(**kwargs)
        elif self.default_spike_train == 'GC_FR':
            self.bin_gauss_convolved_FR(**kwargs)
        elif self.default_spike_train == 'PSP_FR':
            self.postsynaptic_decay_FR(**kwargs)

    def condition_rate_by_trial(self, trial_names, time_window, block_name=None, n_block_occurence=0):
        """. """
        trial_index = self.session.get_trial_index(trial_names, block_name, n_block_occurence)
        trial_trains = np.full((np.count_nonzero(trial_index), time_window[1]-time_window[0]), np.nan)
        out_row = 0
        for t in range(0, len(self.trial_spikes)):
            if not trial_index[t]:
                continue
            time_index = self.session.trials[t]['time_series'].find_index_range(time_window[0], time_window[1])
            if time_index is None:
                # Entire trial window is beyond available data
                trial_index[t] = False
                continue
            if round(self.session.trials[t]['time_series'].start) - time_window[0] > 0:
                # Window start is beyond available data
                out_start = round(self.session.trials[t]['time_series'].start) - time_window[0]
            else:
                out_start = 0
            if round(self.session.trials[t]['time_series'].stop) - time_window[1] < 0:
                # Window stop is beyond available data
                out_stop = trial_trains.shape[1] - (time_window[1] - round(self.session.trials[t]['time_series'].stop))
            else:
                out_stop = trial_trains.shape[1]
            trial_trains[out_row, out_start:out_stop] = self.trial_spikes[t][self.default_spike_train][time_index]
            if self.nan_saccades:
                for start, stop in self.session.trials[t]['saccade_windows']:
                    # Align start/stop with desired offset
                    start += self.nan_saccade_lag
                    stop += self.nan_saccade_lag
                    # Align start/stop with time index and window
                    if stop > time_index[0] and stop < time_index[-1]:
                        start = max(start, time_index[0])
                    elif start > time_index[0] and start < time_index[-1]:
                        stop = min(stop, time_index[-1])
                    else:
                        continue
                    # Align start/stop with trial_trains
                    start += out_start - time_index[0]
                    stop += out_start - time_index[0]
                    trial_trains[out_row, start:stop] = np.nan
            out_row += 1
        trial_trains = trial_trains[0:out_row, :]
        return trial_trains, trial_index

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
        ISIs = np.diff(self.spike_times)
        diff_ISI = np.abs(np.diff(ISIs))
        sum_ISI = ISIs[0:-1] + ISIs[1:]
        sum_ISI = sum_ISI[diff_ISI != 0]
        diff_ISI = diff_ISI[diff_ISI != 0]
        CV2s = 2 * diff_ISI / sum_ISI
        return CV2s
