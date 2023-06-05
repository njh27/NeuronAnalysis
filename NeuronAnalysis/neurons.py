import numpy as np
import warnings
from NeuronAnalysis.general import fit_cos_fixed_freq
from NeuronAnalysis.fit_neuron_to_eye import bin_data, quick_fit_piecewise_acc, piece_wise_eye_data
from SessionAnalysis.utils import eye_data_series




# Define our default cell type naming scheme for cerebellum recording
def get_cb_name(neuron_dict, default_name="N"):
    """ Defines a scheme for converting cell type labels from neuroviz sorting
    into names of cell types for data storage/analysis for cerebellum
    recordings. """
    try:
        if neuron_dict['type__'] == 'NeurophysToolbox.ComplexSpikes':
            use_name = "CS"
            print("Found a CS without any SS for unit.")
        elif neuron_dict['type__'] == 'NeurophysToolbox.PurkinjeCell':
            use_name = "PC"
            if neuron_dict['cs_spike_indices__'].size == 0:
                raise ValueError("Input unit is a confirmed PC but does not have a CS match in its Neuron object!")
        else:
            use_name = neuron_dict.get('label')
        if use_name is None:
            # Skip to below
            raise KeyError()
        elif use_name.lower() in ["unlabeled", "unknown"]:
            raise KeyError()
        else:
            if use_name in ["putative_pc"]:
                use_name = "putPC"
            elif use_name in ["putative_cs", "CS"]:
                use_name = "CS"
            elif use_name in ["putative_basket", "MLI"]:
                use_name = "MLI"
            elif use_name in ["putative_mf", "MF"]:
                use_name = "MF"
            elif use_name in ["putative_golgi", "GC"]:
                use_name = "GC"
            elif use_name in ["putative_ubc", "UBC"]:
                use_name = "UBC"
            elif use_name in ["putative_stellate", "SC"]:
                use_name = "SC"
            else:
                if use_name != "PC":
                    raise ValueError("Unrecognized neuron label {0}.".format(use_name))
    except KeyError:
        # Neuron does not have a class field so use default
        use_name = default_name
    return use_name


class Neuron(object):
    """ Class that defines a neuron...
    """

    def __init__(self, neuron_dict, name, cell_type="unknown", session=None):
        """ Neuron dict is a dictionary of neuron information as contained in each
            list element output by neuroviz. Creates an object that can store
            cell type class, tuning properties, and valid trials for a neuron. """
        # Properties that should minimially define a neuron
        self.spike_indices = np.sort(neuron_dict['spike_indices__'])
        self.channel = neuron_dict['channel_id__']
        self.sampling_rate = neuron_dict['sampling_rate__']
        if cell_type.lower() == "cb_name":
            self.cell_type = get_cb_name(neuron_dict)
        else:
            self.cell_type = cell_type
        self.layer = None
        self.min_trials_per_condition = 5
        self.use_series = name
        self.optimal_cos_funs = {}
        self.optimal_cos_vectors = {}
        self.optimal_cos_time_window = {}

        # Properties for linking with session behavior and trial data
        self.name = name # The name of this neuron and its dataseries in the session
        self.check_stability()
        if session is not None:
            self.join_session(session)

    def check_stability(self, win_size=30000, duration=2, tol_percent=20):
        min_rate = 0.05
        # Smooth all data with sigma of 1 bin
        seg_bins, bin_rates, bin_t_wins, raw_bin_rates = find_stable_ranges(self.get_spikes_ms(),
                                            win_size, duration, tol_percent,
                                            min_rate, sigma_smooth=2)
        self.seg_bins = seg_bins
        self.bin_t_wins = bin_t_wins
        self.smooth_bin_rates = bin_rates
        self.raw_bin_rates = raw_bin_rates
        # No look for stable time windows smoothed with bigger sigma and more
        # tolerance for connecting over segments
        self.stable_time_wins_ms = get_stable_time_wins(seg_bins, bin_rates,
                                    bin_t_wins, tol_percent=20, sigma_smooth=2)
        self.stable_time_wins_ind = []
        for stw in self.stable_time_wins_ms:
            ind_win = [stw[0] * (self.sampling_rate / 1000), stw[1] * (self.sampling_rate / 1000)]
            ind_win[0] = int(np.around(ind_win[0]))
            ind_win[1] = int(np.around(ind_win[1]))
            self.stable_time_wins_ind.append(ind_win)

    def recompute_fits(self):
        """ Calls all the fitting/tuning based functions for this unit for
        convenience either as neurons are made or as valid trials etc. are
        modified. """
        # If we already did this for multiple blocks, redo them
        if len(self.optimal_cos_vectors) > 0:
            for block in self.optimal_cos_vectors.keys():
                self.set_optimal_pursuit_vector(self.optimal_cos_time_window[block],
                                                block)

    def join_session(self, session):
        self.session = session
        self.compute_valid_trials()
        self.recompute_fits()

    def compute_valid_trials(self):
        """ Computes the valid trials based on the stable time windows and adds
        them to self.sess as trial sets. """
        self.session.trial_sets[self.name] = np.zeros(len(self.session), dtype='bool')
        trial_ind = 0
        stw_ind = 0
        while stw_ind < len(self.stable_time_wins_ind):
            curr_win = self.stable_time_wins_ind[stw_ind]
            while trial_ind < len(self.session):
                trial_start = self.session._trial_lists['neurons'][trial_ind]['meta_data']['pl2_start_stop'][0]
                trial_stop = self.session._trial_lists['neurons'][trial_ind]['meta_data']['pl2_start_stop'][1]
                if trial_stop <= curr_win[0]:
                    # Trial completely precedes current valid window
                    trial_ind += 1
                elif ( (trial_start >= curr_win[0]) and (trial_stop <= curr_win[1]) ):
                    # Trial is fully within current valid window
                    self.session.trial_sets[self.name][trial_ind] = True
                    trial_ind += 1

                elif ( (trial_start < curr_win[1]) and (trial_stop >= curr_win[1]) ):
                    # Trial extends beyond current valid window
                    stw_ind += 1
                    break
                elif ( (trial_start < curr_win[0]) and (trial_stop >= curr_win[0]) ):
                    # Trial starts just before and overlaps current valid window
                    trial_ind += 1
                elif (trial_start >= curr_win[1]):
                    # Trial starts after the current valid window
                    stw_ind += 1
                    break
                else:
                    print("Condition not found for:", trial_start, trial_stop, curr_win)
            if trial_ind >= len(self.session):
                break

    def append_valid_trial_set(self, trial_sets):
        """ Takes the input trial_sets and appends to it the valid trial set
        for the current neuron for selecting data from trials specific to when
        this neuron was valid. The trial_sets input itself may be modified and
        is returned. """
        if trial_sets is None:
            trial_sets = [self.name]
        elif isinstance(trial_sets, str):
            if trial_sets == self.name:
                # trial_sets is already this valid neuron set
                trial_sets = [trial_sets]
        if not isinstance(trial_sets, list):
            trial_sets = [trial_sets]
            trial_sets.append(self.name)
        else:
            #trial_sets is a list
            for x in trial_sets:
                if isinstance(x, str):
                    if x == self.name:
                        # Neuron name is already in the trial_sets list so do nothing
                        return trial_sets
            # trial_sets does not contain our neruon's valid set so add it
            trial_sets.append(self.name)
        return trial_sets

    def get_firing_traces(self, time_window, blocks=None, trial_sets=None,
                         return_inds=False):
        """ Gets data for default self.use_series within time_window. """
        # session.get_data_array calls self.append_valid_trial_set(trial_sets)
        fr, t = self.session.get_data_array(self.use_series, time_window,
                                        blocks=blocks, trial_sets=trial_sets,
                                        return_inds=True)
        if return_inds:
            return fr, t
        else:
            return fr

    def get_mean_firing_trace(self, time_window, blocks=None, trial_sets=None,
                                return_inds=False):
        warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
        """ Calls session.get_data_array and takes the mean over rows of the output. """
        # session.get_data_array calls self.append_valid_trial_set(trial_sets)
        fr, t = self.get_firing_traces(time_window, blocks=blocks,
                                        trial_sets=trial_sets, return_inds=True)
        if len(fr) == 0:
            # Found no matching data
            if return_inds:
                return fr, t
            else:
                return fr
        with warnings.catch_warnings():
            fr = np.nanmean(fr, axis=0)
        if return_inds:
            return fr, t
        else:
            return fr

    def compute_tuning_by_condition(self, time_window, block='StandTunePre'):
        """ Computes the tuning angle by conditions. The tuning space is computed
        from the eye data as stored in session. Thus it will be ROTATED if the
        joined session returns rotated eye data!
        """
        fr_by_set = []
        theta_by_set = []
        for curr_set in self.session.four_dir_trial_sets:
            curr_fr = self.get_mean_firing_trace(time_window, block, curr_set)
            fr_by_set.append(np.nanmean(curr_fr))
            trial_sets = [curr_set, self.name] # append valid trial set for this neuron
            targ_p, targ_l = self.session.get_mean_xy_traces(
                            "target position", time_window, blocks=block,
                            trial_sets=trial_sets, rescale=False)
            # Just get single vector for each target dimension
            if len(targ_p) > 1:
                targ_p = targ_p[-1] - targ_p[0]
                targ_l = targ_l[-1] - targ_l[0]
            # Compute angle of target position delta in learning/position axis space
            theta_by_set.append(np.arctan2(targ_l, targ_p))
        fr_by_set = np.array(fr_by_set)
        theta_by_set = np.array(theta_by_set)
        theta_order = np.argsort(theta_by_set)
        theta_by_set = theta_by_set[theta_order]
        fr_by_set = fr_by_set[theta_order]
        return theta_by_set, fr_by_set

    def set_optimal_pursuit_vector(self, time_window, block='StandTunePre'):
        """ Saves the pursuit vector with maximum rate according to a cosine
        tuning curve fit to all conditions in 'block' for the average firing
        rate within 'time_window'. """
        theta, rho = self.compute_tuning_by_condition(time_window,
                                                        block=block)
        amp, phase, offset = fit_cos_fixed_freq(theta, rho)
        self.optimal_cos_funs[block] = lambda x: (amp * (np.cos(x + phase)) + offset)
        self.optimal_cos_vectors[block] = -1 * phase
        self.optimal_cos_time_window[block] = time_window

    def set_use_series(self, series_name):
        if not series_name in self.session.get_series_names():
            raise ValueError("Series name {0} not found in this Session's neuron's series names.".format(series_name))
        self.use_series = series_name

    def fit_FR_model(self, blocks, trials, dataseries):
        pass

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

    def get_spikes_ms(self):
        """ Convert spike times in units of indices to units of milliseconds. """
        spikes_ms = self.spike_indices / (self.sampling_rate / 1000)
        return spikes_ms


class PurkinjeCell(Neuron):
    """ Adds functions and tuning for the Complex Spikes associated with
    Purkinje cells in the Neuron class.
    """
    def __init__(self, neuron_dict, name, cs_name=None, session=None, min_CS_ISI=100):
        Neuron.__init__(self, neuron_dict, name, cell_type="PC", session=session)
        self.min_CS_ISI = min_CS_ISI
        if hasattr(self, "session"):
            self.assign_cs_by_trial()
        self.use_series_cs = cs_name if cs_name is not None else ("name" + "_CS")
        self.optimal_cos_funs_cs = {}
        self.optimal_cos_vectors_cs = {}
        self.optimal_cos_time_window_cs = {}

    def join_session(self, session):
        self.session = session
        self.assign_cs_by_trial() # Add this in addition to parent function
        self.compute_valid_trials()
        self.recompute_fits()

    def assign_cs_by_trial(self):
        """ Copy all trial CS indices to a list and REMOVE ISI violations while
        we're at it. """
        self.trial_cs_times = []
        last_cs_time = -np.inf
        for t in range(0, len(self.session)):
            trial_obj = self.session._trial_lists["neurons"][t]
            trial_CS = trial_obj[self.session.meta_dict_name][self.name]['complex_spikes']
            if trial_CS is None:
                self.trial_cs_times.append([])
                continue
            elif len(trial_CS) == 0:
                self.trial_cs_times.append([])
                continue
            else:
                trial_CS_ISIs = np.diff(trial_CS)
                remove_CS = np.zeros((trial_CS.shape[0], ), dtype='bool')
                # shift over so remove second spike from violation
                remove_CS[1:] = trial_CS_ISIs < self.min_CS_ISI
                if (trial_CS[0] - last_cs_time) < self.min_CS_ISI:
                    # Also need to remove first CS this trial for violation
                    remove_CS[0] = True
                last_cs_time = trial_CS[-1]
                trial_CS = trial_CS[~remove_CS]
                if trial_CS.shape[0] == 0:
                    trial_CS = []
                self.trial_cs_times.append(trial_CS)

    def set_mean_CS_prob(self, time_window, blocks=None, trial_sets=None):
        """ Finds the mean CS probability over all 1 ms bins for comparison
        with time varying probabilities by trial type. """
        trial_duration = time_window[1] - time_window[0]
        # get_cs_by_trial calls self.append_valid_trial_set(trial_sets)
        cs_by_trial = self.get_cs_by_trial(time_window, blocks, trial_sets)
        if len(cs_by_trial) == 0:
            raise ValueError("No trials found for input blocks and trial sets! Mean CS probability not set.")
        n_CS = 0
        for t_cs in cs_by_trial:
            n_CS += len(t_cs)
        self.p_CS_per_ms = n_CS / (trial_duration * len(cs_by_trial))

    def get_CS_dataseries_by_trial(self, time_window, blocks=None,
                                        trial_sets=None, nan_sacc=False):
        """ Returns a n trials by t time points numpy array of the CS events in
        ._timeseries.dt width bins. The series is a 1 if a spike was in that bin
        and 0 otherwise. """
        # get_cs_by_trial calls self.append_valid_trial_set(trial_sets)
        cs_by_trial, cs_t_inds = self.get_cs_by_trial(time_window, blocks=blocks,
                                            trial_sets=trial_sets,
                                            return_inds=True)
        CS_dataseries_by_trial = np.zeros((len(cs_by_trial), time_window[1] - time_window[0]))
        if nan_sacc:
            pos_p, pos_l = self.session.get_xy_traces("eye position",
                                    time_window, blocks, cs_t_inds,
                                    return_inds=False)
            is_pos_nan = np.isnan(pos_p)
        else:
            is_pos_nan = np.zeros(CS_dataseries_by_trial.shape, dtype='bool')
        # Just need to find data dt for one trial and all should be the same
        for t_num in range(0, len(self.session)):
            try:
                dt_data = self.session['neurons'][t_num]._timeseries.dt
                # Got a dt
                break
            except:
                # set to default of 1 in case this doesn't work for some reason
                dt_data = 1.0
                # dt_data not found for whatever reason so keep trying
                continue
        # Go through all CS
        for t_cs_ind, t_cs in enumerate(cs_by_trial):
            if len(t_cs) == 0:
                continue
            # Bin CS into 1 ms bins
            CS_dataseries_by_trial[t_cs_ind, np.int32(np.floor( (np.array(t_cs) / dt_data) - time_window[0]))] = 1.0
            CS_dataseries_by_trial[t_cs_ind, is_pos_nan[t_cs_ind, :]] = 0.
        return CS_dataseries_by_trial

    def get_gauss_convolved_CS_by_trial(self, time_window, blocks=None,
                                        trial_sets=None, sigma=50,
                                        cutoff_sigma=4,
                                        nan_sacc=False):
        """ Returns a n trials by t time points numpy array of the smoothed
        CS rates over the blocks and trial sets input. """
        # get_cs_by_trial calls self.append_valid_trial_set(trial_sets)
        cs_by_trial, cs_t_inds = self.get_cs_by_trial(time_window, blocks=blocks,
                                            trial_sets=trial_sets,
                                            return_inds=True)
        smooth_CS_by_trial = np.zeros((len(cs_by_trial), time_window[1] - time_window[0]))
        if nan_sacc:
            pos_p, pos_l = self.session.get_xy_traces("eye position",
                                    time_window, blocks, cs_t_inds,
                                    return_inds=False)
            is_pos_nan = np.isnan(pos_p)
        else:
            is_pos_nan = np.zeros(smooth_CS_by_trial.shape, dtype='bool')

        # Just need to find data dt for one trial and all should be the same
        for t_num in range(0, len(self.session)):
            try:
                dt_data = self.session['neurons'][t_num]._timeseries.dt
                # Got a dt
                break
            except:
                # set to default of 1 in case this doesn't work for some reason
                dt_data = 1.0
                # dt_data not found for whatever reason so keep trying
                continue
        # Go through all CS
        for t_cs_ind, t_cs in enumerate(cs_by_trial):
            if len(t_cs) == 0:
                continue
            # Bin CS into 1 ms bins
            smooth_CS_by_trial[t_cs_ind, np.int32(np.floor( (np.array(t_cs) / dt_data) - time_window[0]))] = 1.0
            smooth_CS_by_trial[t_cs_ind, is_pos_nan[t_cs_ind, :]] = 0.
            smooth_CS_by_trial[t_cs_ind, :] = gauss_convolve(
                                        smooth_CS_by_trial[t_cs_ind, :],
                                        sigma=sigma, cutoff_sigma=cutoff_sigma)
        return smooth_CS_by_trial

    def get_mean_gauss_convolved_CS_by_trial(self, time_window, blocks=None,
                                                trial_sets=None, sigma=50,
                                                cutoff_sigma=4,
                                                nan_sacc=False):
        """ Gets mean over all input trials averaging over output from
        get_gauss_convolved_CS_by_trial above. """
        smooth_CS_by_trial = self.get_gauss_convolved_CS_by_trial(time_window,
                                            blocks=blocks,
                                            trial_sets=trial_sets, sigma=sigma,
                                            cutoff_sigma=cutoff_sigma,
                                            nan_sacc=nan_sacc)
        return np.mean(smooth_CS_by_trial, axis=0)

    def get_cs_by_trial(self, time_window, blocks=None, trial_sets=None,
                         return_inds=False):
        """ Gets complex spikes by trial for the input blocks/trials within
        time_window. Done seprate because there is no point in a timeseries for
        the infrequent CS events. """
        trial_sets = self.append_valid_trial_set(trial_sets)
        t_inds = self.session._parse_blocks_trial_sets(blocks, trial_sets)
        data_out = []
        t_inds_out = []
        for t in t_inds:
            if not self.session._session_trial_data[t]['incl_align']:
                # Trial is not aligned with others due to missing event
                continue
            trial_CS = self.trial_cs_times[t]
            win_cs = []
            if len(trial_CS) == 0:
                # No CSs on this trial
                data_out.append([])
                t_inds_out.append(t)
                continue
            # Align trial CS on current alignment event NOT IN PLACE!!!
            aligned_CS = trial_CS - self.session._session_trial_data[t]['aligned_time']
            for t_cs in aligned_CS:
                if (t_cs >= time_window[0]) & (t_cs < time_window[1]):
                    win_cs.append(t_cs)
            data_out.append(win_cs)
            t_inds_out.append(t)
        if return_inds:
            return data_out, np.array(t_inds_out, dtype=np.int64)
        else:
            return data_out

    def get_cs_rate(self, time_window, blocks=None, trial_sets=None,
                         return_inds=False):
        """Converts the CSs found to rates within the time window. """
        # get_cs_by_trial calls self.append_valid_trial_set(trial_sets)
        fr, t = self.get_cs_by_trial(time_window, blocks=blocks,
                                     trial_sets=trial_sets, return_inds=True)
        if len(fr) > 0:
            # found at least 1 matching trial
            fr_out = np.zeros(len(fr), dtype=np.float64)
            for fr_ind in range(0, len(fr)):
                fr_out[fr_ind] = 1000 * len(fr[fr_ind]) / (time_window[1] - time_window[0])
        else:
            fr_out = fr
        if return_inds:
            return fr_out, t
        else:
            return fr_out

    def get_mean_cs_rate(self, time_window, blocks=None, trial_sets=None,
                                return_inds=False):
        warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
        """ Calls self.get_cs_rate and takes the mean over trials of the output. """
        # get_cs_by_trial calls self.append_valid_trial_set(trial_sets)
        fr, t = self.get_cs_rate(time_window, blocks=blocks,
                                        trial_sets=trial_sets, return_inds=True)
        if len(fr) == 0:
            # Found no matching data
            if return_inds:
                return fr, t
            else:
                return fr
        with warnings.catch_warnings():
            fr = np.nanmean(fr, axis=0)
        if return_inds:
            return fr, t
        else:
            return fr

    def compute_cs_tuning_by_condition(self, time_window, block='StandTunePre'):
        """ Computes the tuning angle by conditions. The tuning space is computed
        from the eye data as stored in session. Thus it will be ROTATED if the
        joined session returns rotated eye data!
        """
        fr_by_set = []
        theta_by_set = []
        for curr_set in self.session.four_dir_trial_sets:
            # get_cs_by_trial calls self.append_valid_trial_set(trial_sets)
            curr_fr = self.get_mean_cs_rate(time_window, block, curr_set)
            fr_by_set.append(curr_fr)
            trial_sets = [curr_set, self.name] # append valid trial set for this neuron
            targ_p, targ_l = self.session.get_mean_xy_traces(
                            "target position", time_window, blocks=block,
                            trial_sets=curr_set, rescale=False)
            # Just get single vector for each target dimension
            if len(targ_p) > 1:
                targ_p = targ_p[-1] - targ_p[0]
                targ_l = targ_l[-1] - targ_l[0]
            # Compute angle of target position delta in learning/position axis space
            theta_by_set.append(np.arctan2(targ_l, targ_p))
        fr_by_set = np.array(fr_by_set)
        theta_by_set = np.array(theta_by_set)
        theta_order = np.argsort(theta_by_set)
        theta_by_set = theta_by_set[theta_order]
        fr_by_set = fr_by_set[theta_order]
        return theta_by_set, fr_by_set

    def set_optimal_pursuit_vector(self, time_window, block='StandTunePre',
                                    cs_time_window=[35, 200]):
        """ Saves the pursuit vector with maximum rate according to a cosine
        tuning curve fit to all conditions in 'block' for the average firing
        rate within 'time_window'. """
        # Get vectors as usual for SS
        super().set_optimal_pursuit_vector(time_window, block=block)

        # Modified functions for getting CS vectors
        theta, rho = self.compute_cs_tuning_by_condition(cs_time_window,
                                                        block=block)
        amp, phase, offset = fit_cos_fixed_freq(theta, rho)
        self.optimal_cos_funs_cs[block] = lambda x: (amp * (np.cos(x + phase)) + offset)
        self.optimal_cos_vectors_cs[block] = -1 * phase
        self.optimal_cos_time_window_cs[block] = cs_time_window

    # def get_CS_spikes_ms(self):
    #     """ Convert CS times in units of indices to units of milliseconds. """
    #     CS_spikes_ms = self. / (self.sampling_rate / 1000)
    #     return CS_spikes_ms


def fit_pcwise_lin_eye_kinematics(self, bin_width=10, bin_threshold=1,
                                fit_constant=True, fit_avg_data=False,
                                quick_lag_step=10, knees=[0., 0.]):
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
        s_dim2 = 9 if fit_constant else 8
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
            # Copy over velocity data to make room for positions
            eye_data[:, :, 4:6] = eye_data[:, :, 2:4]
            eye_data[:, :, 2:4] = eye_data[:, :, 0:2]
            eye_data[:, :, 6:8] = eye_data_series.acc_from_vel(eye_data[:, :, 4:6],
                            filter_win=self.neuron.session.saccade_ind_cushion)
            # Use bin smoothing on data before fitting
            bin_eye_data = bin_data(eye_data, bin_width, bin_threshold)
            if fit_avg_data:
                bin_eye_data = np.nanmean(bin_eye_data, axis=0, keepdims=True)

            # Need to get the +/- position data separate AFTER BINNING AND MEAN!
            select_pursuit = bin_eye_data[:, :, 0] >= knees[0]
            bin_eye_data[~select_pursuit, 0] = 0.0 # Less than knee dim0 = 0
            bin_eye_data[select_pursuit, 2] = 0.0 # Less than knee dim2 = 0
            select_learning = bin_eye_data[:, :, 1] >= knees[1]
            bin_eye_data[~select_learning, 1] = 0.0 # Less than knee dim1 = 0
            bin_eye_data[select_learning, 3] = 0.0 # Less than knee dim3 = 0

            bin_eye_data = bin_eye_data.reshape(bin_eye_data.shape[0]*bin_eye_data.shape[1], bin_eye_data.shape[2], order='C')
            temp_FR = binned_FR.reshape(binned_FR.shape[0]*binned_FR.shape[1], order='C')
            select_good = ~np.any(np.isnan(bin_eye_data), axis=1)
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
                # Copy over velocity data to make room for positions
                eye_data[:, :, 4:6] = eye_data[:, :, 2:4]
                eye_data[:, :, 2:4] = eye_data[:, :, 0:2]
                eye_data[:, :, 6:8] = eye_data_series.acc_from_vel(eye_data[:, :, 4:6],
                                filter_win=self.neuron.session.saccade_ind_cushion)
                # Use bin smoothing on data before fitting
                bin_eye_data = bin_data(eye_data, bin_width, bin_threshold)
                if fit_avg_data:
                    bin_eye_data = np.nanmean(bin_eye_data, axis=0, keepdims=True)

                # Need to get the +/- position data separate AFTER BINNING AND MEAN!
                select_pursuit = bin_eye_data[:, :, 0] >= knees[0]
                bin_eye_data[~select_pursuit, 0] = 0.0 # Less than knee dim0 = 0
                bin_eye_data[select_pursuit, 2] = 0.0 # Less than knee dim2 = 0
                select_learning = bin_eye_data[:, :, 1] >= knees[1]
                bin_eye_data[~select_learning, 1] = 0.0 # Less than knee dim1 = 0
                bin_eye_data[select_learning, 3] = 0.0 # Less than knee dim3 = 0


                bin_eye_data = bin_eye_data.reshape(bin_eye_data.shape[0]*bin_eye_data.shape[1], bin_eye_data.shape[2], order='C')
                temp_FR = binned_FR.reshape(binned_FR.shape[0]*binned_FR.shape[1], order='C')
                select_good = ~np.any(np.isnan(bin_eye_data), axis=1)
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
                                'predict_fun': self.predict_pcwise_lin_eye_kinematics,
                                'knees': knees}
        
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


def comp_block_scaling_factors(primary_blocks, scaled_blocks, neuron, time_window=[-100, 900], 
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
    if not isinstance(scaled_blocks, list):
        scaled_blocks = [scaled_blocks]
    for ind, block in reversed(list(enumerate(scaled_blocks))):
        for pblock in primary_blocks:
            if pblock == block:
                del scaled_blocks[ind]
                break
    
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
    fr_fix, fr_inds_all = neuron.get_firing_traces(fix_time_window, all_blocks,
                                                    trial_sets, return_inds=True)
    fr_fix = np.nanmean(fr_fix, axis=1)
    fr_fix = gauss_convolve(fr_fix, 5, cutoff_sigma=4, pad_data=True)

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


def find_stable_ranges(spikes_ms, win_size, duration, tol_percent=20, min_rate=0.05, sigma_smooth=None):
    """ Duration is in units of 'win_size', i.e. for a duration of N win_sizes. """
    tol_percent = abs(tol_percent)
    if tol_percent > 1:
        tol_percent = tol_percent / 100
    duration = int(max(duration, 1)) # duration must be at least 1 bin

    t_start = 0
    bin_rates = []
    bin_t_wins = []
    # Get all the binned firing rates
    while t_start < spikes_ms[-1]:
        t_stop = min(t_start + win_size, spikes_ms[-1] + 1)
        bin_spikes = np.count_nonzero((spikes_ms >= t_start) & (spikes_ms < t_stop))
        bin_rates.append(1000* bin_spikes / (t_stop - t_start))
        bin_rates[-1] = max(1e-6, bin_rates[-1]) # avoid zero division later
        bin_t_wins.append([t_start, t_stop])
        t_start = t_stop
    bin_rates = np.array(bin_rates)
    raw_bin_rates = np.copy(bin_rates)

    if sigma_smooth is not None:
        if sigma_smooth <= 0.:
            raise ValueError("sigma_smooth must be greater than zero but {0} was given.".format(sigma_smooth))
        bin_rates = gauss_convolve(bin_rates, sigma_smooth, pad_data=True)

    # Need to find segments of "stability"
    seg_bins = [[]]
    curr_seg = 0
    nbin = 0
    found_start = False
    # Look for "duration" number of stable bins for a new seg starting point
    while nbin < len(bin_rates):
        if nbin == len(bin_rates)-1:
            # Last bin, just append if still going
            if ( (bin_rates[nbin] >= min_rate) and (found_start) ):
                seg_bins[curr_seg].append(nbin)
                nbin += 1 # Won't hurt but not currently necessary
                break # Finished
            elif not found_start:
                # On last bin and have not found a start
                if duration <= 1:
                    # Any bin is a new start so add it
                    curr_seg += 1
                    seg_bins.append([])
                    seg_bins[curr_seg].append(nbin)
                else:
                    # Not enough bins left to start a new seg so we are done
                    nbin += 1 # Won't hurt but not currently necessary
                    break
            else:
                # Have not found start at this point
                pass
        if not found_start:
            # Need to find a stable start point to begin a segment
            max_check_bin = min(nbin + duration, len(bin_rates)-1)
            for bin_ind in range(nbin, max_check_bin):
                if ( ((abs(bin_rates[bin_ind + 1] - bin_rates[bin_ind]) / bin_rates[bin_ind]) > tol_percent)
                    or (bin_rates[bin_ind + 1] < min_rate) ):
                    # Jump between bins over threshold within the duration window
                    found_start = False
                    nbin += 1
                    break
                # Made it here on last iteration then we found a stable start point of at least length "duration"
                found_start = True
            if found_start:
                for bin_rs in range(nbin, max_check_bin):
                    seg_bins[curr_seg].append(bin_rs)
                nbin = max_check_bin
        else:
            if ( ((abs(bin_rates[nbin + 1] - bin_rates[nbin]) / bin_rates[nbin]) > tol_percent)
                or (bin_rates[nbin + 1] < min_rate) ):
                # Found a sudden change point
                consec_bad_bins = []
                max_check_bin = min(nbin + duration, len(bin_rates)-1)
                for bin_ind in range(nbin, max_check_bin):
                    # Contrast nbin with all next bins to see if firing rate returns
                    if ( ((abs(bin_rates[bin_ind + 1] - bin_rates[nbin]) / bin_rates[nbin]) > tol_percent)
                        or (bin_rates[bin_ind + 1] < min_rate) ):
                        consec_bad_bins.append(bin_ind + 1)
                    else:
                        # We found a bin that recovered
                        break
                if len(consec_bad_bins) >= duration:
                    # Found enough changed bins that we start a new segment
                    seg_bins[curr_seg].append(nbin)
                    curr_seg += 1
                    seg_bins.append([])
                    found_start = False
                    nbin += 1
                else:
                    # Changed bins were too short of a blip so just keep going
                    seg_bins[curr_seg].append(nbin)
                    for cbb in consec_bad_bins:
                        # Only add if this is over min_rate
                        if bin_rates[cbb] < min_rate:
                            curr_seg += 1
                            seg_bins.append([])
                        else:
                            seg_bins[curr_seg].append(cbb)
                    nbin = cbb + 1 # pick up at the first bin that didnt get added to consec_bad_bins
            else:
                seg_bins[curr_seg].append(nbin)
                nbin += 1

    # One easy check for something going terribly wrong in the above confusing logic
    for sb_ind, sb in enumerate(seg_bins):
        unique_sb = np.unique(sb)
        array_sb = np.sort(np.array(sb))
        if ~np.all(unique_sb == array_sb):
            raise RuntimeError("Must have double counted or skipped something for seg {0}!".format(sb_ind))
    # Remove any segs that never had a bin added
    for sb_ind in reversed(range(0, len(seg_bins))):
        if len(seg_bins[sb_ind]) == 0:
            del seg_bins[sb_ind]
    # Want numpy array output for easy indexing later
    for sb_ind in range(0, len(seg_bins)):
        seg_bins[sb_ind] = np.array(seg_bins[sb_ind])
    bin_t_wins = np.array(bin_t_wins)

    return seg_bins, bin_rates, bin_t_wins, raw_bin_rates

def get_stable_time_wins(seg_bins, bin_rates, bin_t_wins, tol_percent=15,
                         sigma_smooth=None, cutoff_sigma=4):
    """
    """
    tol_percent = abs(tol_percent)
    if tol_percent > 1:
        tol_percent = tol_percent / 100

    seg_medians = []
    all_valid_median = []
    for sb in seg_bins:
        if sigma_smooth is not None:
            # Do smoothing WITHIN segments only
            if sigma_smooth <= 0.:
                raise ValueError("sigma_smooth must be greater than zero but {0} was given.".format(sigma_smooth))
            if (sb[-1] - sb[0]) >= (2*cutoff_sigma*sigma_smooth + 1):
                # Seg should have enough points to do useful convolution
                bin_rates[sb[0]:sb[-1]+1] = gauss_convolve(bin_rates[sb[0]:sb[-1]+1], sigma_smooth, cutoff_sigma, pad_data=True)
            else:
                # Very few data points in segment so just take median
                bin_rates[sb[0]:sb[-1]+1] = np.median(bin_rates[sb[0]:sb[-1]+1])
        seg_rates = bin_rates[sb]
        seg_medians.append(np.median(seg_rates))
        all_valid_median.extend(seg_rates)
    all_valid_median = np.median(all_valid_median)

    # Try to determine best "main" segment
    best_dist = np.inf
    best_duration = 0
    best_seg_num = 0
    check_duration = False
    for sm_ind, sm in enumerate(seg_medians):
        curr_dist = np.abs(sm - all_valid_median)
        curr_duration = len(seg_bins[sm_ind])
        if curr_duration > (len(bin_rates) // 2):
            # This seg has duration over half of the entire recording so take it as base point
            best_dist = curr_dist
            best_duration = curr_duration
            best_seg_num = sm_ind
            break
        elif ( ((curr_dist/all_valid_median) < tol_percent) and (curr_duration > best_duration) ):
            # Prefer longer duration near median than absolute nearest median segment
            best_dist = curr_dist
            best_duration = curr_duration
            best_seg_num = sm_ind
            check_duration = True
        elif ( (curr_dist < best_dist) and (not check_duration) ):
            # If we haven't selected anything based on duration above, simply choose the closest median
            best_dist = curr_dist
            best_duration = curr_duration
            best_seg_num = sm_ind

    # Temporally attempt to join adjacent good segments
    keep_segs = []
    keep_segs.append(seg_bins[best_seg_num])
    if best_seg_num > 0:
        # Check backward from best
        ref_seg_ind = best_seg_num
        for com_seg_ind in range(best_seg_num-1, -1, -1):
            curr_dist = np.abs(bin_rates[seg_bins[ref_seg_ind][0]] - bin_rates[seg_bins[com_seg_ind][-1]])
            curr_median_dist = np.abs(seg_medians[ref_seg_ind] - seg_medians[com_seg_ind])
            if (curr_dist / (bin_rates[seg_bins[ref_seg_ind][0]])) < tol_percent:
                # Connect/add this seg for keeping
                keep_segs.append(seg_bins[com_seg_ind])
                ref_seg_ind = com_seg_ind
            elif (curr_median_dist / seg_medians[ref_seg_ind]) < tol_percent:
                # medians are similar even if endpoints not
                # Connect/add this seg for keeping
                keep_segs.append(seg_bins[com_seg_ind])
                ref_seg_ind = com_seg_ind
    if best_seg_num < len(seg_bins)-1:
        # Check forward from best
        ref_seg_ind = best_seg_num
        for com_seg_ind in range(best_seg_num+1, len(seg_bins)):
            curr_dist = np.abs(bin_rates[seg_bins[ref_seg_ind][-1]] - bin_rates[seg_bins[com_seg_ind][0]])
            curr_median_dist = np.abs(seg_medians[ref_seg_ind] - seg_medians[com_seg_ind])
            if (curr_dist / (bin_rates[seg_bins[ref_seg_ind][-1]])) < tol_percent:
                # Connect/add this seg for keeping
                keep_segs.append(seg_bins[com_seg_ind])
                ref_seg_ind = com_seg_ind
            elif (curr_median_dist / seg_medians[ref_seg_ind]) < tol_percent:
                # medians are similar even if endpoints not
                # Connect/add this seg for keeping
                keep_segs.append(seg_bins[com_seg_ind])
                ref_seg_ind = com_seg_ind

    # Ensure time windows are in needed tempral order and gather good ones
    win_order = np.argsort(bin_t_wins[:, 0])
    bin_t_wins = bin_t_wins[win_order, :]
    keep_time_wins = []
    for ks in keep_segs:
        keep_time_wins.append(bin_t_wins[ks, :])
    keep_time_wins = np.vstack(keep_time_wins)
    win_order = np.argsort(keep_time_wins[:, 0])
    keep_time_wins = keep_time_wins[win_order, :]

    # Convert all time windows to simple continuous 2 element windows
    stable_time_wins = []
    curr_start = keep_time_wins[0, 0]
    curr_stop = keep_time_wins[-1, 1]
    for ind in range(0, keep_time_wins.shape[0]):
        if ind == keep_time_wins.shape[0]-1:
            # Last time window so must be stop
            curr_stop = keep_time_wins[ind, 1]
            if curr_stop <= curr_start:
                # should not be possible
                raise RuntimeError("Problem in code for finding stable time windows!")
            stable_time_wins.append([curr_start, curr_stop])
        elif keep_time_wins[ind, 1] != keep_time_wins[ind+1, 0]:
            # Found disconnect in times
            curr_stop = keep_time_wins[ind, 1]
            if curr_stop <= curr_start:
                # should not be possible
                raise RuntimeError("Problem in code for finding stable time windows!")
            stable_time_wins.append([curr_start, curr_stop])
            curr_start = keep_time_wins[ind+1, 0]

    return stable_time_wins
