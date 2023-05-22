import argparse
import sys
import os
import logging
import pickle
from LearnDirTunePurk.build_session import create_neuron_session
from NeuronAnalysis.fit_NN_model import FitNNModel
from NeuronAnalysis.fit_learning_model import get_intrisic_rate_and_CSwin


fit_blocks=["StandTunePre", "StabTunePre"]
fit_trial_sets=None
fit_time_window=[-300, 1100]
lag_range_eye=[-50, 150]

learn_blocks=["Learning"]
n_learn_trials_to_fit = 100
weights_blocks=["Learning"]
training_time_window = [-2100, 1450]

bin_width = 10
bin_threshold = 5
files_to_fit = ["LearnDirTunePurk_Dandy_29", "LearnDirTunePurk_Dandy_30",
                "LearnDirTunePurk_Dandy_31", "LearnDirTunePurk_Dandy_40",
                "LearnDirTunePurk_Dandy_44" ]

def setup_logger(filename):
    log_filename = f"{filename}.log"
    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    # Add file handler
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.DEBUG)
    # Create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    # Clear any existing handlers
    logger.handlers.clear()
    # Add the handlers to the logger
    logger.addHandler(file_handler)
    # Redirect stdout and stderr
    sys.stdout = open(log_filename, 'w')
    sys.stderr = sys.stdout
    # Now any print statements or uncaught exceptions will go to your log file
    print(f"Setup to log file: {filename}")

def root_fname(filename):
    """ Strips input file name into "LearnDirTune..." root format"""
    fname = filename.split(".")[0]
    if fname[-4:].lower() == "_viz":
        fname = fname.split("_viz")[0]
    if fname[0:8].lower() == "neurons_":
        fname = fname.split("neurons_")[1]
    if fname[0:16] != "LearnDirTunePurk":
        raise ValueError(f"Could not find base root 'LearnDirTunePurk' in filename {filename}. Found {fname} instead.")
    fnum = int(fname[-2:])

    return fname, fnum

""" Call this script to load all the files hard coded in "files_to_fit" and fit
    all the confirmed PCs found there.

    pc_learning_model.py --neurons_dir "/path/to/neurons" --PL2_dir "/path/to/PL2"
                        --maestro_dir "/path/to/maestro"
                        --maestro_save_dir "/path/to/save" --save_dir "my_save_name"
    """
if __name__ == '__main__':
    # Get all the inputs
    parser = argparse.ArgumentParser()
    parser.add_argument("--neurons_dir", default="/home/nate/neuron_viz_final/")
    parser.add_argument("--PL2_dir", default="/mnt/isilon/home/nathan/Data/LearnDirTunePurk/PL2FilesRaw/")
    parser.add_argument("--maestro_dir", default='/mnt/isilon/home/nathan/Data/LearnDirTunePurk/MaestroFiles/')
    parser.add_argument("--maestro_save_dir", default='/home/nate/Documents/MaestroPickles/')
    parser.add_argument("--save_dir", default=None)
    args = parser.parse_args()
    if args.save_dir is None:
        args.save_dir = args.neurons_dir

    # Get list of files in neurons_dir
    file_list = os.listdir(args.neurons_dir)
    for filename in file_list:
        full_path = os.path.join(args.neurons_dir, filename)
        if not os.path.isfile(full_path):  # Skip directories
            continue
        fname, fnum = root_fname(filename)
        if fname not in files_to_fit:
            # File not in list of known PCs to fit so skip making session
            continue
        print(f"Building session for {fname}.")
        # Build the session object of the PC to be fit
        ldp_sess = create_neuron_session(fname, args.neurons_dir, args.PL2_dir, args.maestro_dir,
                            save_maestro=True, maestro_save_dir=args.maestro_save_dir,
                            rotate_eye_data=True)
        # Make the Gaussian smoothed data we will fit to
        ldp_sess.gauss_convolved_FR(10, cutoff_sigma=4, series_name="_gauss")
        # Get indices of the desired n learning trials
        test_t_win = [ldp_sess.blocks['Learning'][0],
                      min(ldp_sess.blocks['Learning'][0] + n_learn_trials_to_fit, ldp_sess.blocks['Learning'][1])]
        learn_trial_sets = tuple([x for x in range(test_t_win[0], test_t_win[1])])

        # Look for all confirmed PCs here
        for n_name in ldp_sess.get_neuron_names():
            if n_name[0:3] != "PC_":
                # Not a confirmed PC so skip
                continue
            save_name = os.path.join(args.save_dir, f"learn_model_{n_name}_{fname}")
            setup_logger(save_name)
            logging.info(f"Starting fit for unit {n_name}.")
            print(f"starting fit of neuron {n_name}")
            neuron = ldp_sess.neuron_info[n_name]
            fit_NN = FitNNModel(neuron, time_window=fit_time_window, blocks=fit_blocks, trial_sets=fit_trial_sets,
                        lag_range_pf=lag_range_eye, use_series=None)

            result = get_intrisic_rate_and_CSwin(fit_NN, learn_blocks,
                                    learn_trial_sets, training_time_window,
                                    bin_width=bin_width, bin_threshold=bin_threshold)
            logging.info(f"Done fitting {n_name} and now saving to {save_name}.")
            with open(save_name, "wb") as fp:
                pickle.dump(result, fp, protocol=-1)
            sys.stdout.close()
            sys.stderr.close()
    print("All Done!")
