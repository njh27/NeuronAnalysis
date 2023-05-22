import argparse
import sys
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
test_t_win = [ldp_sess.blocks['Learning'][0], ldp_sess.blocks['Learning'][0] + 100]
learn_trial_sets=np.arange(test_t_win[0], test_t_win[1])
# learn_trial_sets=None
weights_blocks=["Learning"]
training_time_window = [-2100, 1450]

files_to_fit = ["LearnDirTunePurk_Dandy_29", "LearnDirTunePurk_Dandy_30",
                "LearnDirTunePurk_Dandy_31", "LearnDirTunePurk_Dandy_40",
                "LearnDirTunePurk_Dandy_44" ]

def gather_neurons(neurons_dir, PL2_dir, maestro_dir, maestro_save_dir,
                    cell_types, data_fun, data_fun_args, data_fun_kwargs):
    """ Loads data according to the name of the files input in neurons dir.
    Creates a session from the maestro data and joins the corresponding
    neurons from the neurons file. Goes through all neurons and if their name
    is found in the list 'cell_types', then 'data_fun' is called on that neuron
    and its output is appended to a list under the output dict key 'neuron_name'.
    """
    rotate = False
    print("Getting WITHOUT rotating data!!")
    if not isinstance(cell_types, list):
        cell_types = [cell_types]
    out_data = {}
    for f in os.listdir(neurons_dir):
        fname = f
        fname = fname.split(".")[0]
        if fname[-4:].lower() == "_viz":
            fname = fname.split("_viz")[0]
        if fname[0:8].lower() == "neurons_":
            fname = fname.split("neurons_")[1]
        fnum = int(fname[-2:])
        save_name = maestro_save_dir + fname + "_maestro"
        fname_PL2 = fname + ".pl2"
        fname_neurons = "neurons_" + fname + "_viz.pkl"
        neurons_file = neurons_dir + fname_neurons

""" Call this script to load all the files hard coded in "files_to_fit" and fit
    all the confirmed PCs found there.

    pc_learning_model.py --neurons_dir "/path/to/neurons" --PL2_dir "/path/to/PL2"
                        --maestro_dir "/path/to/maestro"
                        --maestro_save_dir "/path/to/save" --save_name "my_save_name"
    """
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--neurons_dir", default="/home/nate/neuron_viz_final/")
    parser.add_argument("--PL2_dir", default="/mnt/isilon/home/nathan/Data/LearnDirTunePurk/PL2FilesRaw/")
    parser.add_argument("--maestro_dir", default='/mnt/isilon/home/nathan/Data/LearnDirTunePurk/MaestroFiles/')
    parser.add_argument("--maestro_save_dir", default='/home/nate/Documents/MaestroPickles/')
    parser.add_argument("--save_name", default=None)
    args = parser.parse_args()
    neurons_dir = args.neurons_dir
    PL2_dir = args.PL2_dir
    maestro_dir = args.maestro_dir
    maestro_save_dir = args.maestro_save_dir

    if args.save_name is None:
        fname = "default_fname" # replace this with your own default fname
        save_name = maestro_save_dir + fname + "_maestro"
    else:
        save_name = args.save_name

    log_filename = "logfile.log"
    logging.basicConfig(filename=log_filename, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    # Now we redirect stdout and stderr
    sys.stdout = open(log_filename, 'w')
    sys.stderr = sys.stdout
    # Now any print statements or uncaught exceptions will go to your log file
    print("This will go to the log file!")

    logging.debug("Debug information")
    logging.info("Useful information")
    logging.warning("A warning")
    logging.error("An error occurred")
    logging.critical("A critical error occurred")





    # Build the session object of the PC to be fit
    ldp_sess = create_neuron_session(fname, neurons_dir, PL2_dir, maestro_dir,
                        save_maestro=True, maestro_save_dir=maestro_save_dir,
                        rotate_eye_data=True)

    ldp_sess.gauss_convolved_FR(10, cutoff_sigma=4, series_name="_gauss")

    test_name = "PC_00"
    for n_name in ldp_sess.get_neuron_names():
        if n_name[0:3] != "PC_":
            # Not a confirmed PC so skip
            continue
        neuron = ldp_sess.neuron_info[n_name]
        fit_NN = FitNNModel(neuron, time_window=fit_time_window, blocks=fit_blocks, trial_sets=fit_trial_sets,
                    lag_range_pf=lag_range_eye, use_series=None)

        result, best_intrinsic_rate, best_CS_wins = get_intrisic_rate_and_CSwin(fit_NN,
                                                                    learn_blocks, learn_trial_sets, learn_fit_window=training_time_window,
                                                                    bin_width=bin_width, bin_threshold=bin_threshold)

    print("All Done!")
    sys.stdout.close()
    sys.stderr.close()
