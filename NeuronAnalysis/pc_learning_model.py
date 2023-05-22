import argparse
import pickle
from LearnDirTunePurk.build_session import create_neuron_session



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

    # Build the session object of the PC to be fit
    ldp_sess = create_neuron_session(fname, neurons_dir, PL2_dir, maestro_dir,
                        save_maestro=True, maestro_save_dir=maestro_save_dir,
                        rotate_eye_data=True)
