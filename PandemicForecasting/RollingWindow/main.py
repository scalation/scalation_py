#Import required models
from models.FFNN import FFNN
from models.LSTM import LSTM
from models.GRU import GRU
from models.LSTM_Seq2Seq import LSTM_Seq2Seq
from models.GRU_Seq2Seq import GRU_Seq2Seq
from models.LSTM_Seq2Seq_Att import LSTM_Seq2Seq_Att
from models.GRU_Seq2Seq_Att import GRU_Seq2Seq_Att
from models.Transformer import Transformer
from models.Informer import Informer
from models.FEDformer import FEDformer
from models.Autoformer import Autoformer
from models.RandomWalk import RandomWalk
from models.PatchTST import PatchTST
from models.Dlinear import Dlinear
from models.Nlinear import Nlinear

#Import required libraries
import torch
import numpy as np
from exp.exp_main import Exp_Main
import random
from simple_colors import *

# Dictionary to map model names to their respective classes
model_dict = {
    'Autoformer': Autoformer,
    'Transformer': Transformer,
    'Informer': Informer,
    'FEDformer': FEDformer,
    "Dlinear": Dlinear,
    "Nlinear": Nlinear,
    "FFNN": FFNN,
    "LSTM": LSTM,
    "GRU": GRU,
    "LSTM_Seq2Seq": LSTM_Seq2Seq,
    "GRU_Seq2Seq": GRU_Seq2Seq,
    "LSTM_Seq2Seq_Att": LSTM_Seq2Seq_Att,
    "GRU_Seq2Seq_Att": GRU_Seq2Seq_Att,
    "PatchTST": PatchTST,
    "RandomWalk": RandomWalk
}

# Dictionary to map dataset names to their input-output dimensions
dataset_dict = {
    "national_illness": 7,
    "covid_till14May22": 7
}

# Global variables for the dataset and its input-output dimension
global_dataset = "covid_till14May22"
in_out_dim = dataset_dict[global_dataset]
fix_seed = 2021 # Fixed seed for reproducibility

# Dictionary to store parameters for each model for hyperparams tuning
#The hyperparams specified below are only for reproducibility
reasonable_params_dict = {
    "FFNN": [[0.01], [7], [64], [16], [0]],
    "LSTM": [[0.01], [15], [256], [16], [1]],
    "GRU": [[0.01], [15], [128],[16], [2]],
    "LSTM_Seq2Seq": [[0.01], [10], [256], [16], [2]],
    "GRU_Seq2Seq": [[0.01], [15], [256], [16], [1]],
    "LSTM_Seq2Seq_Att": [[0.01], [10], [64], [16], [2]],
    "GRU_Seq2Seq_Att": [[0.01], [7], [64], [16], [2]],
    "Transformer": [[0.001], [7], [512], [16], [2]],
    "Informer": [[0.001], [7], [128], [16], [5]],
    "Autoformer": [[0.001], [7], [128], [16], [2]],
    "FEDformer": [[0.001], [15], [128], [16], [2]],
    "Dlinear": [[0.1], [10], [100], [16], [0]],
    "PatchTST": [[0.0001], [15], [2048], [16], [2]],
    "Nlinear": [[0.02], [10], [100], [16], [0]],
    "RandomWalk": [[0.01], [6], [100], [1], [0]]
}
# List of models to be tested
mlist = ["FFNN"]
# mlist = ["RandomWalk", "FFNN","LSTM", "GRU", "LSTM_Seq2Seq", "GRU_Seq2Seq", "LSTM_Seq2Seq_Att", "GRU_Seq2Seq_Att","Transformer",
#          "Informer", "Autoformer", "FEDformer", "Dlinear", "Nlinear", "PatchTST"]

# Window size for the sliding window approach
window_size = 19 #define window size, which should be equal to the number of test steps


# Lists to store performance metrics
smape_all = [] #store best metrics for the hyperparameters
mae_all = []
realmae_all = []

smape_pl = [] #store best metrics for the windows
mae_pl = []
realmae_pl = []

# Dictionary to store the final metrics results for each model
final_metrics = {"RandomWalk": [200, 200, 200], "PatchTST": [200, 200, 200], "FEDformer": [200, 200, 200],
                 "Dlinear": [200, 200, 200], "Nlinear": [200, 200, 200], "Informer": [200, 200, 200],
                 "Autoformer": [200, 200, 200], "Transformer": [200, 200, 200], "FFNN": [200, 200, 200],
                 "LSTM": [200, 200, 200], "GRU": [200, 200, 200],
                 "GRU_Seq2Seq": [200, 200, 200], "LSTM_Seq2Seq": [200, 200, 200], "LSTM_Seq2Seq_Att": [200, 200, 200],
                 "GRU_Seq2Seq_Att": [200, 200, 200], "RandomWalk": [200, 200, 200]}

# Loop through each prediction length (horizons)
horizons = 6
for pl in [horizons]:
    for m in mlist: # Loop through each model in the list
        figs = ""
        test_setting = ""
        print("MODEL", blue(m, 'bold'))

        # Loop through the hyperparameters for the current model
        for lr in reasonable_params_dict[m][0]:
            for sl in reasonable_params_dict[m][1]:
                for model_dim in reasonable_params_dict[m][2]:
                    for batch_size in reasonable_params_dict[m][3]:
                        for layers in reasonable_params_dict[m][4]:
                            print(layers, lr, sl, model_dim, batch_size)

                            for w in range(window_size): # Loop through each window
                                random.seed(fix_seed) # Set seeds for reproducibility
                                torch.manual_seed(fix_seed)
                                np.random.seed(fix_seed)

                                # Configuration class for setting model parameters
                                class Configs(object):
                                    units_layer1 = model_dim  # for FFNN
                                    units_layer2 = model_dim  # for FFNN
                                    task_name = "long_term_forecast"
                                    is_training = True
                                    root_path = "datasets/"
                                    data_path = global_dataset + ".csv"
                                    model_id = "ModelRun"
                                    model = model_dict[m]
                                    data = "custom"
                                    features = 'M'
                                    seq_len = sl
                                    label_len = sl - pl
                                    pred_len = pl
                                    e_layers = layers
                                    d_layers = layers - 1
                                    factor = 3
                                    enc_in = in_out_dim
                                    dec_in = in_out_dim
                                    c_out = in_out_dim
                                    train_epochs = 20
                                    target = "OT" if global_dataset == "national_illness" else "new_cases"
                                    freq = 'h'
                                    checkpoints = './checkpoints/'
                                    bucket_size = 4
                                    n_hashes = 4
                                    d_model = model_dim
                                    n_heads = 8
                                    d_ff = 2048
                                    distil = True
                                    dropout = 0.05
                                    embed = 'timeF'
                                    activation = 'gelu'
                                    output_attention = False
                                    num_workers = 0
                                    itr = 2
                                    batch_size = batch_size
                                    patience = 3
                                    learning_rate = lr
                                    des = 'test'
                                    loss = 'mse'
                                    lradj = 'type1'
                                    use_amp = False
                                    use_gpu = True
                                    gpu = 0
                                    use_multi_gpu = False
                                    wavelet = 0
                                    ab = 0
                                    modes = 64
                                    mode_select = 'random'
                                    version = 'Fourier'
                                    moving_avg = [12, 24]
                                    L = 3
                                    base = 'legendre'
                                    cross_activation = 'tanh'
                                    individual = True
                                    w = w


                                args = Configs()
                                # Check GPU availability
                                args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
                                if args.use_gpu and args.use_multi_gpu:
                                    args.devices = args.devices.replace(' ', '')
                                    device_ids = args.devices.split(',')
                                    args.device_ids = [int(id_) for id_ in device_ids]
                                    args.gpu = args.device_ids[0]
                                Exp = Exp_Main
                                if args.is_training:

                                    # Initialize the results dictionary with high initial error values
                                    results_dict = {"RandomWalk": [200, 200, 200], "PatchTST": [200, 200, 200],
                                                    "FEDformer": [200, 200, 200], "Dlinear": [200, 200, 200],
                                                    "Nlinear": [200, 200, 200], "Informer": [200, 200, 200],
                                                    "Autoformer": [200, 200, 200], "Transformer": [200, 200, 200],
                                                    "FFNN": [200, 200, 200], "LSTM": [200, 200, 200],
                                                    "GRU": [200, 200, 200],
                                                    "GRU_Seq2Seq": [200, 200, 200], "LSTM_Seq2Seq": [200, 200, 200],
                                                    "LSTM_Seq2Seq_Att": [200, 200, 200],
                                                    "GRU_Seq2Seq_Att": [200, 200, 200], "RandomWalk": [200, 200, 200]}

                                    for ii in range(args.itr):
                                        # setting record of experiments
                                        setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_w{}_it{}'.format(
                                            args.model_id,
                                            args.model,
                                            args.data,
                                            args.features,
                                            args.seq_len,
                                            args.label_len,
                                            args.pred_len,
                                            args.d_model,
                                            args.n_heads,
                                            args.e_layers,
                                            args.d_layers,
                                            args.d_ff,
                                            args.factor,
                                            args.embed,
                                            args.distil,
                                            args.des, args.w, ii)

                                        exp = Exp(args)  # set experiments
                                        # print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
                                        exp.train(setting)

                                        # print('testing : {}'.format(setting))
                                        # print("=====================================")
                                        realmae, mae, smape = exp.test(setting)

                                        # Update results_dict with the best smape score for each window
                                        if type(results_dict[m][2]) == int:

                                            if results_dict[m][2] > np.mean(smape):
                                                results_dict[m] = [realmae, mae, smape]
                                        else:
                                            if np.mean(results_dict[m][2]) > np.mean(smape):
                                                results_dict[m] = [realmae, mae, smape]

                                    # Append the best results of the current iteration
                                    smape_all.append(results_dict[m][2])
                                    mae_all.append(results_dict[m][1])
                                    realmae_all.append(results_dict[m][0])

                            # Compute mean metrics over all windows
                            for horiz in range(pl):
                                smape_pl.append(np.mean([smape_all[w][horiz] for w in range(window_size)]))
                                mae_pl.append(np.mean([mae_all[w][horiz] for w in range(window_size)]))
                                realmae_pl.append(np.mean([realmae_all[w][horiz] for w in range(window_size)]))

                            #print("RESULTS: -----" + str(smape_pl) + str(mae_pl) + str(realmae_pl) + "-----")

                            # Update final_metrics if the current smape_pl is better
                            if type(final_metrics[m][2]) == int:
                                if final_metrics[m][2] > np.mean(smape_pl):
                                    final_metrics[m] = [realmae_pl, mae_pl, smape_pl]
                                    test_setting = 'testing : {}'.format(setting)
                                    figs = "LR: " + str(lr) + ", SL: " + str(sl) + ", Mdim: " + str(
                                        model_dim) + ", L: " + str(layers) + ", BS: " + str(batch_size)

                            else:
                                if np.mean(final_metrics[m][2]) > np.mean(smape_pl):
                                    final_metrics[m] = [realmae_pl, mae_pl, smape_pl]
                                    test_setting = 'testing : {}'.format(setting)
                                    figs = "LR: " + str(lr) + ", SL: " + str(sl) + ", Mdim: " + str(
                                        model_dim) + ", L: " + str(layers) + ", BS: " + str(batch_size)
                            #print("FINAL_METRICS ", final_metrics[m], figs, test_setting)

                            # Reset the metric lists for the next configuration
                            smape_pl = []
                            mae_pl = []
                            realmae_pl = []

                            smape_all = []
                            mae_all = []
                            realmae_all = []

                            torch.cuda.empty_cache() # Clear the CUDA cache

        # Print the best results for the current prediction length
        print("-----------Best (realMAE, normMAE, realSMAPE) for " + str(
            pl) + " pred length " + m + ": " + figs + "------------------" + test_setting)
        for horiz in range(pl):
            print(str(horiz) + ": ", final_metrics[m][0][horiz], final_metrics[m][1][horiz], final_metrics[m][2][horiz])
        print("Average realMAE: ", sum([final_metrics[m][0][horiz] for horiz in range(pl)]) / pl)
        print("Average normMAE: ", sum([final_metrics[m][1][horiz] for horiz in range(pl)]) / pl)
        print("Average realSMAPE: ", sum([final_metrics[m][2][horiz] for horiz in range(pl)]) / pl)
