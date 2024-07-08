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
    "FFNN": [[0.01], [7], [128], [16], [0]],  # 0.01,  7,  128,256
    "LSTM": [[0.01], [10], [256], [16], [1]],  # 0.01,    10,  256,  1
    "GRU": [[0.01], [10], [128], [16], [2]],  # 0.01, 10, 128, 2
    "LSTM_Seq2Seq": [[0.01], [10], [128], [16], [1]],  # 0.01 10 128 1
    "GRU_Seq2Seq": [[0.01], [10], [64], [16], [2]],  # 0.01, 10, 64, 2
    "LSTM_Seq2Seq_Att": [[0.001], [10], [1024], [16], [1]],  # 0.001, 10, 1024 1
    "GRU_Seq2Seq_Att": [[0.001], [15], [256], [16], [1]],  # 0.001, 15, 256, 1
    "Transformer": [[0.01], [7], [32], [16], [2]],  # 0.01, 7, 32, 2,1
    "Informer": [[0.01], [7], [512], [16], [2]],  # 0.001, 7, 512, 2,1
    "Autoformer": [[0.0001], [7], [512], [16], [2]],  # 0.0001,  7,  512,  2,1
    "FEDformer": [[0.0001], [7], [512], [16], [4]],  # 0.0001, 7, 512, 4,3
    "Dlinear": [[0.2], [10], [100], [16], [0]],  # 0.2 ,10,  -,  -
    "PatchTST": [[0.0001], [10], [512], [16], [3]],  # 0.0001, 10, 512, 3
    "Nlinear": [[0.2], [10], [100], [16], [0]],  # 0.2,  10,  -,  -
    "RandomWalk": [[0.01], [6], [100], [1], [0]]
}
# List of models to be tested
mlist = ["RandomWalk", "FFNN","LSTM", "GRU", "LSTM_Seq2Seq", "GRU_Seq2Seq", "LSTM_Seq2Seq_Att", "GRU_Seq2Seq_Att","Transformer", "Informer", "Autoformer", "FEDformer", "Dlinear", "Nlinear", "PatchTST"]

# Initialize the results dictionary with high initial error values
results_dict = {"RandomWalk": [100, 100, 100], "PatchTST": [100, 100, 100], "FEDformer": [100, 100, 100],
                "Dlinear": [100, 100, 100], "Nlinear": [100, 100, 100], "Informer": [100, 100, 100],
                "Autoformer": [100, 100, 100], "Transformer": [100, 100, 100], "FFNN": [100, 100, 100],
                "LSTM": [100, 100, 100], "GRU": [100, 100, 100],
                "GRU_Seq2Seq": [100, 100, 100], "LSTM_Seq2Seq": [100, 100, 100], "LSTM_Seq2Seq_Att": [100, 100, 100],
                "GRU_Seq2Seq_Att": [100, 100, 100], "RandomWalk": [100, 100, 100]}

# Loop over prediction lengths
for pl in [6]:
    for m in mlist:
        figs = "" #stores the best parameters
        print("MODEL", blue(m, 'bold'))

        # Iterate over each combination of hyperparameters for the current model
        for lr in reasonable_params_dict[m][0]:
            for sl in reasonable_params_dict[m][1]:
                for model_dim in reasonable_params_dict[m][2]:
                    for batch_size in reasonable_params_dict[m][3]:
                        for layers in reasonable_params_dict[m][4]:

                            # Set random seed for reproducibility
                            random.seed(fix_seed)
                            torch.manual_seed(fix_seed)
                            np.random.seed(fix_seed)


                            # Configuration class for experiment settings
                            class Configs(object):
                                units_layer1 = model_dim  # for FFNN
                                units_layer2 = model_dim + model_dim  # for FFNN
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
                                itr = 3
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


                            args = Configs()

                            # Check GPU availability
                            args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
                            if args.use_gpu and args.use_multi_gpu:
                                args.devices = args.devices.replace(' ', '')
                                device_ids = args.devices.split(',')
                                args.device_ids = [int(id_) for id_ in device_ids]
                                args.gpu = args.device_ids[0]

                            # Set up experiment
                            Exp = Exp_Main
                            if args.is_training:
                                for ii in range(args.itr):
                                    # setting record of experiments
                                    setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
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
                                        args.des, ii)

                                    exp = Exp(args)  # set experiments
                                    # print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
                                    exp.train(setting)

                                    print('testing : {}'.format(setting))
                                    realmae, mae, smape = exp.test(setting)

                                    # Update best results if current results are better
                                    if type(results_dict[m][2]) == int:

                                        if results_dict[m][2] > np.mean(smape):
                                            results_dict[m] = [realmae, mae, smape]
                                            figs = str(lr) + " " + str(sl) + " " + str(model_dim) + " " + str(layers)
                                    else:
                                        if np.mean(results_dict[m][2]) > np.mean(smape):
                                            results_dict[m] = [realmae, mae, smape]
                                            figs = str(lr) + " " + str(sl) + " " + str(model_dim) + " " + str(layers)


                                    torch.cuda.empty_cache()

        # Print the best results for the current prediction length and model
        #realMAE: MAEs for original data
        #normMAE: MAEs for normalized data
        #realSMAPE: sMAPEs for original data
        print("---------------Best (realMAE, normMAE, realSMAPE) for " + str(
            pl) + " pred length " + m + ": " + figs + "--------------------------")
        for horiz in range(pl):
            print(str(horiz) + ": ", results_dict[m][0][horiz], results_dict[m][1][horiz], results_dict[m][2][horiz])
        print("Average realMAE: ", sum([results_dict[m][0][horiz] for horiz in range(pl)]) / pl)
        print("Average normMAE: ", sum([results_dict[m][1][horiz] for horiz in range(pl)]) / pl)
        print("Average realSMAPE: ", sum([results_dict[m][2][horiz] for horiz in range(pl)]) / pl)