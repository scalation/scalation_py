#Import required modules
from data.data_processing import data_processing
from data.data_splitting import train_test_split, make_input_output_sequences, shift_sequence
from data.data_transform import data_transform_std
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
from utils.metrics import SMAPE, MAE

#Import required libraries
import torch
import numpy as np
import random
from simple_colors import *
import pandas as pd
import torch.utils.data as data_utils
from functools import reduce
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # Set device to GPU if available, otherwise use CPU

# Define a configuration class to handle model parameters
class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)

# Define experiment parameters
horizons = 4
test_ratio = 0.795 #defined to get the interval date range
file_name = 'datasets/covid_weekly_cdc.csv'
features = ['date','new_deaths','icu_patients','hosp_patients','reproduction_rate'] #features tuned through grid search
epochs = 20
patience = 3
in_dim = len(features)-1
tf = False #teacher forcing
random_seed = 2021 #Seed for reproducibility

# Specify the models to train
# models = ["RandomWalk", "FFNN","LSTM", "GRU", "LSTM_Seq2Seq", "GRU_Seq2Seq", "LSTM_Seq2Seq_Att", "GRU_Seq2Seq_Att","Transformer",
# #          "Informer", "Autoformer", "FEDformer", "Dlinear", "Nlinear", "PatchTST"]
models = ["Dlinear"]


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
    "RandomWalk":RandomWalk,
}

# Define hyperparameter values for each model
#the following parameters are for reproducibility only
reasonable_params_dict = {
    "FFNN": [[0.001], [6], [256], [4], [0]],
    "LSTM": [[0.001], [6], [64], [16], [1]],
    "GRU": [[0.001], [6], [512],[8], [3]],
    "LSTM_Seq2Seq": [[0.001], [6], [32], [8], [1]],
    "GRU_Seq2Seq": [[0.001], [6], [8], [4], [3]],
    "LSTM_Seq2Seq_Att": [[0.001], [5], [32], [8], [3]],
    "GRU_Seq2Seq_Att": [[0.001], [6], [256], [16], [2]],
    "Transformer": [[0.0001], [8], [256], [8], [2]],
    "Informer": [[0.0001], [8], [256], [8], [2]],
    "Autoformer": [[0.001], [8], [512], [8], [2]],
    "FEDformer": [[0.001], [8], [256], [8], [2]],
    "Dlinear": [[0.02], [7], [100], [16], [0]],
    "PatchTST": [[0.0001], [8], [512], [32], [4]],
    "Nlinear": [[0.0001], [4], [100], [4], [0]],
    "RandomWalk": [[0.001], [4], [100], [1], [0]]
}

# Iterate through each model and their hyperparameter combinations
for m in models:
    print(blue(m, 'bold'))
    for lr in reasonable_params_dict[m][0]:
        for sl in reasonable_params_dict[m][1]:
            for model_dim in reasonable_params_dict[m][2]:
                for batch_size in reasonable_params_dict[m][3]:
                    for layers in reasonable_params_dict[m][4]:
                        print(layers, lr, sl, model_dim, batch_size)

                        # Prepare forecast matrix to store results and set random seeds
                        forecast_matrix = pd.DataFrame(columns=['date'])
                        random.seed(random_seed)
                        np.random.seed(random_seed)
                        torch.manual_seed(random_seed)

                        # Process and transform the data
                        data, observed = data_processing(file_name, features)
                        scalers, df = data_transform_std(data, test_ratio)
                        obs_scalers, observed_df = data_transform_std(observed, 1.0)
                        x, y = make_input_output_sequences(data.values, sl, horizons, True)
                        x_train, x_test, y_train, y_test = train_test_split(x, y, test_ratio)

                        # Define model configuration parameters
                        config_classes = {
                            m: {
                                'ab': 0,
                                'modes': 64,
                                'mode_select': 'random',
                                'version': 'Fourier',
                                'moving_avg': [12, 24],
                                'L': 3,
                                'base': 'legendre',
                                'cross_activation': 'tanh',
                                'seq_len': sl,
                                'label_len': horizons,
                                'pred_len': horizons,
                                'output_attention': True,
                                'enc_in': in_dim,
                                'individual': True,
                                'dec_in': in_dim,
                                'd_model': model_dim,
                                'embed': 'timeF',
                                'dropout': 0.05,
                                'freq': 'd',
                                'factor': 1,
                                'n_heads': 8,
                                'd_ff': 2048,
                                'e_layers': layers,
                                'd_layers': layers-1,
                                'distil': False,
                                'c_out': in_dim,
                                'activation': 'gelu',
                                'wavelet': 0,
                                'units_layer1': model_dim,
                                'units_layer2': model_dim,
                                'batch_size': batch_size,
                                'task_name': "long_term_forecast"

                            }
                        }
                        # Initialize the model
                        model_class = model_dict[m]
                        config_dict = config_classes.get(m, {})
                        config = Config(**config_dict)
                        model = model_class(config).to(device)

                        ListPred = []
                        ListTrue = []
                        for w in range(x_test.shape[0]): # Iterate through test data windows
                            n_epochs_stop = patience
                            epochs_no_improve = 0
                            early_stop = False
                            min_val_loss = np.inf
                            results = pd.DataFrame(columns=['date'])
                            y_test_dates = pd.DataFrame(y_test[0, :, 0])
                            x_train_tensor = torch.from_numpy(np.array(x_train[:, :, 1:], dtype=np.float32)).float().to(device)
                            y_train_tensor = torch.from_numpy(np.array(y_train[:, :, 1:], dtype=np.float32)).float().to(device)
                            x_test_tensor = torch.from_numpy(np.array(x_test[:, :, 1:], dtype=np.float32)).float().to(device)
                            y_test_tensor = torch.from_numpy(np.array(y_test[:, :, 1:], dtype=np.float32)).float().to(device)
                            results['date'] = y_test_dates
                            loss_fn = torch.nn.MSELoss()
                            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

                            # Train the model
                            train = data_utils.TensorDataset(x_train_tensor, y_train_tensor)
                            train_loader = data_utils.DataLoader(train, batch_size=batch_size, shuffle=False)
                            for epoch in range(epochs):
                                train_losses = []
                                model.train()
                                for x_t, y_t in train_loader:
                                    optimizer.zero_grad()
                                    y_pred = model(x_t, y_t, tf, False)
                                    loss = loss_fn(y_pred[:, :, 0], y_t[:, :, 0])
                                    loss.backward()
                                    optimizer.step()
                                    train_losses.append(loss.item())
                                train_loss = np.average(train_losses)
                                if train_loss < min_val_loss:
                                    epochs_no_improve = 0
                                    min_val_loss = train_loss
                                else:
                                    epochs_no_improve += 1
                                if epochs_no_improve == n_epochs_stop:
                                    break
                            with torch.no_grad(): #Evaluate on test data
                                model.eval()
                                y_test_pred = model(x_test_tensor, y_test_tensor, False, False)

                                # Append true and predicted values to lists
                                ListTrue.append(
                                    scalers['scaler_new_deaths'].inverse_transform(y_test_tensor[:, :, 0].cpu().numpy())[0, :])
                                ListPred.append(
                                    scalers['scaler_new_deaths'].inverse_transform(y_test_pred[:, :, 0].cpu().numpy())[0, :])

                                results['window' + str(w)] = y_test_pred[0, :, 0].cpu().flatten().numpy() # Add predictions to results DataFrame

                            # Shift sequence for the next window
                            x_train, x_test, y_train, y_test = shift_sequence(x_train, y_train, x_test, y_test, 1, True)

                            # Merge forecast results into the forecast matrix w.r.t date
                            if (w == 0):
                                forecast_matrix = pd.merge(forecast_matrix, results, on=['date'], how='right')
                            else:
                                forecast_matrix = pd.merge(forecast_matrix, results, on=['date'], how='outer')

                        ListTrue = np.array(ListTrue)
                        ListPred = np.array(ListPred)

                        # Process forecast matrix and observed data for evaluation
                        date = pd.DataFrame(forecast_matrix['date'])
                        forecast_matrix.set_index('date', inplace=True)
                        scaler = scalers['scaler_new_deaths']
                        # forecast_matrix=scaler.inverse_transform(forecast_matrix)
                        forecast_matrix = forecast_matrix.to_numpy()
                        forecast_matrix = pd.DataFrame(forecast_matrix)
                        forecast_matrix['date'] = date
                        forecast_matrix.set_index('date', inplace=True)

                        # Create DataFrame for weekly predictions with horizons as columns
                        start = 0
                        end = horizons
                        weekly_predictions = []
                        window = 0
                        for i in range(len(forecast_matrix.columns) - (horizons-1)):
                            forecast_matrix_temp = forecast_matrix.iloc[start:end]
                            date = forecast_matrix_temp.tail(1).index.item()
                            last_row = forecast_matrix_temp.tail(1)
                            weekly_predictions.append([date, last_row[window][0], last_row[window + 1][0],
                                                       last_row[window + 2][0], last_row[window + 3][0]])
                            window = window + 1
                            start = end
                            end = end + 1
                        weekly_predictions = pd.DataFrame(weekly_predictions,
                                                          columns=['date', str('h4_' + m), str('h3_' + m), str('h2_' + m),
                                                                   str('h1_' + m)])
                        # Prepare observed data for merging
                        observed_df.reset_index(inplace=True, drop=True)
                        observed_df['date'] = pd.to_datetime(observed_df['date'])
                        weekly_predictions['date'] = pd.to_datetime(weekly_predictions['date'])
                        matrix = reduce(lambda x, y: pd.merge(x, y, on='date'), [observed_df, weekly_predictions])

                        #Remove dates that are missing in CDC data
                        dates_to_remove = ['2021-01-23', '2021-08-14', '2021-09-25', '2021-10-02', '2021-11-13', '2021-12-04',
                                           '2022-01-15', '2022-05-07']
                        matrix.set_index('date', inplace=True)
                        matrix = matrix[~matrix.index.isin(dates_to_remove)]

                        #specify start date and end date for the interval to compare with the MIT_LCP model
                        start_date = pd.to_datetime('2020-12-05')
                        end_date = pd.to_datetime('2022-06-04')
                        matrix = matrix.loc[start_date:end_date]
                        matrix.reset_index(inplace=True, drop=True)
                        metrics = np.empty([4, 3]) # Initialize array for metrics

                        # Calculate evaluation metrics for each horizon
                        for idx, j in enumerate(['h1', 'h2', 'h3', 'h4']):
                            norm_mae_metric = ('%.3f' % MAE(matrix['new_deaths'], matrix[j + '_' + str(m)]))
                            smape_metric = ('%.3f' % SMAPE(ListTrue[:, idx], ListPred[:, idx]))
                            mae_metric = ('%.3f' % MAE(ListTrue[:, idx], ListPred[:, idx]))
                            print(str(idx) + ' norm_mae: ' +
                                  str(norm_mae_metric) + ' smape: ' + str(smape_metric) + ' mae: ' + str(mae_metric))
                            metrics[idx, :] = (norm_mae_metric, smape_metric, mae_metric)
                        print(blue("Avgs: ", 'bold') +
                              blue(" norm_mae: ", 'bold') + str(('%.3f' % np.average(metrics[:, 0]))) + blue(" smape: ", 'bold') +
                              red(str(('%.3f' % np.average(metrics[:, 1]))), 'bold') + blue(" mae: ", 'bold')  + str(
                            ('%.3f' % np.average(metrics[:, 2]))))