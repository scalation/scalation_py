from util.data_loading import data_provider
from modeling.neuralmodeling.models import Transformer, Informer, Autoformer, FEDformer, Crossformer, PatchTST, GPT4TS, LSTM_FFNN, Linear, DLinear, NLinear, FreTS, iTransformer, RLinear, LSTM_Seq2Seq, \
    FFNN
from modeling.neuralmodeling.models import Koopa
from util.tools import EarlyStopping, adjust_learning_rate
import torch
import torch.nn as nn
from torch import optim
import random
import os
import time
import warnings
import numpy as np
from ray.air import session
import matplotlib.pyplot as plt
from util.QoF import get_metrics
import pandas as pd
from util.tools import save_results
import datetime
warnings.filterwarnings('ignore')

class Experiment(object):
    def __init__(self, args):
        if args is not None:
            self.args = args
            self.device = self._acquire_device()
            self.model = self._build_model()

    def _build_model(self):
        print(self.args)
        fix_seed = self.args['random_seed']
        random.seed(fix_seed)
        torch.manual_seed(fix_seed)
        np.random.seed(fix_seed)
        model_dict = {
            'Transformer': Transformer,
            'Informer': Informer,
            'Autoformer': Autoformer,
            'FEDformer': FEDformer,
            'Crossformer': Crossformer,
            'Transformer': Transformer,
            'PatchTST': PatchTST,
            'GPT4TS': GPT4TS,
            'LSTM_FFNN': LSTM_FFNN,
            'Linear': Linear,
            'DLinear': DLinear,
            'NLinear': NLinear,
            'FreTS': FreTS,
            'iTransformer': iTransformer,
            'RLinear': RLinear,
            'LSTM_Seq2Seq': LSTM_Seq2Seq,
            'Koopa': Koopa,
            'FFNN': FFNN
        }
        model = model_dict[self.args['model_name']].Model(self.args).float()

        print(model)

        if self.args['use_multi_gpu'] and self.args['use_gpu']:
            model = nn.DataParallel(model, device_ids=self.args['device_ids'])

        return model.to(self.device)

    def _acquire_device(self):
        if self.args['use_gpu']:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args['gpu']) if not self.args['use_multi_gpu'] else self.args['devices']
            device = torch.device('cuda:{}'.format(self.args['gpu']))
            print('Use GPU: cuda:{}'.format(self.args['gpu']))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self, flag, visualize=False, startIndex=0, validation='In Sample', use_original_data=False):
        data_set, data_loader = data_provider(self.args, flag, visualize=visualize, startIndex=startIndex,
                                              validation=validation, use_original_data=use_original_data)
        self.Data = data_set
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args["learning_rate"])
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def _get_results(self, preds, trues, preds_original, trues_original):
        results = pd.DataFrame(columns=["mse_normalized", "mae_normalized", "smape_original"])
        target_int = self.Data.columns.get_loc(self.args['target']) - 1
        print(f"\n\033[1mSurvey Results: {self.args['validation']}\033[0m")
        print(f"\033[1mModel: \033[0m {self.args['model_name']} \033[1mDataset: \033[0m {self.args['dataset']}")
        print(f"total_trainable_params: {self.total_trainable_params}")
        print(f"total_training_time: {self.total_training_time}")
        print(f"total_inference_time: {self.total_inference_time}\n")
        N, mse, mae, smape = get_metrics(preds, trues)
        mse = ('%.3f' % mse)
        mae = ('%.3f' % mae)
        smape = ('%.3f' % smape)

        results_path = os.path.join(os.getcwd(), self.args['dataset'], 'results')
        if not os.path.exists(results_path):
            os.makedirs(results_path)

        print(f"\033[1mN: Test Data, H: Horizon, MSE Normalized, MAE Normalized, sMAPE Original - Per Horizon\033[0m")
        for i in self.args['horizons']:
            i = i - 1
            ignore = None
            if i > 0:
                ignore = -i
            if self.args['in_sample']:
                test_start = (self.args['seq_len']) + i
                num_test = self.Data.num_train - self.args['seq_len'] - i
                N, mse_normalized, mae_normalized, smape_normalized = get_metrics(
                    np.expand_dims(np.expand_dims(preds[0:ignore, i, target_int], axis=1), axis=1),
                    np.expand_dims(np.expand_dims(trues[0:ignore, i, target_int], axis=1), axis=1))

                N, mse_original, mae_original, smape_original = get_metrics(
                    np.expand_dims(np.expand_dims(preds_original[0:ignore, i, target_int], axis=1), axis=1),
                    np.expand_dims(np.expand_dims(trues_original[0:ignore, i, target_int], axis=1), axis=1))
                mse_normalized = mse_normalized
                mae_normalized = mae_normalized
                smape_original = smape_original

                values = [mse_normalized, mae_normalized, smape_original]
                new_row = pd.DataFrame([values], columns=results.columns)
                results = pd.concat([results, new_row], ignore_index=True)

            else:


                N, mse_normalized, mae_normalized, smape_normalized = get_metrics(
                    np.expand_dims(np.expand_dims(preds[0:ignore, i, target_int], axis=1), axis=1),
                    np.expand_dims(np.expand_dims(trues[0:ignore, i, target_int], axis=1), axis=1))

                N, mse_original, mae_original, smape_original = get_metrics(
                    np.expand_dims(np.expand_dims(preds_original[0:ignore, i, target_int], axis=1), axis=1),
                    np.expand_dims(np.expand_dims(trues_original[0:ignore, i, target_int], axis=1), axis=1))

                mse_normalized = mse_normalized
                mae_normalized = mae_normalized
                smape_original = smape_original
                values = [mse_normalized, mae_normalized, smape_original]
                new_row = pd.DataFrame([values], columns=results.columns)
                results = pd.concat([results, new_row], ignore_index=True)
        results.loc[len(results)] = results.mean()
        results = results.reset_index(drop=True)
        results.index = self.args['horizons'] + ['Avg.']
        results = results.astype(float).round(3)
        print(results)
        today = str(datetime.datetime.today().strftime('%Y-%m-%d'))
        file_path = os.path.join(results_path, 'Results ' + today + '.csv')
        save_results(self, file_path, results)

        plots_path = os.path.join(os.getcwd(), self.args['dataset'], 'plots', self.args['validation'], self.args['model_name'], str(self.args['dataset']), f"{self.args['seq_len']}_{self.args['pred_len']}_{self.args['random_seed']}")
        if not os.path.exists(plots_path):
            os.makedirs(plots_path)

        for i in self.args['horizons']:
            file_path = os.path.join(plots_path,
                                     str(self.args["validation"]) + '_' + str(self.args["model_name"]) + '_' + str(
                                         self.args["seq_len"]) + '_' + str(self.args["pred_len"]) + '_' + str(
                                         i) + '_' + str(self.args["random_seed"]) + '.png')

            i = i - 1
            ignore = None
            if i > 0:
                ignore = -i
            if self.args['in_sample']:
                test_start = (self.args['seq_len']) + i
                num_test = self.Data.num_train - self.args['seq_len'] - i
                test_data, test_loader = self._get_data(flag='test', visualize=True, startIndex=0,
                                                        validation=self.args['validation'],
                                                        use_original_data=self.args['use_original_data'])
                idx = np.arange(test_start, test_start + num_test, 1)
                if i == 0:
                    print('idx: ', idx.shape, test_start, test_start + num_test)
                plt.plot(idx, preds_original[0:ignore, i, target_int], color='red',
                         label=str('Forecasts h = ' + str(i + 1)), marker='o', markersize=1, linewidth=0.5)
                plt.legend()
                plt.savefig(file_path, bbox_inches='tight')

            else:
                test_data, test_loader = self._get_data(flag='test', visualize=True, startIndex=0,
                                                        validation=self.args['validation'],
                                                        use_original_data=self.args['use_original_data'])
                idx = np.arange(self.Data.test_start, self.Data.test_start + self.Data.num_test, 1)
                plt.plot(idx, preds_original[
                              self.args['pred_len'] - i - 1:self.args['pred_len'] - 1 - i + self.Data.num_test, i,
                              target_int], color='red', label=str('Forecasts h = ' + str(i + 1)), marker='o',
                         markersize=1, linewidth=0.5)
                plt.legend()
                plt.savefig(file_path, bbox_inches='tight')

        return trues_original, preds_original


    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        target_int = self.Data.columns.get_loc(self.args['target']) - 1
        with torch.no_grad():
            for i, (batch_x, batch_x_original, batch_y, batch_y_original, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args["pred_len"]:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args["label_len"], :], dec_inp], dim=1).float().to(self.device)
                
                if self.args["use_amp"]:
                    with torch.cuda.amp.autocast():
                        if self.args["output_attention"]:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, False)
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, False)
                else:
                    if self.args["output_attention"]:
                        outputs, attention_weights = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, False)
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, False)
                            
                f_dim = -1 if self.args["features"] == 'MS' else 0
                outputs = outputs[:, -self.args["pred_len"]:, f_dim:]
                batch_y = batch_y[:, -self.args["pred_len"]:, f_dim:].to(self.device)
                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                loss = criterion(pred, true)
                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting = None):
        if setting is None:
            setting = 'id{}_m{}_d{}_f{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_dff{}_do{}_lr{}'.format(self.args["model_id"], self.args["model_name"], self.args["data"], self.args["features"], self.args["seq_len"], self.args["label_len"], self.args["pred_len"], self.args["d_model"], self.args["n_heads"], self.args["e_layers"], self.args["d_layers"], self.args["d_ff"], self.args["dropout"], self.args["learning_rate"])
        else:
            setting = setting

        train_data, train_loader = self._get_data(flag='train', visualize = False, startIndex = 0)
        test_data, test_loader = self._get_data(flag='train', visualize = False, startIndex = 0)
        
        if self.args["in_sample"]:
            train_data, train_loader = self._get_data(flag='train', visualize = False, startIndex = 0, validation = self.args['validation'], use_original_data = self.args['use_original_data'])
            test_data, test_loader = self._get_data(flag='train', visualize = False, startIndex = 0, validation = self.args['validation'], use_original_data = self.args['use_original_data'])
        else:
            train_data, train_loader = self._get_data(flag='train', visualize = False, startIndex = 0, validation = self.args['validation'], use_original_data = self.args['use_original_data'])
            test_data, test_loader = self._get_data(flag='test', visualize = False, startIndex = 0, validation = self.args['validation'], use_original_data = self.args['use_original_data'])
        
        print('train test lengths: ', len(train_data), len(test_data))
                    
        path = os.path.join(self.args["checkpoints"], setting)
        if not os.path.exists(path):
            os.makedirs(path)

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args["patience"], verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args["use_amp"]:
            scaler = torch.cuda.amp.GradScaler()

        trainingEpoch_loss = []
        validationEpoch_loss = []
        testEpoch_loss = []
        time_now = time.time()
        training_start_time = time.time()
        target_int = self.Data.columns.get_loc(self.args['target']) - 1
        for epoch in range(self.args["train_epochs"]):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_x_original, batch_y, batch_y_original, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                dec_inp = torch.zeros_like(batch_y[:, -self.args["pred_len"]:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args["label_len"], :], dec_inp], dim=1).float().to(self.device)

                if self.args["use_amp"]:
                    with torch.cuda.amp.autocast():
                        if self.args["output_attention"]:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, self.args["use_tf"])[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, self.args["use_tf"])

                        f_dim = -1 if self.args["features"] == 'MS' else 0
                        outputs = outputs[:, -self.args["pred_len"]:, f_dim:]
                        batch_y = batch_y[:, -self.args["pred_len"]:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args["output_attention"]:
                        outputs, attention_weights = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, self.args["use_tf"])
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, self.args["use_tf"])
                    f_dim = -1 if self.args["features"] == 'MS' else 0
                    outputs = outputs[:, -self.args["pred_len"]:, f_dim:]
                    batch_y = batch_y[:, -self.args["pred_len"]:, f_dim:].to(self.device)
                    loss = criterion(outputs[:,:,target_int], batch_y[:,:,target_int])
                    train_loss.append(loss.item())

                if self.args["use_amp"]:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(train_data, train_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)
            trainingEpoch_loss.append(train_loss)
            validationEpoch_loss.append(vali_loss)
            testEpoch_loss.append(test_loss)
            
            session.report({"vali_loss": vali_loss})

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        self.ep = epoch        
        self.total_training_time = time.time() - training_start_time
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path, map_location='cuda:0'))
        return self.model

    def test(self, setting = None):
        if setting is None:
            setting = 'id{}_m{}_d{}_f{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_dff{}_do{}_lr{}'.format(self.args["model_id"], self.args["model_name"], self.args["data"], self.args["features"], self.args["seq_len"], self.args["label_len"], self.args["pred_len"], self.args["d_model"], self.args["n_heads"], self.args["e_layers"], self.args["d_layers"], self.args["d_ff"], self.args["dropout"], self.args["learning_rate"])
        else:
            setting = setting

        if self.args["in_sample"]:
            test_data, test_loader = self._get_data(flag='train', visualize = False, startIndex = 0, validation = self.args['validation'], use_original_data = self.args['use_original_data'])
        else:
            test_data, test_loader = self._get_data(flag='test', visualize = False, startIndex = 0, validation = self.args['validation'], use_original_data = self.args['use_original_data'])
        preds = None
        trues = None
        preds_original = None
        trues_original = None

        if self.args['use_original_data']: 
            mean_X_train, std_X_train = 0, 1
        else:
            mean_X_train, std_X_train = self.Data.scaler_train.mean_, self.Data.scaler_train.scale_
        
        inference_start_time = time.time()
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_x_original, batch_y, batch_y_original, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args["pred_len"]:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args["label_len"], :], dec_inp], dim=1).float().to(self.device)
                
                if self.args["use_amp"]:
                    with torch.cuda.amp.autocast():
                        if self.args["output_attention"]:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, False)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, False)
                else:
                    if self.args["output_attention"]:
                        outputs, attention_weights = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, False)
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, False)

                f_dim = -1 if self.args["features"] == 'MS' else 0
                outputs = outputs[:, -self.args["pred_len"]:, f_dim:]
                batch_y = batch_y[:, -self.args["pred_len"]:, f_dim:].to(self.device)

                batch_x_original = batch_x_original
                batch_y_original = batch_y_original
                outputs_original = outputs
                
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if preds is None and trues is None:
                    preds = outputs[:,-self.args["pred_len"]:,:]
                    trues = batch_y[:,-self.args["pred_len"]:,:]
                else:
                    preds = np.concatenate((preds, outputs[:,-self.args["pred_len"]:,:]), axis = 0)
                    trues = np.concatenate((trues, batch_y[:,-self.args["pred_len"]:,:]), axis = 0)
                
                outputs_original = outputs_original.cpu() * std_X_train + mean_X_train
                batch_x_original = batch_x_original.cpu()
                batch_y_original = batch_y_original.cpu()
                outputs_original = outputs_original.detach().cpu().numpy()
                batch_y_original = batch_y_original.detach().cpu().numpy()
                if preds_original is None and trues_original is None:
                    preds_original = outputs_original[:,-self.args["pred_len"]:,:]
                    trues_original = batch_y_original[:,-self.args["pred_len"]:,:]
                else:
                    preds_original = np.concatenate((preds_original, outputs_original[:,-self.args["pred_len"]:,:]), axis = 0)
                    trues_original = np.concatenate((trues_original, batch_y_original[:,-self.args["pred_len"]:,:]), axis = 0)
        self.total_inference_time = time.time() - inference_start_time
        preds = np.array(preds)
        trues = np.array(trues)
        preds_original = np.array(preds_original)
        trues_original = np.array(trues_original)
        self.total_params = sum(p.numel() for p in self.model.parameters())
        self.total_trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return self._get_results(preds, trues, preds_original, trues_original)
    
    def rolling_validation(self, setting = None):
        if setting is None:
            setting = 'id{}_m{}_d{}_f{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_dff{}_do{}_lr{}'.format(self.args["model_id"], self.args["model_name"], self.args["data"], self.args["features"], self.args["seq_len"], self.args["label_len"], self.args["pred_len"], self.args["d_model"], self.args["n_heads"], self.args["e_layers"], self.args["d_layers"], self.args["d_ff"], self.args["dropout"], self.args["learning_rate"])
        else:
            setting = setting
        preds = None
        trues = None
        preds_original = None
        trues_original = None
        test_data, test_loader = self._get_data(flag='test', visualize = False, startIndex = 0, validation = self.args['validation'], use_original_data = self.args['use_original_data'])
        test_data_size = len(test_data)
        target_int = self.Data.columns.get_loc(self.args['target']) - 1
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        for ii in range(test_data_size):
            self.model.train()
            if ii > self.args['pred_len'] - 1:
                startIndexTrain = ii - (self.args['pred_len'] - 1)
                startIndexTest = ii
            else:
                startIndexTest = ii
                startIndexTrain = 0

            train_data, train_loader = self._get_data(flag='train', visualize = False, startIndex = startIndexTrain, validation = self.args['validation'], use_original_data = self.args['use_original_data'])
            test_data, test_loader = self._get_data(flag='test', visualize = False, startIndex = startIndexTest, validation = self.args['validation'], use_original_data = self.args['use_original_data'])
            checkpoints_path = self.args["checkpoints"]
            path = os.path.join(checkpoints_path, setting)
            if not os.path.exists(path):
                os.makedirs(path)

            train_steps = len(train_loader)
            early_stopping = EarlyStopping(patience=self.args["patience"], verbose=True)

            if self.args["use_amp"]:
                scaler = torch.cuda.amp.GradScaler()

            trainingEpoch_loss = []
            validationEpoch_loss = []
            testEpoch_loss = []
            time_now = time.time()
            training_start_time = time.time()
            if ii%1 == 0:
                for epoch in range(self.args["train_epochs"]):
                    iter_count = 0
                    train_loss = []
                    self.model.train()
                    epoch_time = time.time()
                    for i, (batch_x, batch_x_original, batch_y, batch_y_original, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                        iter_count += 1
                        model_optim.zero_grad()
                        batch_x = batch_x.float().to(self.device)

                        batch_y = batch_y.float().to(self.device)
                        batch_x_mark = batch_x_mark.float().to(self.device)
                        batch_y_mark = batch_y_mark.float().to(self.device)

                        dec_inp = torch.zeros_like(batch_y[:, -self.args["pred_len"]:, :]).float()
                        dec_inp = torch.cat([batch_y[:, :self.args["label_len"], :], dec_inp], dim=1).float().to(self.device)

                        if self.args["use_amp"]:
                            with torch.cuda.amp.autocast():
                                if self.args["output_attention"]:
                                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, self.args["use_tf"])[0]
                                else:
                                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, self.args["use_tf"])

                                f_dim = -1 if self.args["features"] == 'MS' else 0
                                outputs = outputs[:, -self.args["pred_len"]:, f_dim:]
                                batch_y = batch_y[:, -self.args["pred_len"]:, f_dim:].to(self.device)
                                loss = criterion(outputs, batch_y)
                                train_loss.append(loss.item())
                        else:
                            if self.args["output_attention"]:
                                outputs, attention_weights = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, self.args["use_tf"])
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, self.args["use_tf"])
                            f_dim = -1 if self.args["features"] == 'MS' else 0
                            outputs = outputs[:, -self.args["pred_len"]:, f_dim:]
                            batch_y = batch_y[:, -self.args["pred_len"]:, f_dim:].to(self.device)
                            loss = criterion(outputs, batch_y)
                            train_loss.append(loss.item())

                        if self.args["use_amp"]:
                            scaler.scale(loss).backward()
                            scaler.step(model_optim)
                            scaler.update()
                        else:
                            loss.backward()
                            model_optim.step()

                    print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
                    train_loss = np.average(train_loss)
                    vali_loss = self.vali(train_data, train_loader, criterion)
                    test_loss = self.vali(test_data, test_loader, criterion)
                    trainingEpoch_loss.append(train_loss)
                    validationEpoch_loss.append(vali_loss)
                    testEpoch_loss.append(test_loss)
                    
                    session.report({"vali_loss": vali_loss})

                    print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                        epoch + 1, train_steps, train_loss, vali_loss, test_loss))
                    early_stopping(vali_loss, self.model, path)
                    if early_stopping.early_stop:
                        print("Early stopping")
                        break

                    if self.args["lradj"] != 'TST':
                        adjust_learning_rate(model_optim, epoch + 1, self.args)
                    else:
                        print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))
                self.ep = epoch        
                self.total_training_time = time.time() - training_start_time

            test_data, test_loader = self._get_data(flag='test', visualize = False, startIndex = startIndexTest, validation = self.args['validation'], use_original_data = self.args['use_original_data'])

            mean_X_train, std_X_train = self.Data.scaler_train.mean_, self.Data.scaler_train.scale_
            inference_start_time = time.time()
            self.model.eval()
            with torch.no_grad():
                for i, (batch_x, batch_x_original, batch_y, batch_y_original, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)

                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                    dec_inp = torch.zeros_like(batch_y[:, -self.args["pred_len"]:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args["label_len"], :], dec_inp], dim=1).float().to(self.device)

                    if self.args["use_amp"]:
                        with torch.cuda.amp.autocast():
                            if self.args["output_attention"]:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, False)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, False)
                    else:
                        if self.args["output_attention"]:
                            outputs, attention_weights = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, False)
                            attn_weights = attention_weights
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, False)

                    f_dim = -1 if self.args["features"] == 'MS' else 0
                    
                    outputs = outputs[:, -self.args["pred_len"]:, f_dim:]
                    batch_y = batch_y[:, -self.args["pred_len"]:, f_dim:].to(self.device)
                    
                    batch_x_original = batch_x_original
                    batch_y_original = batch_y_original
                    outputs_original = outputs

                    outputs = outputs.detach().cpu().numpy()
                    batch_y = batch_y.detach().cpu().numpy()
                    
                    if preds is None and trues is None:
                        preds = np.expand_dims(outputs[0,-self.args["pred_len"]:,:], axis=0) 
                        trues = np.expand_dims(batch_y[0,-self.args["pred_len"]:,:], axis=0)  
                    else:
                        preds = np.concatenate((preds, np.expand_dims(outputs[0,-self.args["pred_len"]:,:], axis=0)), axis = 0)
                        trues = np.concatenate((trues, np.expand_dims(batch_y[0,-self.args["pred_len"]:,:], axis=0)) , axis = 0)

                    outputs_original = outputs_original.cpu() * std_X_train + mean_X_train
                    batch_x_original = batch_x_original.cpu()
                    batch_y_original = batch_y_original.cpu()
                    outputs_original = outputs_original.detach().cpu().numpy()
                    batch_y_original = batch_y_original.detach().cpu().numpy()
                    if preds_original is None and trues_original is None:
                        preds_original = np.expand_dims(outputs_original[0,-self.args["pred_len"]:,:], axis=0)
                        trues_original = np.expand_dims(batch_y_original[0,-self.args["pred_len"]:,:], axis=0)
                    else:
                        preds_original = np.concatenate((preds_original, np.expand_dims(outputs_original[0,-self.args["pred_len"]:,:], axis=0)), axis = 0)
                        trues_original = np.concatenate((trues_original, np.expand_dims(batch_y_original[0,-self.args["pred_len"]:,:], axis=0)), axis = 0)
                    break
        test_data, test_loader = self._get_data(flag='test', visualize = False, startIndex = 0, validation = self.args['validation'], use_original_data = self.args['use_original_data'])
        self.total_inference_time = time.time() - inference_start_time
        preds = np.array(preds)
        trues = np.array(trues)
        preds_original = np.array(preds_original)
        trues_original = np.array(trues_original)
        self.total_params = sum(p.numel() for p in self.model.parameters())
        self.total_trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return self._get_results(preds, trues, preds_original, trues_original)
  