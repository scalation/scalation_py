from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import PatchTST
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric

from models.PatchTST import PatchTST

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler 

import os
import time

import warnings

import numpy as np

import pandas as pdgf
import matplotlib.pyplot as plt
# %matplotlib inline

warnings.filterwarnings('ignore')

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            # 'Autoformer': Autoformer,
            # 'Transformer': Transformer,
            # 'Informer': Informer,
            # 'DLinear': DLinear,
            # 'NLinear': NLinear,
            # 'Linear': Linear,
            'PatchTST': PatchTST,
        }
    
        # model = model_dict[self.args.model].Model(self.args).float()
        model = model_dict[self.args.model](self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion
    
    def differencing(self, data):
        
        if self.args.difforder == 'First':
            diff_df = data[:, 1:, :] - data[:, :-1, :]
            
        elif self.args.difforder == 'Second':
            first_diff = data[:, 1:, :] - data[:, :-1, :]
            diff_df = first_diff[:, 1:, :] - first_diff[:, :-1, :] #take difference again of the first difference

        elif self.args.difforder == 'Seasonal':
            diff_df = data[:, self.args.seasonal:, :] - data[:, :-self.args.seasonal, :]
            
        return diff_df
    
    def reverse_differencing(self, first_value, preds):
        
        if self.args.difforder == 'First':
            if not isinstance(first_value, torch.Tensor):
                first_value = torch.tensor(first_value, dtype=preds.dtype, device=preds.device)
            if not isinstance(preds, torch.Tensor):
                preds = torch.tensor(preds, dtype=first_value.dtype, device=first_value.device)
            reconstruct = torch.cat([first_value, preds], dim=1)
            reconstruct = torch.cumsum(reconstruct, dim=1)
            reconstruct = reconstruct[:, 1:, :]
            
        elif self.args.difforder == 'Second':
            if not isinstance(first_value, torch.Tensor):
                first_value = torch.tensor(first_value, dtype=preds.dtype, device=preds.device)
            if not isinstance(preds, torch.Tensor):
                preds = torch.tensor(preds, dtype=first_value.dtype, device=first_value.device)
    
            first_diff_initial = first_value[:, 1:2, :] - first_value[:, 0:1, :]
            first_diff = torch.cat([first_diff_initial, preds], dim=1)
            first_diff_reconstructed = torch.cumsum(first_diff, dim=1)
    
            data_initial = first_value[:, 0:1, :]
            data = torch.cat([data_initial, first_diff_reconstructed], dim=1)
            reconstruct = torch.cumsum(data, dim=1)
            reconstruct = reconstruct[:, 2:, :]
                    
        elif self.args.difforder == 'Seasonal':
            if not isinstance(first_value, torch.Tensor):
                first_value = torch.tensor(first_value, dtype=preds.dtype, device=preds.device)
            if not isinstance(preds, torch.Tensor):
                preds = torch.tensor(preds, dtype=first_value.dtype, device=first_value.device)

            data_initial = first_value[:, 0:self.args.seasonal, :]
            reconstruct = torch.cat([data_initial, preds], dim=1)
            for t in range(self.args.seasonal,reconstruct.size(1)):
                reconstruct[:,t,:] = reconstruct[:,t,:] + reconstruct[:,t-self.args.seasonal,:] 
            reconstruct = reconstruct[:,self.args.seasonal:,:]
        return reconstruct
    

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                if self.args.difference == True:

                    batch_x= self.differencing(batch_x)
                    batch_y= self.differencing(batch_y)
                    batch_x_mark=  self.differencing(batch_x_mark)
                    batch_y_mark= self.differencing(batch_y_mark)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
            
        scheduler = lr_scheduler.OneCycleLR(optimizer = model_optim,
                                            steps_per_epoch = train_steps,
                                            pct_start = self.args.pct_start,
                                            epochs = self.args.train_epochs,
                                            max_lr = self.args.learning_rate)

        max_residual_len = self.args.residual_window
        residual_buffer = None
        residuals = None
        
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                if self.args.difference == True:
                    batch_x = self.differencing(batch_x)
                    batch_y= self.differencing(batch_y)
                    batch_x_mark=  self.differencing(batch_x_mark)
                    batch_y_mark= self.differencing(batch_y_mark)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                            
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                           
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)
                    # print(outputs.shape,batch_y.shape)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    
                        
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                # if (i + 1) % 100 == 0:
                #     print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                #     speed = (time.time() - time_now) / iter_count
                #     left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                #     print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                #     iter_count = 0
                #     time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
                    
                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            # print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

#             print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
#                 epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                # print("Early stopping")
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                pass
                # print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        
        if test:
            #print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        inputx = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                batch_y_fv = batch_y.float().to(self.device)
                
                if self.args.difference == True:

                    batch_x = self.differencing(batch_x)
                    batch_y= self.differencing(batch_y)
                    batch_x_mark=  self.differencing(batch_x_mark)
                    batch_y_mark= self.differencing(batch_y_mark)
                

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                            
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                
                # print(outputs.shape,batch_y.shape)
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                
                if self.args.difference == True:

                    if self.args.difforder == 'First':
                        batch_y_fv = batch_y_fv[:, -self.args.pred_len-1:-1, :].to(self.device)
                        first_batch_y = batch_y_fv[:,:1,:] #[ 4747.]
                        outputs= self.reverse_differencing(first_batch_y,outputs)
                        batch_y= self.reverse_differencing(first_batch_y,batch_y)
                    elif self.args.difforder == 'Second':
                        batch_y_fv = batch_y_fv[:, -self.args.pred_len-2:-1, :].to(self.device)
                        first_two_batch_y = batch_y_fv[:, :2, :] #[ 4961.,  4747.] for target
                        outputs= self.reverse_differencing(first_two_batch_y,outputs)
                        batch_y= self.reverse_differencing(first_two_batch_y,batch_y)
                    elif self.args.difforder == 'Seasonal':
                        batch_y_fv = batch_y_fv[:, -self.args.pred_len-self.args.seasonal:-1, :].to(self.device)
                        seasonal_value = batch_y_fv[:, :self.args.seasonal, :].to(self.device) #[ 4961.,  4747.] if seaosnal = 2
                        outputs= self.reverse_differencing(seasonal_value,outputs)
                        batch_y= self.reverse_differencing(seasonal_value,batch_y)
                
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

#                 if test_data.scale and self.args.inverse:
#                     shape = outputs.shape
#                     outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
#                     batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    # visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        if self.args.test_flop:
            test_params_flop((batch_x.shape[1],batch_x.shape[2]))
            exit()
        preds = np.array(preds)
        trues = np.array(trues)
        inputx = np.array(inputx)
        
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        inputx = np.concatenate(inputx, axis=0)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])
        print('test shape:', preds.shape, trues.shape) 
        

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, smape, mspe, rse= metric(preds, trues)
        print("Prediction length: " + str(self.args.pred_len) +  "\nTransform: " 
              + str(self.args.scale_method))
        print('\nMSE of all: {}, \nMAE of all: {}'.format(mse, round(mae,3)))
        

        # Applying inverse transformation
        if self.args.scale and self.args.inverse:
#         if self.args.difference:
            print("\nINVERSE TRANSFORMED RESULTS:\n")
            
            
            shape_preds = preds.shape  # (batch_size, seq_len, num_features)
            shape_trues = trues.shape  # (batch_size, seq_len, num_features)
            
            preds = preds.reshape(shape_preds[0] * shape_preds[1], -1)  # (batch_size * seq_len, num_features)
            trues = trues.reshape(shape_trues[0] * shape_trues[1], -1) # (batch_size * seq_len, num_features)
            
            preds = test_data.inverse_transform(preds).reshape(shape_preds)
            trues = test_data.inverse_transform(trues).reshape(shape_trues)


            true_data = trues[:, -1:, self.args.target_index]  
            pred_data = preds[:, -1:, self.args.target_index]

            plt.figure(figsize=(5.5, 3.5))
            print('trues --------------------------> ', len(true_data))
            print('preds --------------------------> ', len(pred_data))
            plt.plot(true_data, color='red', label='True Data')
            plt.plot(pred_data, color='blue', label='Predictions')
            plt.xlabel("Time Steps")
            plt.ylabel("ILITOTAL (Feature 5)")
            plt.legend()
            plt.show()

            
            if self.args.scale_method == 'yeo-johnson':
            
                if np.isnan(preds[:,:,1]).any():
                    preds[:,:,1][np.isnan(preds[:,:,1])] = np.nanmean(preds[:,:,1])
                
        
            mae, mse, rmse, smape, mspe, rse= metric(preds, trues)
            print('inverse MSE of all: {}, \ninverse MAE of all: {}'.format(round(mse,3), round(mae,3)))

            #for target variable

            smape_list = []
            mae_list = []
            df_smape = pdgf.DataFrame(columns=[f'horizon_{i+1}' for i in range(self.args.pred_len)])
            df_mae = pdgf.DataFrame(columns=[f'horizon_{i+1}' for i in range(self.args.pred_len)])

            for ti in [self.args.target_index]:
                print("\nTARGET INDEX: ",ti)

                for i in range(len(trues)):
                    for h in range(self.args.pred_len):
                        new_preds = preds[i,h:h+1,:]
                        new_trues = trues[i,h:h+1,:]
                        mae_T, mse_T, rmse_T, smape_T, mspe_T, rse_T = metric(new_preds[:,ti], new_trues[:,ti])
                        smape_list.append(smape_T)
                        mae_list.append(mae_T)
                    df_smape.loc[i] = smape_list
                    df_mae.loc[i] = mae_list   
                    smape_list = []
                    mae_list =[]
                smape_mean = df_smape.mean().mean()
                mae_mean = df_mae.mean().mean()

                print('inverse MAE of target: {}, \ninverse sMAPE of target: {}\n'.format(round(mae_mean, 3), round(smape_mean, 2)))


        return

   