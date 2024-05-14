import argparse
import json
import os
import sys
import torch
from capacity_dataset import CapacityDataset
from utils import to_var, collate, Normalizer, PreprocessNormalizer
from model import LSTMNet, TCN
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import pickle
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from utils import build_loc_net, get_fc_graph_struc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV
import statsmodels.api as sm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from statsmodels.tsa.stattools import adfuller


def mean_absolute_percentage_error(y_true, y_pred):
    mape = 0
    for i, j in zip(y_true, y_pred):
        mape += (np.abs((i - j) / i)) * 100
    return mape

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch capacity estimation')
    parser.add_argument('--fold_num', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--model', type=str, default='LSTMNet')
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--load_saved_dataset', action='store_true')


    args = parser.parse_args()

    print("args", args)

    # load dataset
    if args.load_saved_dataset:
        with open(f'saved_dataset/train_dataset_fold_{args.fold_num}.pkl', 'rb') as f:
            train_dataset = pickle.load(f)
        with open(f'saved_dataset/test_dataset_{args.fold_num}.pkl', 'rb') as f:
            test_dataset = pickle.load(f)
        train_data, val_data = train_test_split(train_dataset, test_size=0.2, random_state=42)

    else:
        train_pre = CapacityDataset(train=True, fold_num=args.fold_num)
        test_pre = CapacityDataset(train=False, fold_num=args.fold_num)

        normalizer = Normalizer(dfs=[train_pre[i][0] for i in range(200)],
                                     variable_length=False)
        train_dataset = PreprocessNormalizer(train_pre, normalizer_fn=normalizer.norm_func)
        test_dataset = PreprocessNormalizer(test_pre, normalizer_fn=normalizer.norm_func)
        os.makedirs("saved_dataset", exist_ok=True)
        with open(f'saved_dataset/train_dataset_fold_{args.fold_num}.pkl', 'wb') as f:
            pickle.dump(train_dataset, f)
        with open(f'saved_dataset/test_dataset_{args.fold_num}.pkl', 'wb') as f:
            pickle.dump(test_dataset, f)


    # LSTM
    if args.model == "LSTM":
        model = LSTMNet(input_dim=8, hidden_dim=128, output_dim=1, dropout=0.3).cuda()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.1, verbose=True)
        criterion = nn.MSELoss()
        criterion2 = nn.L1Loss()

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        train_losses = []
        test_losses = []

        # Training loop
        for epoch in range(args.num_epochs):
            model.train()
            epoch_loss = 0
            for batch_idx, batch_data in enumerate(tqdm(train_loader)):
                data = to_var(batch_data[0].float())
                capacity = to_var(batch_data[1]['capacity'].float())
                output = model(data)
                loss = criterion(output.reshape(-1), capacity.reshape(-1))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss+=loss.item() * data.shape[0]

            train_losses.append(epoch_loss / len(train_dataset))

            print('Epoch [{}/{}], Train Loss: {:.4f}'.format(epoch + 1, args.num_epochs, epoch_loss/len(train_dataset)))

            # Test loop
            model.eval()
            test_loss = 0
            mean_absolute_error_lstm = 0
            total_absolute_percentage_error = 0

            with torch.no_grad():
                for batch_idx, batch_data in enumerate(tqdm(test_loader)):
                    data = to_var(batch_data[0].float())
                    capacity = to_var(batch_data[1]['capacity'].float())
                    output = model(data)
                    loss = criterion(output.reshape(-1), capacity.reshape(-1))
                    mean_absolute_error_test = criterion2(output.reshape(-1), capacity.reshape(-1))
                    test_loss+=loss.item() * data.shape[0]
                    mean_absolute_error_lstm += mean_absolute_error_test.item() * data.shape[0]
                    absolute_percentage_error = torch.abs((output.reshape(-1) - capacity.reshape(-1)) / capacity.reshape(-1))
                    total_absolute_percentage_error += torch.sum(absolute_percentage_error).item()

            
            test_losses.append(test_loss / len(test_dataset))

            test_rmse = np.sqrt((test_loss / len(test_dataset)))
            mean_absolute_percentage_error_lstm = (total_absolute_percentage_error / len(test_dataset))*100

            print(
                'Epoch [{}/{}], Test Loss: {:.4f}, Test RMSE: {:.4f}, Test MAE: {:.4f}, Test MAPE: {:.4f}'.format(epoch + 1, args.num_epochs, test_loss/len(test_dataset), test_rmse, mean_absolute_error_lstm/len(test_dataset), mean_absolute_percentage_error_lstm))
    # XGBoost
    elif args.model == "XGBoost":
        train_data_list = [train_dataset.__getitem__(i)[0] for i in range(len(train_dataset))]
        train_labels_list = [train_dataset.__getitem__(i)[1]['capacity'] for i in range(len(train_dataset))]
        train_data = np.array(train_data_list).reshape(len(train_data_list), -1)
        train_labels = np.array(train_labels_list)

        test_data_list = [test_dataset.__getitem__(i)[0] for i in range(len(test_dataset))]
        test_labels_list = [test_dataset.__getitem__(i)[1]['capacity'] for i in range(len(test_dataset))]
        test_data = np.array(test_data_list).reshape(len(test_data_list), -1)
        test_labels = np.array(test_labels_list)

        dtrain = xgb.DMatrix(train_data.reshape(len(train_data_list), -1), label=train_labels)
        dtest = xgb.DMatrix(test_data.reshape(len(test_data_list), -1), label=test_labels)

        params = {
            'objective': 'reg:squarederror',
            'eta': 0.075,
            'max_depth': 10,
            'eval_metric': 'rmse',
            'reg_lambda': 0.01
        }

        num_rounds = args.num_epochs
        model = xgb.train(params, dtrain, num_rounds)

        train_predictions = model.predict(dtrain)
        test_predictions = model.predict(dtest)

        print('test MAE', mean_absolute_error(test_labels, test_predictions))
        print('test MAPE', mean_absolute_percentage_error(test_labels, test_predictions) / len(test_labels),'%')
        print("Test RMSE:", np.sqrt(np.mean((test_predictions - test_labels) ** 2)))

    #ARIMA
    elif args.model == 'ARIMA':
        
        train_labels_list = [train_dataset.__getitem__(i)[1]['capacity'] for i in range(len(train_dataset))]
        train_labels = np.array(train_labels_list)
        train_series = np.array(train_labels_list)

        train_data_list = [train_dataset.__getitem__(i)[0] for i in range(len(train_dataset))]
        train_data_np = np.array(train_data_list)
        
        mileage_list = [test_dataset.__getitem__(i)[1]['mileage'] for i in range(len(test_dataset))]
        mileage = np.array(mileage_list)

        test_data_list = [test_dataset.__getitem__(i)[0] for i in range(len(test_dataset))]
        test_data_np = np.array(test_data_list)

        test_labels_list = [test_dataset.__getitem__(i)[1]['capacity'] for i in range(len(test_dataset))]
        test_labels = np.array(test_labels_list)
        test_series = np.array(test_labels_list)

        exog_train = train_data_np.reshape(train_data_np.shape[0], -1)[:, :128]
        exog_test = test_data_np.reshape(test_data_np.shape[0], -1)[:, :128]

        #ARIMAX model fitting
        order = (2, 0, 1) 
        model = ARIMA(train_labels_list, order=order, exog=exog_train, trend='n')
        model_fit = model.fit()

        # Predicting
        test_pred = model_fit.forecast(steps=len(test_labels_list), exog=exog_test)

        test_rmse = np.sqrt(mean_squared_error(test_labels_list, test_pred))
        print("test RMSE:", test_rmse)
        test_mae= mean_absolute_error(test_labels_list, test_pred)
        print("test MAE:", test_mae)

        test_mape = mean_absolute_percentage_error(test_pred, test_labels_list)
        print("test MAPE:", test_mape / len(test_dataset))

    #TCN
    elif args.model == 'TCN':
        model = TCN(input_size=8, output_size=1, num_channels=[128, 64, 32], kernel_size=3, dropout=0.2).cuda()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.1, verbose=True)
        criterion = nn.MSELoss()
        criterion2 = nn.L1Loss()

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        train_losses = []
        test_losses = []

        # Training loop
        for epoch in range(args.num_epochs):
            model.train()
            epoch_loss = 0
            for batch_idx, batch_data in enumerate(tqdm(train_loader)):
                data = to_var(batch_data[0].float())
                data = data.permute(0, 2, 1)
                capacity = to_var(batch_data[1]['capacity'].float())
                output = model(data)
                loss = criterion(output.reshape(-1), capacity.reshape(-1))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * data.shape[0]

            train_losses.append(epoch_loss / len(train_dataset))
            print('Epoch [{}/{}], Train Loss: {:.4f}'.format(epoch + 1, args.num_epochs, epoch_loss / len(train_dataset)))

            
            # test loop
            model.eval()
            test_loss = 0
            test_mae = 0
            total_absolute_percentage_error = 0

            predicted_capacities = []
            actual_capacities = []
            mileages = []
            residuals = []

            with torch.no_grad():
                for batch_idx, batch_data in enumerate(tqdm(test_loader)):
                    data = to_var(batch_data[0].float())
                    data = data.permute(0, 2, 1)
                    capacity = to_var(batch_data[1]['capacity'].float())
                    output = model(data)
                    loss = criterion(output.reshape(-1), capacity.reshape(-1))
                    mae = criterion2(output.reshape(-1), capacity.reshape(-1))
                    test_loss += loss.item() * data.shape[0]
                    test_mae += mae.item() * data.shape[0]
                    absolute_percentage_error = torch.abs((output.reshape(-1) - capacity.reshape(-1)) / capacity.reshape(-1))
                    total_absolute_percentage_error += torch.sum(absolute_percentage_error).item()

            test_losses.append(test_loss / len(test_dataset))        
            
            test_rmse = np.sqrt((test_loss / len(test_dataset)))
            mean_absolute_percentage_error_tcn = (total_absolute_percentage_error / len(test_dataset))*100

            print('Epoch [{}/{}], Validation Loss: {:.4f}, Validation RMSE: {:.4f}, Validation MAE: {:.4f}, Validation MAPE: {:.4f}%'.format(
                epoch + 1, args.num_epochs, test_loss / len(test_dataset), test_rmse, test_mae / len(test_dataset), mean_absolute_percentage_error_tcn))

    else:
        raise NotImplementedError


