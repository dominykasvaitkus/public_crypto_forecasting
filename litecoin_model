import logging
import os
import sys
import time
import curses

import numpy as np
from numpy.lib.function_base import average
import pandas as pd
import torch
import torch.nn as nn
from icecream import ic
from sklearn.preprocessing import MinMaxScaler

def dataPreprocess(df):
    # df = df.truncate(after=df.index[1000])
    all_currencies = good_currencies.copy()
    all_currencies.append(pred_currency)
    y_scaler = MinMaxScaler(feature_range=(-1, 1)) # adds predicting cryptocurrency close
    input_data = df['close_'+pred_currency][:].values.astype(float).reshape(-1, 1)
    input_data = y_scaler.fit_transform(input_data)

    scaler = MinMaxScaler(feature_range=(-1, 1))    
    for idx in range(0, len(crypto_ohlc_cols)-1): # adds predicting cryptocurrency leftover columns
        col_data = df[crypto_ohlc_cols[idx]+pred_currency][:].values.astype(float).reshape(-1, 1)
        col_data = scaler.fit_transform(col_data)
        input_data = np.append(input_data, col_data, axis=1)
    
    for currency in good_currencies: # adds other cryptocurrencies
        for column in crypto_ohlc_cols:
            col_data = df[column+currency][:].values.astype(float).reshape(-1, 1)
            col_data = scaler.fit_transform(col_data)
            input_data = np.append(input_data, col_data, axis=1)

    input_data_tensor = []
    for i in range(len(input_data)-train_window-pred_offset): # splits data into x and y
        window_x = np.array(input_data[i:i+train_window])
        window_x = torch.FloatTensor(window_x)
        window_y = np.array(np.array([k[0] for k in input_data[i+train_window+pred_offset:i+train_window+pred_offset+pred_timesteps]]).reshape(1, -1))
        window_y = torch.FloatTensor(window_y)
        input_data_tensor.append([window_x, window_y])

    return input_data_tensor, y_scaler

def update_console_inference(screen, save_dir, train_loss, model_accuracy, p_print, y_print, iterations, data_len):
    try:
        screen.clear()
        screen.addstr(0, 0, f'INFERENCE model ' + save_dir.split('\\')[-1].replace('.ckpt', ''))
        screen.addstr(1, 0, f'{iterations/data_len*100:{1}.{3}}% finished; {iterations} out of {data_len}')
        screen.addstr(3, 0, f'Train loss: {train_loss:{1}.{5}} model accuracy: {model_accuracy:{1}.{5}}')
        screen.addstr(5, 0, f'y_pred: {p_print}')
        screen.addstr(8, 0, f'y_label: {y_print}')
        
        screen.refresh()
    except ValueError:
        pass

def update_console_training(screen, save_dir, train_loss, start_time, total_iter, epochs, train_iterations, epoch, p_print, y_print):
    time_spent = time.time() - start_time
    progress_done = total_iter / (epochs * train_iterations) * 100 + 0.00001

    screen.clear()
    screen.addstr(0, 0, 'Model: ' + save_dir.split('\\')[-1].replace('.ckpt', ''))
    screen.addstr(1, 0, f'Training for: {epochs} epochs, {train_iterations} datapoints')
    screen.addstr(2, 0, f'Feature size: {feature_size} | LSTM input size: {input_size} | LSTM neuron number: {lstm_neuron_num}')
    screen.addstr(4, 0, f'Time spent: {time_spent:{1}.{6}} seconds | ETA: {time_spent*100/progress_done:{1}.{6}} | iterations: {total_iter} | epoch: {epoch+1}')
    screen.addstr(5, 0, f'Iterations per second: {total_iter*(epoch+1)/time_spent:{1}.{4}} | progress done: {progress_done:{1}.{5}}%')
    screen.addstr(6, 0, f'Train loss: {train_loss:{1}.{3}} | y_pred: {p_print[0][0]:{1}.{5}} | y_label: {y_print[0][0]:{1}.{5}}')
    screen.refresh()
    # TODO: add layer size

def update_cell(model):
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    model.hidden_cell = (torch.zeros(lstm_layer_count,1,model.hidden_layer_size, requires_grad=True).to(device), torch.zeros(lstm_layer_count,1,model.hidden_layer_size, requires_grad=True).to(device))
    pass

def initialize_logger(filename, logger_name):
    dir = 'C:\\Coding\\git_crypto_forecasting\\coinbase_bot\\nn_models\\logs\\'
    logger = logging.getLogger(logger_name)
    handler = logging.FileHandler(dir + filename, mode='w')
    formatter = logging.Formatter('%(message)s')

    handler.setFormatter(formatter)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

    return logger

def training(train_data, scaler):
    device = ('cuda' if torch.cuda.is_available() else 'cpu')

    screen = curses.initscr()
    # model.load_state_dict(torch.load(save_dir))

    model.train()
    logger_train = initialize_logger('training_' + file_name, 'training')

    train_iterations = len(train_data)
    start_time = time.time()
    cell_update = update_cell
    for epoch in range(epochs):
        total_iter = 0
        for i, (x_input, y_label) in enumerate(train_data):
            try:
                x_input, y_label = x_input.to(device), y_label.to(device)
                cell_update(model)
                y_pred = model(x_input)
                train_loss = loss_function(y_pred, y_label)
                train_loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                if i % 100 == 0:
                    torch.save(model.state_dict(), save_dir)
                    with torch.no_grad():
                        p_print = scaler.inverse_transform (y_pred.cpu())
                        y_print = scaler.inverse_transform(y_label.cpu().reshape(1, -1))
                        update_console_training(screen, save_dir, train_loss, start_time, total_iter, epochs, train_iterations, epoch, p_print, y_print)
                        logger_train.info(str(train_loss.item()) + ';' + str(p_print[0][0]) + ';' + str(y_print[0][0]))

                total_iter += 1
            except (RuntimeError, OSError) as error:
                screen.addstr(0, 0, 'Critical error has occured during training')
                screen.addstr(1, 0, error)

def inference(test_data, scaler):
    device = ('cuda' if torch.cuda.is_available() else 'cpu')

    model.load_state_dict(torch.load(save_dir))
    model.eval()

    screen = curses.initscr()
    file_name = os.path.basename(__file__).replace('.py', '.log')
    logger = initialize_logger('inference_' + file_name, 'inference')

    model_accuracy = 0
    temp_rolling_average = 0
    total_profit = 0
    num_of_trades = 0
    cell_update = update_cell
    for iterations, (x_input, y_label) in enumerate(test_data):
        try: 
            with torch.no_grad():
                x_input, y_label = x_input.to(device), y_label.to(device)
                cell_update(model)

                y_pred = model(x_input)
                train_loss = loss_function(y_pred, y_label)
                p_print = scaler.inverse_transform(y_pred.cpu())
                y_print = scaler.inverse_transform(y_label.cpu().reshape(1, -1))
                log_info = str(train_loss.item()) + ';' + str(p_print[0][0]) + ';' + str(y_print[0][0])
                logger.info(log_info)
                if iterations % 1000 == 0:
                    update_console_inference(screen, save_dir, train_loss, 0, p_print, y_print, iterations, len(test_data))
    
        except RuntimeError:
            break # pasiekta inferencijos duomenu pabaiga
        
    print(f'Model learning finished succesfully, model accuracy: {model_accuracy*100}, trades: {num_of_trades}, profit: {total_profit}')
    torch.save(model.state_dict(), save_dir)

# Trained in total: 10 epochs
class LSTM(nn.Module):
    def __init__(self, feature_size, input_size, hidden_layer_size, stride):
        super().__init__()
        torch.manual_seed(0)
        device = ('cuda' if torch.cuda.is_available() else 'cpu')
        self.hidden_layer_size = hidden_layer_size
        self.conv = nn.Conv2d(1, 1, (kernel_size, kernel_size), stride)
        self.lstm = nn.LSTM(input_size, hidden_layer_size, dropout=0.2, num_layers=lstm_layer_count).to(device)
        self.activation = nn.SELU()
        self.linear = nn.Linear(hidden_layer_size, 1).to(device)
        self.hidden_cell = (torch.zeros(lstm_layer_count,1,self.hidden_layer_size, requires_grad=True).to(device), torch.zeros(lstm_layer_count,1,self.hidden_layer_size, requires_grad=True).to(device))
    
    def forward(self, input_seq):
        device = ('cuda' if torch.cuda.is_available() else 'cpu')
        input_seq = torch.reshape(input_seq, (1, 1, train_window, feature_size))
        input_seq = self.conv(input_seq)
        ic(input_seq.shape)
        input_seq = torch.reshape(input_seq, (processed_out_channels, input_size))

        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        lstm_out = self.activation(lstm_out)
        predictions = self.linear(lstm_out.view(len(input_seq), 1, -1)).to(device)

        return predictions[-1]

if __name__ == '__main__':
    # Base setup
    torch.manual_seed(0)
    device = ('cuda' if torch.cuda.is_available() else 'cpu') # paskelbiama kuriame prietaise bus procesuojami tenzoriai
    
    # Input feature details
    good_currencies = ['DASH', 'LTC', 'ZEC', 'LINK', 'TRB', 'ATOM']
    pred_currency = 'LTC'
    crypto_ohlc_cols = ['open_', 'high_', 'low_', 'close_']
    feature_size = 4 + len(good_currencies)*4
    # Convolutional layer parameters 
    kernel_size = 2
    stride = 1
    out_channels = 75
    processed_out_channels = 75  - kernel_size + 1
    # LSTM parameters
    lstm_layer_count = 2
    lstm_neuron_num = 256
    input_size = int((feature_size - (kernel_size - 1)) / stride)
    pred_offset = 0 # how far into the future we're predicting
    pred_timesteps = 1  # amount of predictions generated
    train_window = 75 # input window size (minutes)
    # Training parameters 
    epochs = 5
    learn_rate = 0.000000062525
    
    # Directories
    file_name = os.path.basename(__file__).replace('.py', '')   # automatinis issaugojimas
    save_dir = 'C:\\Coding\\git_crypto_forecasting\\coinbase_bot\\nn_models\\checkpoints\\' + file_name + '.ckpt'
    train_df_dir = 'C:\Coding\\2021_q2_1m_all3.1.3.csv' # treniravimo duomenu direktorija
    test_df_dir = 'C:\Coding\git_crypto_forecasting\coinbase_bot\data\\train_data\\2021-07_1m_all3.csv' # treniravimo duomenu direktorija

    # Model and optimizers
    model = LSTM(feature_size, input_size, lstm_neuron_num, stride).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate, weight_decay =0.00005)
    loss_function = torch.nn.SmoothL1Loss()

    # Training
    train_data, train_scaler = dataPreprocess(pd.read_csv(train_df_dir))
    training(train_data, train_scaler)

    # Testing
    test_data, test_scaler = dataPreprocess(pd.read_csv(test_df_dir))
    inference(test_data, test_scaler)
