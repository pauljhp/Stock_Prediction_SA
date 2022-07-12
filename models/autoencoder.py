import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import sqlite3
import numpy as np
import pandas as pd
import os
from typing import Tuple, Dict, Optional, Union


CWD = os.getcwd()
if CWD.split(r"/")[-1].split("\\")[-1] == 'Stock_Prediction_SA':
    from utils import Config, padding
else:
    from ..utils import Config, padding


DEFAULT_CONFIG = Config(input_dims=(88, 100), hidden_dims=(5, ),
    activation=torch.tanh, dtype=torch.float64,)


class AutoEncoder(nn.Module):
    def __init__(self, config=DEFAULT_CONFIG, 
        input_dims: Optional[Tuple[int]]=(88, 50),
        hidden_dims: Optional[Tuple[int]]=(5, ),
        lstm_hidden: int=10,
        ):
        super(AutoEncoder, self).__init__()
        self.config = config
        if config.get('input_dims', 'object'): 
            input_dims = config.input_dims
            self.input_dims = input_dims

        if config.get('hidden_dims', 'object'): 
            hidden_dims = config.hidden_dims
            self.hidden_dims = hidden_dims
        
        if config.get('lstm_hidden', 'object'): 
            lstm_hidden = config.lstm_hidden
            self.lstm_hidden = lstm_hidden
        assert self.lstm_hidden // 2 == self.lstm_hidden / 2, "lstm_hidden must be even."

        if config.get('activation', 'object'): activation = config.activation
        else: activation = torch.tanh
        self.activation = activation

        if config.get('dtype', 'object'): dtype = config.dtype
        else: dtype = torch.float64
        self.dtype = dtype

        if config.get('padding', 'float'): pad = config.padding
        else: pad = -1.
        self.pad = pad

        self.LSTM_encoder = nn.LSTM(input_dims[-1], lstm_hidden, batch_first=config.get('batch_first', False), dtype=self.dtype)
        self.Conv1D_encoder1 = nn.Conv1d(input_dims[0], input_dims[0], stride=1,
            padding=1, kernel_size=3, dtype=self.dtype)
        self.Conv1D_encoder2 = nn.Conv1d(input_dims[0], input_dims[0], stride=1,
            padding=2, kernel_size=5, dtype=self.dtype)
        self.Conv1D_encoder3 = nn.Conv1d(input_dims[0], input_dims[0], stride=1,
            padding=3, kernel_size=7, dtype=self.dtype)
        self.MaxPool1D_encoder = nn.MaxPool1d(kernel_size=2, stride=2, padding=0,
            return_indices=True)
        self.Dense_encoder = nn.Linear(in_features=input_dims[0] * lstm_hidden // 2, 
            out_features=hidden_dims[0], dtype=self.dtype)
        self.Dense_decoder = nn.Linear(in_features=hidden_dims[0], 
            out_features=input_dims[0] * lstm_hidden // 2, dtype=self.dtype)
        self.MaxUnpool1D_decoder = nn.MaxUnpool1d(kernel_size=2, stride=2, padding=0)
        self.DeConv1D_decoder1 = nn.ConvTranspose1d(input_dims[0], input_dims[0], stride=1,
            padding=3, kernel_size=7, dtype=self.dtype)
        self.DeConv1D_decoder2 = nn.ConvTranspose1d(input_dims[0], input_dims[0], stride=1,
            padding=2, kernel_size=5, dtype=self.dtype)
        self.DeConv1D_decoder3 = nn.ConvTranspose1d(input_dims[0], input_dims[0], stride=1,
            padding=1, kernel_size=3, dtype=self.dtype)
        self.LSTM_decoder = nn.LSTM(lstm_hidden, input_dims[-1], batch_first=False, dtype=self.dtype)
    
    def forward(self, x: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
        if x.shape[-1] < self.input_dims[-1]:
            x = padding(x, direction='left', pad_value=self.pad, repeat=self.input_dims[0] - x.shape[1])
        assert x.T.detach().numpy().shape == self.input_dims, f"x is of the wrong shape! Expected {self.input_dims}, got {x.T.shape}."
        x, _ = self.LSTM_encoder.forward(x.T)
        x = self.activation(x)
        x = self.Conv1D_encoder1.forward(x)
        x = self.activation(x)
        x = self.Conv1D_encoder2.forward(x)
        x = self.activation(x)
        x = self.Conv1D_encoder3.forward(x)
        x = self.activation(x)
        x, ind = self.MaxPool1D_encoder.forward(x)
        x = self.activation(x)
        x = x.reshape(-1)
        z = self.Dense_encoder.forward(x)
        # z = self.activation(z)
        x_ = self.Dense_decoder.forward(z)
        x_ = self.activation(x_)
        x_ = x_.reshape((self.input_dims[0], -1))
        x_ = self.MaxUnpool1D_decoder.forward(x_, indices=ind)
        x_ = self.activation(x_)
        x_ = self.DeConv1D_decoder1.forward(x_)
        x_ = self.DeConv1D_decoder2.forward(x_)
        x_ = self.DeConv1D_decoder3.forward(x_)
        x_ = self.activation(x_)
        inv_idx = torch.arange(x_.size(1)-1, -1, -1).long()
        inv_h_f = x_.index_select(1, inv_idx)
        x_, _ = self.LSTM_decoder.forward(inv_h_f)
        # x_ = self.activation(x_)
        return x_.T, z

    def __call__(self, x):
        return self.forward(x)