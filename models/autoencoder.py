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
if CWD.split(r"/")[-1] == 'Stock_Prediction_SA':
    from utils import Config, padding
else:
    from ..utils import Config, padding


DEFAULT_CONFIG = Config(input_dims=(88, 100), hidden_dims=(5, ),
    activation=F.tanh, dtype=torch.float64,)


class AutoEncoder(nn.Module):
    def __init__(self, config=DEFAULT_CONFIG, 
        input_dims: Optional[Tuple[int]]=(88, 100),
        hidden_dims: Optional[Tuple[int]]=(5, ),
        lstm_hidden: int=10):
        super(AutoEncoder, self).__init__()
        self.config = config
        if config.get('input_dims', 'object'): 
            input_dims = config.input_dims
            self.input_dims = input_dims

        if config.get('hidden_dims', 'object'): 
            hidden_dims = config.hidden_dims
            self.hidden_dims = hidden_dims
        if config.get('activation', 'object'): activation = config.activation
        else: activation = F.tanh
        self.activation = activation

        if config.get('dtype', 'object'): dtype = config.dtype
        else: dtype = torch.float64
        self.dtype = dtype

        self.LSTM_encoder = nn.LSTM(input_dims[-1], lstm_hidden, batch_first=False, dtype=self.dtype)
        self.Conv1D_encoder = nn.Conv1d(input_dims[0], input_dims[0], stride=1,
            padding=0, kernel_size=3, dtype=self.dtype)
        self.MaxPool1D_encoder = nn.MaxPool1d(kernel_size=2, stride=2, padding=0,
            return_indices=True)
        self.Dense_encoder = nn.Linear(in_features=352, 
            out_features=hidden_dims[0], dtype=self.dtype)
        self.Dense_decoder = nn.Linear(in_features=hidden_dims[0], 
            out_features=352, dtype=self.dtype)
        self.MaxUnpool1D_decoder = nn.MaxUnpool1d(kernel_size=2, stride=2, padding=0)
        self.DeConv1D_decoder = nn.ConvTranspose1d(input_dims[0], input_dims[0], stride=1,
            padding=0, kernel_size=3, dtype=self.dtype)
        self.LSTM_decoder = nn.LSTM(lstm_hidden, 100, batch_first=False, dtype=self.dtype)
    
    def forward(self, x: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
        x = padding(x, direction='left', pad_value=0., repeat=self.input_dims[1] - x.shape[1])
        assert x.shape == self.input_dims, f"x is of the wrong shape! Expected {self.input_dims}, got {x.shape}."
        x, _ = self.LSTM_encoder.forward(x)
        x = self.Conv1D_encoder.forward(x)
        x, ind = self.MaxPool1D_encoder.forward(x)
        x = x.reshape(-1)
        z = self.Dense_encoder.forward(x)
        x_ = self.Dense_decoder.forward(z)
        x_ = x_.reshape((self.input_dims[0], -1))
        x_ = self.MaxUnpool1D_decoder.forward(x_, indices=ind)
        x_ = self.DeConv1D_decoder.forward(x_)
        inv_idx = torch.arange(x_.size(1)-1, -1, -1).long()
        inv_h_f = x_.index_select(1, inv_idx)
        x_, _ = self.LSTM_decoder.forward(inv_h_f)
        return x_, z

    def __call__(self, x):
        return self.forward(x)