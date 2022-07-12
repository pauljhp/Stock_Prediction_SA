import torch
from torch.utils.data import Dataset, DataLoader
import sqlite3
import numpy as np
import pandas as pd
import os
import datetime as dt
from typing import Tuple, Union, Optional
from sklearn.preprocessing import MinMaxScaler
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging


CWD = os.getcwd()
if CWD.split(r"/")[-1].split("\\")[-1] == 'Stock_Prediction_SA':
    from utils import iter_by_chunk, Config
else:
    from ..utils import iter_by_chunk, Config

LOGPATH = './.log/'
LOGFILE = os.path.join(LOGPATH, 'log.log')

if not os.path.exists(LOGPATH):
    os.makedirs(LOGPATH)

logging.basicConfig(filename=LOGFILE, 
    level=logging.DEBUG)


class BaseDataset(Dataset):
    def load_ticker_data(self, 
        ticker: str,
        look_back: int=100,
        look_forward: int=5, 
        mode: str='train',
        dtype: torch.dtype=torch.float64,
        fill_value: float=-1.,
        fill_na: bool=True,
        ignore_error: bool=True
        ):
        """
        :param ticker: str of the ticker to load
        :param look_back: int of the number of days to look back
        """
        assert ticker in self.tickers, f"{ticker} not in tickers"
        df_sent = pd.read_sql(f"SELECT * FROM `{ticker}`", self.sql_conn_sent)
        df_sent = df_sent.set_index(["level_0", "level_1"])
        df_sent.index.names = ['category', 'item']
        df_sent.columns = df_sent.columns.to_series().apply(
            lambda x: dt.datetime.strptime(x, '%Y-%m-%d').date()).values
        df_funda = pd.read_sql(f"SELECT * FROM `{ticker}`", self.sql_conn_funda)
        df_funda = df_funda.set_index("index")
        df_funda.columns = df_funda.columns.to_series().apply(
            lambda x: dt.datetime.strptime(x, "%Y-%m-%d %H:%M:%S").date()).values
        df_funda.columns.names = ['date']
        df_funda = df_funda.T.reset_index().sort_values(
            "date", ascending=True).set_index("date").T
        available_periods = df_sent.columns

        if mode == 'train':
            available_periods = available_periods.to_series()[: self.train_test_split_date]
        elif mode in ['test', 'val']:
            available_periods = available_periods.to_series()[self.train_test_split_date:]
        else:
            raise NotImplementedError  
        for periods in iter_by_chunk(available_periods, look_back):
            try:
                f_price_loc = available_periods.index.get_loc(periods[-1]) + look_forward
                if f_price_loc <= len(available_periods):
                    if fill_na:
                        x1 = df_sent.loc[:, periods].fillna(fill_value)
                        x1 = self.sent_scaler.transform(x1.T.values).T
                        x2 = df_funda.loc[:, :periods[-1]].fillna(fill_value)
                        x2 = self.funda_scaler.transform(x2.T.values).T
                    else:
                        x1 = df_sent.loc[:, periods]
                        x1 = self.sent_scaler.transform(x1.T.values).T
                        x2 = df_funda.loc[:, :periods[-1]]
                        x2 = self.funda_scaler.transform(x2.T.values).T

                    y = (df_sent.iloc[10, f_price_loc] / df_sent.iloc[10, f_price_loc - look_forward]) / df_sent.iloc[10, f_price_loc - look_forward]
                    yield (torch.tensor(x1, dtype=dtype), 
                        torch.tensor(x2, dtype=dtype), 
                        torch.tensor(y, dtype=dtype))
            except Exception as e:
                if ignore_error:
                    logging.error(f"{e} - happened while loading {ticker}")
                    pass
                else:
                    raise e


    def __init__(self, config: Config, 
            data_path: Optional[str]="./data", 
            look_back: int=100,
            look_forward: int=5,
            train_test_split_date: Optional[dt.date]=dt.date(2021, 6, 1),
            dtype: Optional[torch.dtype]=torch.float64,
            fill_value: Optional[float]=-1.,
            fill_na: Optional[bool]=True,
            mode: Optional[str]='train',
            feature_range: Tuple[float] = (-1., 1.),
            num_workers: int=0,
            ):
        """
        :param config: Config object. Required. You can initialize a 
            utils.Config object, then fill it in with values. You can also
            instantiate a Config object by passing a dictionary.
            
            e.g.
            ```
            from utils import Config
            config = Config(mode='train', data_path='./data')
            config.mode = 'train'
            config.dtype = torch.float64

            ---or---
            config = Config({'mode': 'train', 'dtype': torch.float64})
            ```
        
        The following params will be overwritten if you have passed in
        a Config object, and it contains the required keys:
        :param data_path: str of the path to the data directory. 
        :param train_test_split_date: dt.date of the date to split the data
        :param dtype: torch.dtype of the data
        :param fill_value: float of the value to fill in missing values
        :param fill_na: bool of whether to fill missing values
        :param mode: str of the mode to load data for. Must be one of 'train',
            'test', or 'val'.
        """
        super(BaseDataset, self). __init__()
        if config.get('train_test_split_date'): 
            train_test_split_date = config.train_test_split_date
        if config.get('dtype', returntype='object'): dtype = config.dtype
        if config.get('fill_value', 'float'): fill_value = config.fill_value
        if config.get('fill_na', 'bool'): fill_na = config.fill_na
        if config.get('mode'): mode = config.mode
        if config.get('look_back', 'object'): look_back = config.look_back
        if config.get('look_forward', 'object'): look_forward = config.look_forward
        if config.get('data_path'): data_path = config.data_path
        if config.get('feature_range'): feature_range = config.feature_range

        self.sql_conn_sent = sqlite3.connect(
            os.path.join(data_path, 'spx_news_sentiment_price.db'))
        self.sql_conn_funda = sqlite3.connect(
            os.path.join(data_path, 'spx_news_sentiment_fundamental.db')
        )
        self.sent_cur = self.sql_conn_sent.cursor()
        self.funda_cur = self.sql_conn_funda.cursor()
        self.train_test_split_date = train_test_split_date
        news_tickers = pd.read_sql("""SELECT `name` FROM `sqlite_master` 
        WHERE type ='table' AND name NOT LIKE 'sqlite_%';""", 
        self.sql_conn_sent)['name'].to_list()
        funda_tickers = pd.read_sql("""SELECT `name` FROM `sqlite_master` 
        WHERE type ='table' AND name NOT LIKE 'sqlite_%';""", 
        self.sql_conn_funda)['name'].to_list()
        self.tickers = [i for i in news_tickers if i in funda_tickers] # in case both databases don't contain the same tickers, which shouldn't happen
        self.sent_scaler, self.funda_scaler = (MinMaxScaler(feature_range=feature_range), 
            MinMaxScaler(feature_range=feature_range))
        self.sent_scaler.fit(pd.concat([pd.read_sql(
            f"select * from `{ticker}`", self.sql_conn_sent
                ).set_index(["level_0", "level_1"]).T
            for ticker in self.tickers]).values,
            )
        self.funda_scaler.fit(pd.concat([pd.read_sql(
            f"select * from `{ticker}`", self.sql_conn_funda
                ).set_index(["index"]).T
            for ticker in self.tickers]).values,
            )
        self.data = list()
        if num_workers < 1:
            for ticker in self.tickers:
                for data in self.load_ticker_data(ticker=ticker, 
                    look_back=look_back, look_forward=look_forward,
                    mode=mode, dtype=dtype, fill_value=fill_value, 
                    fill_na=fill_na,
                    ignore_error=config.get('ignore_error', default=True, 
                        returntype='bool')
                    ):
                    self.data.append(data)
        else:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                for chunk in iter_by_chunk(self.ticker):
                    futures = [executor.submit(self.load_ticker_data, 
                        ticker=ticker, look_back=look_back, 
                        look_forward=look_forward,
                        mode=mode, dtype=dtype, fill_value=fill_value, 
                        fill_na=fill_na,
                        ignore_error=config.get('ignore_error', default=True, 
                            returntype='bool')) 
                        for ticker in chunk]
                    for future in as_completed(futures):
                        self.data.append(future.result())

    def __len__(self):
        return len(self.tickers)

    def __iter__(self):
        return self.data.__iter__()