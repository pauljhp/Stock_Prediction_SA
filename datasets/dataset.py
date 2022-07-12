import torch
from torch.utils.data import Dataset, DataLoader
import sqlite3
import numpy as np
import pandas as pd
import os

class baseDataset(Dataset):
    def __init__(self, data_path: str="./data"):
        self.sql_conn_sent = sqlite3.connect(
            os.path.join(data_path, 'spx_news_sentiment_price.db'))
        self.sql_conn_funda = sqlite3.connect(
            os.path.join(data_path, 'spx_news_sentiment_fundamental.db')
        )
        self.sent_cur = self.sql_conn_sent.cursor()
        self.funda_cur = self.sql_conn_funda.cursor()
        news_tickers = pd.read_sql("""SELECT `name` FROM `sqlite_schema` 
        WHERE type ='table' AND name NOT LIKE 'sqlite_%';""", 
        self.sql_conn_sent)['name'].to_list()
        funda_tickers = pd.read_sql("""SELECT `name` FROM `sqlite_schema` 
        WHERE type ='table' AND name NOT LIKE 'sqlite_%';""", 
        self.sql_conn_funda)['name'].to_list()
        self.tickers = [i for i in news_tickers if i in funda_tickers] # in case both databases don't contain the same tickers, which shouldn't happen

    def __len__(self):
        return len(self.tickers)

    def __getitem__(self, idx):
        return self.data[idx]