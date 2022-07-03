import datetime as dt 
import pandas as pd
import numpy as np
from typing import Union, Optional, Any, Dict, List
from copy import deepcopy
import itertools
from types import SimpleNamespace
import torch.nn.functional as F
import torch


def pandas_strptime(df: pd.DataFrame, 
    index_name: Optional[Union[str, List[str]]]=None,
    index_iloc: Optional[Union[int, List[str]]]=None,
    axis: Union[str, int]=0,
    datetime_format: str ="%Y-%m-%d",
    inplace: bool=False):
    """converts str datetime to np.datetime64
    :param index_name: index or column name to be processed
    :param index_iloc: positional index of the row/column to be processed
    :param axis: takes either 0/1, or 'index'/'columns'
    :param datetime_format: datetime.strptime format
    :param inplace: False by default, will create a deepcopy of the original 
        frame. Otherwise will changed the original frame inplace
    """
    assert index_name or index_iloc, 'index_name and index_iloc cannot be both unspecified'
    axes = {'index': 0, 'columns': 1}
    if isinstance(axis, str):
        axis = axes.get(axis)
    if inplace:
        if index_name:
            if isinstance(index_name, str):
                if axis:
                    df.loc[:, index_name] = df.loc[:, index_name]\
                        .apply(lambda x: dt.datetime.strptime(x, datetime_format))
                else:
                    df.loc[index_name, :] = df.loc[index_name, :]\
                        .apply(lambda x: dt.datetime.strptime(x, datetime_format))
            elif isinstance(index_name, list):
                if axis:
                    for ind, s in df.loc[:, index_name].iteritems():
                        df.loc[:, ind] = s.apply(lambda x: dt.datetime.strptime(x, datetime_format))
                else:
                    for ind, s in df.loc[index_name, :].iterrows():
                        df.loc[ind, :] = s.apply(lambda x: dt.datetime.strptime(x, datetime_format))

        else:   
            if isinstance(index_iloc, int):
                if axis:
                    df.iloc[:, index_iloc] = df.iloc[:, index_iloc]\
                        .apply(lambda x: dt.datetime.strptime(x, datetime_format))
                else:
                    df.iloc[index_iloc, :] = df.iloc[index_iloc, :]\
                        .apply(lambda x: dt.datetime.strptime(x, datetime_format))
            elif isinstance(index_iloc, list):
                if axis:
                    for ind, s in df.iloc[:, index_iloc].iteritems():
                        df.loc[:, ind] = s.apply(lambda x: dt.datetime.strptime(x, datetime_format))
                else:
                    for ind, s in df.iloc[index_iloc, :].iterrows():
                        df.loc[ind, :] = s.apply(lambda x: dt.datetime.strptime(x, datetime_format))
        return df

    else:
        newdf = deepcopy(df)
        if index_name:
            if isinstance(index_name, str):
                if axis:
                    newdf.loc[:, index_name] = newdf.loc[:, index_name]\
                        .apply(lambda x: dt.datetime.strptime(x, datetime_format))
                else:
                    newdf.loc[index_name, :] = newdf.loc[index_name, :]\
                        .apply(lambda x: dt.datetime.strptime(x, datetime_format))
            elif isinstance(index_name, list):
                if axis:
                    for ind, s in newdf.loc[:, index_name].iteritems():
                        newdf.loc[:, ind] = s.apply(lambda x: dt.datetime.strptime(x, datetime_format))
                else:
                    for ind, s in df.loc[index_name, :].iterrows():
                        newdf.loc[ind, :] = s.apply(lambda x: dt.datetime.strptime(x, datetime_format))

        else:   
            if isinstance(index_iloc, int):
                if axis:
                    newdf.iloc[:, index_iloc] = newdf.iloc[:, index_iloc]\
                        .apply(lambda x: dt.datetime.strptime(x, datetime_format))
                else:
                    newdf.iloc[index_iloc, :] = newdf.iloc[index_iloc, :]\
                        .apply(lambda x: dt.datetime.strptime(x, datetime_format))
            elif isinstance(index_iloc, list):
                if axis:
                    for ind, s in df.iloc[:, index_iloc].iteritems():
                        newdf.loc[:, ind] = s.apply(lambda x: dt.datetime.strptime(x, datetime_format))
                else:
                    for ind, s in df.iloc[index_iloc, :].iterrows():
                        newdf.loc[ind, :] = s.apply(lambda x: dt.datetime.strptime(x, datetime_format))
      
        return newdf


def iter_by_chunk(iterable: Any, chunk_size: int):
    """iterate by chunk size"""
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, chunk_size))
        if not chunk:
            break
        yield chunk


class Config(SimpleNamespace):
    def __init__(self, d: Optional[Dict]=None, **kwargs):
        if not d: d = dict()
        super(Config, self).__init__(**d, **kwargs)

    def get(self, name, returntype: Union[None, str]='str', 
        default: Any=None, ):
        """
        :param name: The name of the attribute to get.
        :param returntype: The typpe of the returned attribute. Will force 
            returned object into the specified type.
            Takes strings 'dict' and 'bool'
        :param default: The default value to return if the attribute is not found
        """
        if hasattr(self, name):
            res = getattr(self, name)
            if returntype == 'bool':
                return True if res else False
            elif returntype == 'dict':
                return vars(res)
            elif returntype == 'list':
                return list(res)
            elif returntype == 'str':
                return str(res)
            elif returntype == 'int':
                return int(res)
            elif returntype == 'float':
                return float(res)
            elif returntype == 'object':
                return res
            else:
                raise NotImplementedError("returntype must be 'dict' or 'bool'")
        else:
            return default


def padding(x: torch.tensor, direction: str='left',
    pad_value: Union[float, int, str]=0.,
    repeat: int=1):
    """padd a tensor with a value on one direction
    :param x: input to be padded
    :param direction: takes 'left' or 'right'. For 'up' or 'down', just tranpose
        the data
    :param pad_value: value to pad data with
    :param repeat: number of times to repeat the padding
    """
    if direction == 'left':
        return torch.cat([torch.tensor(pad_value).repeat(
                x.shape[0], repeat, ), x], dim=1)
    elif direction == 'right':
        return torch.cat([x, torch.tensor(pad_value).repeat(
                x.shape[0], repeat, )], dim=1)