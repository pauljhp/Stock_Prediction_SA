# Deep learning with LSTM and Conv layers for stock price prediction, using news sentment data

## 1. Loading data

A dataset class has been implemented already. 

This uses the iterable style of the `Dataset` class.

Initiate the BaseDataset class by passing a Config object, and wrap the kwargs
into the Config object. 

When calling the `__iter__` method using `for data in dataset`, a triple will be returned `(x1, x2, y)`. x1 is the sentiment/price data, x2 is the past financial data up until the train data time, and y is the share price performance during the look_forward period (defaul 5 days, or a trading week). 

The `forward()` method needs to take in (x1, x2), as x2 carries information about the stock's fundamentals and type, but is of a different dimension as x1. An autoencoder model has been implemented and trained to convert the different dimensions of x2 into the same hidden dimensions (and is also useful for dimensionality reduction).

Note x1 and x2 are scaled to between -1 and 1. 

## Model

### Autoencoder

Built for reducing the dimensionality of x2 and standardize dimensions for feeding into the main model. We can pretrain this, and pass it as a parameter to the main model. When calling forward(x1, x2, y), the autoencoder will be called first, and the output will be passed to the main model.