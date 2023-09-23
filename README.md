# Advance CNN LSTM Model for Cryptocurrency Forecasting

This model is based on this [research paper](https://github.com/0xMimir/Advance-CNN-LSTM-Model-for-Cryptocurrency-Forecasting/blob/master/paper.pdf), this model utilizes input data from multiple crypto currencies instead of only using one. Using more than one cryptocurrency info data gives model general overview of market instead of only focusing on one asset, this also avoids over fitting, and when right params are put it's highly competitive model. In this repo I have not optimized model.

Requirements
---

Install requirements with:
```bash
pip3 install -r requirements.txt
```

Running
---

In order to see available options run:
```bash
python3 run.py
usage: run.py [-h] [--coins COINS] [--timeframe {d,h,m}] [--target TARGET] [--data-dir DATA_DIR] [--exchange EXCHANGE] {download,train,predict}
run.py: error: the following arguments are required: action
```

After that you can see all options there are 3 available commands, `download`, `train` and `predict`

### Download data

To download data for bitcoin, litecoin and monero run, 
```bash
python3 run.py --coins BTCUSDT,LTCUSDT,XMRUSDT download
```

By default this will download data for binance exchange, and data will be downloaded from [cryptodatadownload](https://www.cryptodatadownload.com/). You can specify timeframe with `--timeframe` available timeframe's are `d` dor day, `h` for hour and `m` for minute.

### Train

To train model run
```bash
python3 run.py --coins BTCUSDT,LTCUSDT,XMRUSDT train
```

By default this will train on `close` value for first asset specified, in case above, this will train for closing kline value of bitcoin. Same as for download command, you can specify timeframe. This will also create file `plot.png` with plot of loss across epochs.


### Predict

To predict values run:
```bash
python3 run.py --coins BTCUSDT,LTCUSDT,XMRUSDT predict
Predicted return for BTCUSDT is 0.4935%, prediction time: 2023-09-23 02:00:00
```

### Notes

This model is very bare bone implementation of algo from paper mentioned above:
* it only uses simple input data.
* it doesn't have optimization to find best markets.
* it has only one data source, ie. cryptodatadownload instead of using handlers directly from exchanges
* it doesn't have variable output, eg. trying to predict market change instead of asset change, or trying to do classification
