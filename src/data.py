from requests import get
from .exceptions import DataDownload
from os.path import join
from pandas import DataFrame, read_csv, concat
from numpy import ndarray, zeros

def download_data(symbol: str, timeframe: str, exchange: str, data_dir: str):
    url = f"https://www.cryptodatadownload.com/cdd/{exchange.capitalize()}_{symbol.upper()}_{timeframe}.csv"
    data = get(url)

    if not data.ok:
        raise DataDownload(f"Unable to download data for {symbol} at {exchange} with timeframe {timeframe}")

    data = data.text.split('\n', 1)
    file = filename(symbol, exchange, timeframe, data_dir)

    with open(file, 'w') as file:
        file.write(data[1])


def load_data(symbols: list[str], exchange: str, timeframe: str, data_dir: str, **kwargs) -> DataFrame:
    data = {}
    target = kwargs.get('target', 'close')

    for symbol in symbols:
        df = read_csv(filename(symbol, exchange, timeframe, data_dir))
        df.columns = [i.lower() for i in df.columns]
        df = df.set_index('unix').sort_index()
        df = df.rename(columns={target: symbol})
        data[symbol] = df[symbol]

    return concat(data.values(), axis=1).dropna()

def prepare_data(df: DataFrame, target: str, train: bool) -> DataFrame:
    df = df.pct_change()

    if train:
        df['target'] = df[target].shift(-1)
        
    return df.dropna()


def split_dataset(df: DataFrame, lookback: int = 30, split_ratio: float = 0.9) -> (ndarray, ndarray, ndarray, ndarray):
    df_shape = df.shape
    shape = (df_shape[0] - lookback, lookback, df_shape[1] - 1, 1)

    features = [column for column in df.columns if not column == 'target']

    x = zeros(shape)
    y = zeros((df_shape[0] - lookback, 1, 1))

    for index in range(lookback, len(df)):
        row = df.iloc[index - lookback: index][features].to_numpy().reshape(shape[1:])
        x[index - lookback] = row
        y[index - lookback] = df.iloc[index - 1]['target']

    split = int(shape[0] * split_ratio)

    train_x = x[:split]
    test_x = x[split:]

    train_y = y[:split]
    test_y = y[split:]


    return (train_x.swapaxes(0, 2).swapaxes(1, 2), train_y, test_x.swapaxes(0, 2).swapaxes(1, 2), test_y)

def filename(symbol: str, exchange: str, timeframe: str, data_dir: str) -> str:
    file = f"{symbol}-{exchange}-{timeframe}.csv"
    return join(data_dir, file)