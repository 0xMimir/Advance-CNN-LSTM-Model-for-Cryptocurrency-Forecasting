from requests import get
from .exceptions import DataDownload
from os.path import join

def download_data(symbol: str, timeframe: str, exchange: str, data_dir: str):
    url = f"https://www.cryptodatadownload.com/cdd/{exchange.capitalize()}_{symbol.upper()}_{timeframe}.csv"
    data = get(url)

    if not data.ok:
        raise DataDownload(f"Unable to download data for {symbol} at {exchange} with timeframe {timeframe}")

    data = data.text.split('\n', 1)
    file = f"{symbol}-{exchange}-{timeframe}.csv"
    file = join(data_dir, file)

    with open(file, 'w') as file:
        file.write(data[1])