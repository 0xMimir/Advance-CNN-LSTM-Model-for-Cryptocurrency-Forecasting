from requests import get
from .exceptions import DataDownload

def download_data(symbol: str, timeframe: str, exchange: str):
    url = f"https://www.cryptodatadownload.com/cdd/{exchange.capitalize()}_{symbol.upper()}_{timeframe}.csv"
    data = get(url)

    if not data.ok:
        raise DataDownload(f"Unable to download data for {symbol} at {exchange} with timeframe {timeframe}")

    data = data.text
    print(data)