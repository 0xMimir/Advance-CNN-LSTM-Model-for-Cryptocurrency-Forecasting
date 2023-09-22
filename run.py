from argparse import ArgumentParser, Namespace
from src.data import download_data
from os.path import exists 
from os import mkdir

def get_args() -> Namespace:
    parser = ArgumentParser(description='Advance CNN LSTM Model for Cryptocurrency Forecasting')

    parser.add_argument(
        'action', choices=['download', 'train', 'predict']
    )
    parser.add_argument(
        '--coins', default='BTCUSDT,ETHUSDT,XRPUSDT', help='List of coins for input of model, separated by comma', 
    )
    parser.add_argument(
        '--timeframe', default='d', help='Timeframe of input data: d for 1 day, h for hour, m for minute', choices=['d', 'h', 'm']
    )
    parser.add_argument(
        '--target', default='BTC-close', help='Feature to train on'
    )
    parser.add_argument(
        '--data-dir', default='./data', help="Where is training data located"
    )
    parser.add_argument(
        '--exchange', default='binance', help="From what exchange to download data"
    )

    return parser.parse_args()


def main():
    args = get_args()

    action = args.action

    match action:
        case 'download':
            download(args)
            

def download(args: Namespace):
    coins = args.coins.split(',')
    timeframe = args.timeframe
    exchange = args.exchange
    data_dir = args.data_dir

    if not exists(data_dir):
        mkdir(data_dir)
    
    for symbol in coins:
        download_data(symbol, timeframe, exchange, data_dir)
    
    return


if __name__ == "__main__":
    main()