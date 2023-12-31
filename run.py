from argparse import ArgumentParser, Namespace, BooleanOptionalAction
from src import download_data, create_model, load_data, prepare_data, split_dataset, plot_history
from keras.models import load_model
from os.path import exists
from os import mkdir
from datetime import datetime

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
        '--target', default='close', help='Feature to train on'
    )
    parser.add_argument(
        '--data-dir', default='./data', help="Where is training data located"
    )
    parser.add_argument(
        '--exchange', default='binance', help="From what exchange to download data"
    )
    parser.add_argument(
        '--classify', 
        help="Should model try to predict next value or will market direction", 
        action=BooleanOptionalAction,
        default=False, 
    )

    return parser.parse_args()


def main():
    args = get_args()

    action = args.action

    match action:
        case 'download':
            download(args)
        case 'train':
            train(args)
        case 'predict':
            predict(args)
            

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


def train(args: Namespace):
    kwargs = args.__dict__

    symbols = kwargs.pop('coins').split(',')
    timeframe = kwargs.pop('timeframe')
    exchange = kwargs.pop('exchange')
    data_dir = kwargs.pop('data_dir')
    classify = kwargs.get('classify')

    df = load_data(symbols, exchange, timeframe, data_dir, **kwargs)
    df = prepare_data(df, symbols[0], True, classify=classify)
    train_x, train_y, test_x, test_y = split_dataset(df, classify=classify)

    model = create_model(len(symbols), split_ratio=0.85, **kwargs)

    history = model.fit(
        x = [i for i in train_x],
        y = train_y,
        epochs=100,
        validation_data=(
            [i for i in test_x],
            test_y
        )
    ).history
    
    plot_history(history, 'acc' if classify else 'loss')
    model.save('classify-model.keras' if classify else 'regression-model.keras')

def predict(args: Namespace):
    print("Downloading latest data")
    download(args)

    kwargs = args.__dict__

    symbols = kwargs.pop('coins').split(',')
    timeframe = kwargs.pop('timeframe')
    exchange = kwargs.pop('exchange')
    data_dir = kwargs.pop('data_dir')
    lookback = kwargs.get('lookback', 30)
    classify = kwargs.get('classify')

    df = load_data(symbols, exchange, timeframe, data_dir, **kwargs)
    df = prepare_data(df, symbols[0], False, )
    unix = (df.iloc[-1].name + (df.iloc[-1].name - df.iloc[-2].name)) / 1000
    unix = datetime.fromtimestamp(unix)

    df = df.iloc[-lookback:].to_numpy()
    df = df.reshape(len(symbols), 1, lookback, 1)

    model = load_model('classify-model.keras' if classify else 'regression-model.keras')
    prediction = model.predict([i for i in df], verbose=False)[0]

    if classify:
        buy_signal, sell_signal = prediction
        if buy_signal == sell_signal:
            prediction = 'hold'
        else:
            prediction = 'buy' if buy_signal > sell_signal else 'sell'
    else:
        prediction = prediction[0] * 100
        prediction = f'{prediction:.4f}%'

    print(f"Prediction for {symbols[0]} is {prediction}, prediction time: {unix}")


if __name__ == "__main__":
    main()