import ccxt
import pandas as pd
from ccxt import Exchange

def get_bybit_swap_exchange() -> Exchange:
    config = {
        'enableRateLimit': True,
        'options': {
            'createMarketBuyOrderRequiresPrice': False,
            'enableUnifiedAccount': True,
            'defaultType': 'swap',
            'fetchMarkets': {
                'types': ['linear']
            }
        }
    }
    bybit = ccxt.bybit(config)
    bybit.load_markets()

    return bybit

def get_swap_symbols(exchange: Exchange):
    markets = exchange.fetch_markets()
    return [
        market['symbol'] for market in markets
        if market.get('active', False) and market.get('swap', False)
    ]

def get_ohlc_data(symbol, exchange, timeframe='1h', limit=100) -> pd.DataFrame:
    ohlc = exchange.fetch_ohlcv(symbol, timeframe, limit)
    return (pd.DataFrame(ohlc, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
          .assign(timestamp=lambda x: pd.to_datetime(x['timestamp'], unit='ms'))
          .set_index('timestamp')
          #.dropna()
          .sort_index()
          )

def filter_symbols_by_volume(exchange: Exchange, symbols: list[str], min_volume_usdt: int) -> list[str]:
    tickers = exchange.fetch_tickers(symbols)
    valid_symbols = sorted(
        [
            symbol for symbol, data in tickers.items()
            if data.get('quoteVolume') and data.get('quoteVolume') >= min_volume_usdt
        ],
        key=lambda s: tickers[s]['quoteVolume'],
        reverse=True
    )
    return valid_symbols

def filter_symbols_by_volume_range(exchange: Exchange, symbols: list[str], min_volume_usdt: int, max_volume_usdt: int) -> list[str]:
    tickers = exchange.fetch_tickers(symbols)
    valid_symbols = sorted(
        [
            symbol for symbol, data in tickers.items()
            if data.get('quoteVolume') and min_volume_usdt <= data.get('quoteVolume') <= max_volume_usdt
        ],
        key=lambda s: tickers[s]['quoteVolume'],
        reverse=True
    )
    return valid_symbols
