import pandas as pd
from ccxt import Exchange

from src.main.exchange_helper import (
    get_ohlc_data,
    get_swap_symbols,
    filter_symbols_by_volume,
    filter_symbols_by_volume_range,
    get_bybit_swap_exchange,
)

LARGE_CAP_VOLUME_USDT = 10_000_000
MID_CAP_VOLUME_USDT = 2_000_000
SMALL_CAP_VOLUME_USDT = 400_000
LOWEST_CAP_VOLUME_USDT = 30_000

TIMEFRAME = '1h'
TOTAL_LIMIT = 200

# Параметры индикаторов
ATR_PERIOD = 14
ATR_MEAN_LONG_PERIOD = 50
ATR_MEAN_SHORT_PERIOD = 15
MAX_ATR_RATIO = 0.8

# ====== Screener logic ===========

def calculate_atr(df: pd.DataFrame, window: int = ATR_PERIOD) -> pd.Series:
    """Расчет True Range и Average True Range (ATR)."""
    high = df['high']
    low = df['low']
    close = df['close']

    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=window).mean()
    return atr

def detect_volatility_squeeze(symbols_by_volume: list[str], exchange: Exchange):
    print(f"Find symbols for processing: {len(symbols_by_volume)}")
    volatility_squeeze_list: list[tuple[str, float]] = []
    for symbol in symbols_by_volume:
        candles: pd.DataFrame = get_ohlc_data(symbol, exchange, TIMEFRAME, TOTAL_LIMIT)
        atr = calculate_atr(candles, window=ATR_PERIOD)
        atr_mean_long = atr.iloc[-ATR_MEAN_LONG_PERIOD:].mean()
        atr_mean_short = atr.iloc[-ATR_MEAN_SHORT_PERIOD:].mean()
        atr_ratio = atr_mean_short / atr_mean_long
        if atr_ratio < MAX_ATR_RATIO:
            volatility_squeeze_list.append((symbol, atr_ratio))

    print(f"========== Volatility squeeze list(low atr), count: {len(volatility_squeeze_list)} =========")
    volatility_squeeze_list.sort(key=lambda t: t[1])
    for symbol, atr_ratio in volatility_squeeze_list:
        print(f"{symbol}, atr ratio: {atr_ratio}")

def get_large_cap(exchange) -> list[str]:
    symbols = get_swap_symbols(exchange)
    return filter_symbols_by_volume(exchange, symbols, LARGE_CAP_VOLUME_USDT)

def get_mid_cap(exchange) -> list[str]:
    symbols = get_swap_symbols(exchange)
    return filter_symbols_by_volume_range(exchange, symbols, MID_CAP_VOLUME_USDT, LARGE_CAP_VOLUME_USDT)

def get_small_cap(exchange) -> list[str]:
    symbols = get_swap_symbols(exchange)
    return filter_symbols_by_volume_range(exchange, symbols, SMALL_CAP_VOLUME_USDT, MID_CAP_VOLUME_USDT)

def get_lowest_cap(exchange) -> list[str]:
    symbols = get_swap_symbols(exchange)
    return filter_symbols_by_volume_range(exchange, symbols, LOWEST_CAP_VOLUME_USDT, SMALL_CAP_VOLUME_USDT)

def run():
    exchange = get_bybit_swap_exchange()
    print("===== Large Cap Volume =====")
    detect_volatility_squeeze(get_large_cap(exchange), exchange)
    print("===== Mid Cap Volume =====")
    detect_volatility_squeeze(get_mid_cap(exchange), exchange)
    print("===== Small Cap Volume =====")
    detect_volatility_squeeze(get_small_cap(exchange), exchange)
    print("===== Lowest Cap Volume =====")
    detect_volatility_squeeze(get_lowest_cap(exchange), exchange)

if __name__ == "__main__":
    run()