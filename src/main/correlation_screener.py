from itertools import combinations

import pandas as pd
from ccxt import Exchange

from src.main.exchange_helper import get_swap_symbols_by_quote, filter_symbols_by_volume, get_bybit_swap_exchange, \
    get_ohlc_data

MIN_VOLUME_USDT = 10_000_000
LARGE_TIMEFRAME = '1d'
SMALL_TIMEFRAME = '1h'
TOTAL_CANDLE_LIMIT = 600
MIN_CORRELATION = 0.5

def fetch_all_closes(symbols: list[str], exchange: Exchange, timeframe) -> dict[str, pd.Series]:
    """Скачивает свечи для каждого символа 1 раз"""
    print(f"Fetching candles for {len(symbols)} symbols...")
    closes_dict = {}
    for symbol in symbols:
        try:
            df = get_ohlc_data(symbol, exchange, timeframe, TOTAL_CANDLE_LIMIT)
            closes_dict[symbol] = df['close']
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
    return closes_dict

def calculate_correlation(closes_a: pd.Series, closes_b: pd.Series, use_returns: bool = True) -> float:
    if use_returns:
        s1 = closes_a.pct_change().dropna()
        s2 = closes_b.pct_change().dropna()
        return float(s1.corr(s2))
    return float(closes_a.corr(closes_b))

def print_large_cap_pairs_correlation(exchange: Exchange):
    print("========= large cap pairs correlation ============")
    symbols = get_swap_symbols_by_quote(exchange)
    symbols = filter_symbols_by_volume(exchange, symbols, MIN_VOLUME_USDT)
    # 1. Скачиваем свечи один раз
    print(f"Total {len(symbols)} symbols...")
    closes_dict_1h_tf: dict[str, pd.Series] = fetch_all_closes(symbols, exchange, SMALL_TIMEFRAME)
    closes_dict_1d_tf: dict[str, pd.Series] = fetch_all_closes(symbols, exchange, LARGE_TIMEFRAME)
    valid_symbols: list[str] = list(closes_dict_1d_tf.keys())

    all_combinations: list[tuple] = list(combinations(valid_symbols, 2))
    total_pairs = len(all_combinations)
    print(f"Analyzing {total_pairs} pairs combinations...")

    results: list[dict] = []

    # 2. Быстрый перебор комбинаций из памяти
    for i, (symbol_a, symbol_b) in enumerate(all_combinations, start=1):
        corr_1d_tf = calculate_correlation(
            closes_dict_1d_tf[symbol_a],
            closes_dict_1d_tf[symbol_b]
        )
        corr_1h_tf = calculate_correlation(
            closes_dict_1h_tf[symbol_a],
            closes_dict_1h_tf[symbol_b]
        )
        if corr_1d_tf > MIN_CORRELATION:
            results.append({
                'symbol_a': symbol_a,
                'symbol_b': symbol_b,
                'corr_1d_tf': corr_1d_tf,
                'corr_1h_tf': corr_1h_tf,
            })

        if i % 10 == 0 or i == total_pairs:
            print(f"Processed {i}/{total_pairs} pairs")

    results.sort(key=lambda x: x['corr_1d_tf'], reverse=True)
    for pair in results:
        print(pair)

def run():
    exchange = get_bybit_swap_exchange()
    print_large_cap_pairs_correlation(exchange)

if __name__ == "__main__":
    run()