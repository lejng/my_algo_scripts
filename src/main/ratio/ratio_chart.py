import pandas as pd
from matplotlib import pyplot as plt

from src.main.exchange_helper import get_ohlc_data, get_bybit_swap_exchange

TIMEFRAME_SMALL = '1h'
TIMEFRAME_HIGH = '1d'
TOTAL_CANDLE_LIMIT = 600
MA_PERIOD = 48

def draw_chart(ratio_small: pd.Series, ratio_high: pd.Series):
    fig, axes = plt.subplots(2, 1, figsize=(9, 6))

    axes[0].plot(ratio_small.index, ratio_small, label=f"ratio {TIMEFRAME_SMALL}", alpha=0.7, linewidth=2)
    axes[0].plot(ratio_small.index, ratio_small.rolling(MA_PERIOD).mean(), label='MA', alpha=0.7, linewidth=2)
    axes[0].set_title(f"ratio {TIMEFRAME_SMALL}")
    axes[0].legend()

    axes[1].plot(ratio_high.index, ratio_high, label=f"ratio {TIMEFRAME_HIGH}", alpha=0.7, linewidth=2)
    axes[1].plot(ratio_high.index, ratio_high.rolling(MA_PERIOD).mean(), label='MA', alpha=0.7, linewidth=2)
    axes[1].set_title(f"ratio {TIMEFRAME_HIGH}")
    axes[1].legend()

    plt.tight_layout()
    plt.show()

def run():
    symbol_a = 'WIF/USDT:USDT'
    symbol_b = '1000BONK/USDT:USDT'
    exchange = get_bybit_swap_exchange()
    df_a = get_ohlc_data(symbol_a, exchange, TIMEFRAME_SMALL, TOTAL_CANDLE_LIMIT)
    df_b = get_ohlc_data(symbol_b, exchange, TIMEFRAME_SMALL, TOTAL_CANDLE_LIMIT)
    ratio_small = df_a['close'] / df_b['close']

    df_high_a = get_ohlc_data(symbol_a, exchange, TIMEFRAME_HIGH, TOTAL_CANDLE_LIMIT)
    df_high_b = get_ohlc_data(symbol_b, exchange, TIMEFRAME_HIGH, TOTAL_CANDLE_LIMIT)
    ratio_high = df_high_a['close'] / df_high_b['close']
    draw_chart(ratio_small, ratio_high)

if __name__ == "__main__":
    run()

