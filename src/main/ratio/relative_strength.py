import numpy as np
import pandas as pd
from ccxt import Exchange

from src.main.exchange_helper import get_swap_symbols_by_quote, filter_symbols_by_volume, get_ohlc_data, \
    get_bybit_swap_exchange

TIMEFRAME = '1d'
TOTAL_CANDLE_LIMIT = 600
LARGE_CAP_VOLUME_USDT = 5_000_000
ADF_WINDOW = 200 # сколько свечей использовать для ADF
HALF_LIFE_WINDOW = 100
MA_PERIOD = 48 # equivalent lookback for Bollinger Bands
MA_SLOPE_WINDOW = 100
MA_SLOPE_LOOKBACK = 50
MAX_MA_SLOPE = 100#0.03

def fetch_all_closes(symbols: list[str], exchange: Exchange, timeframe = TIMEFRAME) -> dict[str, pd.Series]:
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

def ratio_trend_stats(ratio: pd.Series, window: int) -> dict | None:
    """Метрики тренда для лог-ratio за последние window баров."""
    y = np.log(ratio.dropna().to_numpy()[-window:])
    if len(y) < window:
        return None
    x = np.arange(window)
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - (resid ** 2).sum() / ss_tot if ss_tot > 0 else 0.0
    path = np.abs(np.diff(y)).sum()
    er = abs(y[-1] - y[0]) / path if path > 0 else 0.0
    return {
        "move_pct": (np.exp(slope * (window - 1)) - 1) * 100,  # изменение по линии регрессии за окно
        "r2": r2,
        "er": er,
    }


def relation_trend_to_symbol(
    exchange: Exchange,
    symbol_main: str = 'BTC/USDT:USDT',
    window: int = 100,
    ma_fast_period: int = 20,
    ma_slow_period: int = 50,
    min_r2: float = 0.6,
    min_er: float = 0.12,
    min_move_pct: float = 5.0,
) -> pd.DataFrame:
    symbols = get_swap_symbols_by_quote(exchange)
    symbols = filter_symbols_by_volume(exchange, symbols, LARGE_CAP_VOLUME_USDT)
    symbols = [s for s in symbols if s != symbol_main]

    closes_main = fetch_all_closes([symbol_main], exchange)[symbol_main]
    closes_dict = fetch_all_closes(symbols, exchange)

    rows = []
    for symbol, closes in closes_dict.items():
        ratio = (closes / closes_main).dropna()
        if len(ratio) < max(window, ma_slow_period):
            continue

        stats = ratio_trend_stats(ratio, window)
        if stats is None:
            continue

        ma_fast = ratio.rolling(ma_fast_period).mean()
        ma_slow = ratio.rolling(ma_slow_period).mean()
        price, f, s = ratio.iloc[-1], ma_fast.iloc[-1], ma_slow.iloc[-1]

        # направление: регрессия и расположение MA должны совпадать
        if stats["move_pct"] > 0 and s < f < price:
            direction = "up"
        elif stats["move_pct"] < 0 and s > f > price:
            direction = "down"
        else:
            continue

        # фильтр качества тренда
        if stats["r2"] < min_r2 or stats["er"] < min_er or abs(stats["move_pct"]) < min_move_pct:
            continue

        rows.append({"symbol": symbol, "direction": direction, **stats})

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["score"] = df["r2"] * df["move_pct"].abs()
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.float_format', lambda x: '%.4f' % x)  # Красивый формат для чисел с плавающей точкой

    # Печать отсортированного датафрейма
    print(df.sort_values("score", ascending=False).reset_index(drop=True))
    return df.sort_values("score", ascending=False).reset_index(drop=True)

def run():
    exchange = get_bybit_swap_exchange()
    # ====== Basic screener ========== #
    relation_trend_to_symbol(exchange)


if __name__ == "__main__":
    run()