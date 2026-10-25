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
LOOKBACK_PERIOD = 80  #
TOTAL_LIMIT = LOOKBACK_PERIOD + 20  # Лимит свечей с запасом для корректного расчета

# Сколько топовых лидеров/аутсайдеров выводить в лог
TOP_N = 15

# ====== Screener logic (Ernie Chan Cross-Sectional Momentum) ===========

def calculate_momentum_return(closes: pd.Series, lookback: int = LOOKBACK_PERIOD) -> float:
    """Расчет доходности за период lookback (в %): (Price_t - Price_{t-lookback}) / Price_{t-lookback}."""
    try:
        current_close = closes.iloc[-1]
        past_close = closes.iloc[-(lookback + 1)]
        if past_close == 0 or pd.isna(past_close):
            return 0.0
        return float((current_close - past_close) / past_close) * 100.0
    except Exception as e:
        print(f"Error calculate momentum return: {e}")
        return 0.0

def get_large_cap(exchange: Exchange) -> list[str]:
    symbols = get_swap_symbols(exchange)
    return filter_symbols_by_volume(exchange, symbols, LARGE_CAP_VOLUME_USDT)

def get_mid_cap(exchange: Exchange) -> list[str]:
    symbols = get_swap_symbols(exchange)
    return filter_symbols_by_volume_range(exchange, symbols, MID_CAP_VOLUME_USDT, LARGE_CAP_VOLUME_USDT)

def get_small_cap(exchange: Exchange) -> list[str]:
    symbols = get_swap_symbols(exchange)
    return filter_symbols_by_volume_range(exchange, symbols, SMALL_CAP_VOLUME_USDT, MID_CAP_VOLUME_USDT)

def get_lowest_cap(exchange: Exchange) -> list[str]:
    symbols = get_swap_symbols(exchange)
    return filter_symbols_by_volume_range(exchange, symbols, LOWEST_CAP_VOLUME_USDT, SMALL_CAP_VOLUME_USDT)

def rank_and_print_coins_by_return(symbols_by_volume: list[str], exchange: Exchange):
    """
    Кросс-секционное ранжирование списка монет по метрике относительной силы.
    """
    print(f"Find symbols for processing: {len(symbols_by_volume)}")
    metrics_list: list[dict] = []

    for symbol in symbols_by_volume:
        candles: pd.DataFrame = get_ohlc_data(symbol, exchange, TIMEFRAME, TOTAL_LIMIT)
        closes: pd.Series = candles['close']
        return_80_pct = calculate_momentum_return(closes, lookback=80)
        return_40_pct = calculate_momentum_return(closes, lookback=40)
        return_20_pct = calculate_momentum_return(closes, lookback=20)
        return_10_pct = calculate_momentum_return(closes, lookback=10)
        metrics_list.append({
            "symbol": symbol,
            "return_80_pct": return_80_pct,
            "return_40_pct": return_40_pct,
            "return_20_pct": return_20_pct,
            "return_10_pct": return_10_pct
        })

    if not metrics_list:
        print("No metrics collected.")
        return

    metrics_list.sort(key=lambda t: t.get("return_80_pct"), reverse=True)

    print(f"========== Top Cross-Sectional Leaders (LONG candidates) ==========")
    for ticker_metric in metrics_list[:TOP_N]:
        print(ticker_metric)

    print(f"========== Bottom Cross-Sectional Laggards (SHORT candidates) ==========")
    for ticker_metric in metrics_list[-TOP_N:]:
        print(ticker_metric)

def run():
    exchange = get_bybit_swap_exchange()

    print("===== Large Cap Volume =====")
    rank_and_print_coins_by_return(get_large_cap(exchange), exchange)

    print("\n===== Mid Cap Volume =====")
    rank_and_print_coins_by_return(get_mid_cap(exchange), exchange)

    print("\n===== Small Cap Volume =====")
    rank_and_print_coins_by_return(get_small_cap(exchange), exchange)

    print("\n===== Lowest Cap Volume =====")
    rank_and_print_coins_by_return(get_lowest_cap(exchange), exchange)


if __name__ == "__main__":
    run()