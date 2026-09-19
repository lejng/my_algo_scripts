from dataclasses import dataclass
from typing import Optional

import ccxt
import nolds
import numpy as np
import pandas as pd
from ccxt import Exchange
from statsmodels.tsa.stattools import adfuller

LARGE_CAP_VOLUME_USDT = 10_000_000
MID_CAP_VOLUME_USDT = 2_000_000
SMALL_CAP_VOLUME_USDT = 400_000
LOWEST_CAP_VOLUME_USDT = 30_000
TIMEFRAME = '1h'
TOTAL_LIMIT = 600
Z_SCORE_THRESHOLD = 2
# range coefficients values
HURST_RANGE_VALUE = 0.4
# trend coefficients values
HURST_TREND = 0.6
# candles periods
HURST_PERIOD = 500
SQUEEZE_THRESHOLD = - 1.5

@dataclass
class TickerMetrics:
    symbol: str
    hurst: float
    z_score: float
    p_value: float
    half_live: float
    band_width_z_score: float

    def is_range(self) -> bool:
        return self.hurst < HURST_RANGE_VALUE

    def is_trend(self) -> bool:
        return self.hurst > HURST_TREND

    def is_extreme_z_score(self) -> bool:
        return abs(self.z_score) >= Z_SCORE_THRESHOLD

    def is_flat_confirmed(self) -> bool:
        if self.is_range():
            return self.p_value <= 0.05
        else:
            return False

    def is_volatility_squeeze(self) -> bool:
        """Проверяет наличие фазы сжатия волатильности."""
        if pd.isna(self.band_width_z_score) or np.isinf(self.band_width_z_score):
            return False

        return bool(self.band_width_z_score < SQUEEZE_THRESHOLD) and abs(self.z_score) <= 0.5

    def get_print_format(self) -> str:
        return f"symbol: {self.symbol}, is_flat_confirmed: {self.is_flat_confirmed()}, hurst: {self.hurst}, z-score: {self.z_score}, p-value: {self.p_value}, half-live: {self.half_live}, band width z-score: {self.band_width_z_score}"

# ====== Exchange logic ===========
def get_exchange() -> Exchange:
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

# ====== Screener logic ===========
def calculate_hurst(prices: pd.Series, window: int) -> float:
    log_ret = np.diff(np.log(prices.tail(window)))
    nvals = [16, 24, 32, 48]
    if window >= 200:
        nvals.append(64)
    if window >= 500:
        nvals.append(128)
    return nolds.dfa(log_ret, nvals=nvals, fit_exp="poly")

def calculate_z_score(prices: pd.Series, window: int = 20) -> pd.Series:
    sma = prices.rolling(window=window).mean()
    std = prices.rolling(window=window).std(ddof=0)
    z_score = (prices - sma) / std
    return z_score.fillna(0)

def calculate_adf_pvalue(prices: pd.Series) -> float:
    """
    Возвращает p-value теста Дики-Фуллера.
    Значение < 0.05 означает, что ряд стационарен на 95%.
    """
    # maxlag=1 ускоряет расчет для финансовых рядов
    result = adfuller(prices.values, maxlag=1, autolag=None)
    p_value = result[1]
    return float(p_value)

def calculate_half_life(prices: pd.Series) -> float:
    """
    Расчет времени полувозврата (Half-Life) в барах.
    """
    y = prices.values
    delta_y = np.diff(y)          # Δy_t = y_t - y_{t-1}
    y_lag = y[:-1]                # y_{t-1}

    # МНК регрессия: delta_y = lambda * y_lag + intercept
    X = np.vstack([y_lag, np.ones(len(y_lag))]).T
    lambda_coef, _ = np.linalg.lstsq(X, delta_y, rcond=None)[0]

    # Если lambda >= 0, ряд расширяется/трендует (возврата нет)
    if lambda_coef >= 0:
        return float('inf')

    half_life = -np.log(2) / lambda_coef
    return float(half_life)

def calculate_band_width_z_score(
    prices: pd.Series,
    window: int = 20,
    bb_std: float = 2.0,
    squeeze_len: int = 10,
) -> float:
    """Рассчитывает средний Z-Score ширины Полос Боллинджера за последние N свечей.

    :param prices: Ряд цен (обычно Close)
    :param window: Период скользящей средней для Боллинджера
    :param bb_std: Количество стандартных отклонений
    :param squeeze_len: За сколько последних свечей усреднять Z-Score (глубина
        накопления)
    :return: Среднее значение Z-Score за последние squeeze_len свечей
    """
    # Минимально необходимое количество свечей
    min_periods = (window * 2) + squeeze_len - 1
    if len(prices) < min_periods:
        return float("nan")

    # 1. Расчет базовых показателей цены
    sma = prices.rolling(window=window).mean()
    std = prices.rolling(window=window).std(ddof=0)

    # 2. Относительная ширина полос (Bandwidth)
    bandwidth = (2 * std * bb_std) / sma

    # 3. Среднее и отклонение самой ширины
    bw_sma = bandwidth.rolling(window=window).mean()
    bw_std = bandwidth.rolling(window=window).std(ddof=0)

    # 4. Расчет серии Z-Score
    # Защищаем от деления на 0
    bw_z_score = (bandwidth - bw_sma) / bw_std.replace(0, np.nan)

    # 5. Берем срез последних squeeze_len свечей
    recent_z_scores = bw_z_score.iloc[-squeeze_len:]

    # Если в срезе есть NaN или inf (нехватка данных)
    if recent_z_scores.isna().any() or np.isinf(recent_z_scores).any():
        return float("nan")

    # Возвращаем среднее значение Z-score за выбранный период накопления
    mean_z_score = recent_z_scores.mean()

    return float(mean_z_score)

def calculate_metrics(symbol: str, exchange: Exchange) -> Optional[TickerMetrics]:
    try:
        candles: pd.DataFrame = get_ohlc_data(symbol, exchange, TIMEFRAME, TOTAL_LIMIT)
        closes: pd.Series = candles['close']
        hurst = calculate_hurst(closes, HURST_PERIOD)
        z_series = calculate_z_score(closes, window=20)
        latest_z = float(z_series.iloc[-1])
        p_value = -1
        half_life = -1
        if hurst < HURST_RANGE_VALUE:
            p_value = calculate_adf_pvalue(closes)
            half_life = calculate_half_life(closes)
        band_width_z_score = calculate_band_width_z_score(closes)
        return TickerMetrics(symbol, hurst, latest_z, p_value, half_life, band_width_z_score)
    except Exception as e:
        print(f"Error during processing {symbol}: {e}")
        return None

def get_large_cap(exchange) -> list[str]:
    symbols = get_swap_symbols(exchange)
    return filter_symbols_by_volume(exchange, symbols, LARGE_CAP_VOLUME_USDT)

def get_mid_cap(exchange) -> list[str]:
    symbols = get_swap_symbols(exchange)
    return filter_symbols_by_volume_range(exchange, symbols, MID_CAP_VOLUME_USDT, LARGE_CAP_VOLUME_USDT)

def get_small_cap(exchange) -> list[str]:
    symbols = get_swap_symbols(exchange)
    return filter_symbols_by_volume_range(exchange, symbols, SMALL_CAP_VOLUME_USDT , MID_CAP_VOLUME_USDT)

def get_lowest_cap(exchange) -> list[str]:
    symbols = get_swap_symbols(exchange)
    return filter_symbols_by_volume_range(exchange, symbols, LOWEST_CAP_VOLUME_USDT, SMALL_CAP_VOLUME_USDT)

def find_coins(symbols_by_volume: list[str], exchange: Exchange):
    print(f"Find symbols for processing: {len(symbols_by_volume)}")
    range_list: list[TickerMetrics] = []
    trend_list: list[TickerMetrics] = []
    volatility_squeeze_list: list[TickerMetrics] = []
    for symbol in symbols_by_volume:
        ticker_metric: TickerMetrics = calculate_metrics(symbol, exchange)
        if ticker_metric:
            if ticker_metric.is_range() and ticker_metric.is_extreme_z_score():
                range_list.append(ticker_metric)
            if ticker_metric.is_trend():
                trend_list.append(ticker_metric)
            if ticker_metric.is_volatility_squeeze():
                volatility_squeeze_list.append(ticker_metric)

    print(f"========== Range list, count: {len(range_list)} =========")
    range_list.sort(key=lambda t: t.hurst)
    for ticker_metric in range_list:
        print(ticker_metric.get_print_format())

    print(f"========== Volatility squeeze list, count: {len(volatility_squeeze_list)} =========")
    volatility_squeeze_list.sort(key=lambda t: t.band_width_z_score)
    for ticker_metric in volatility_squeeze_list:
        print(ticker_metric.get_print_format())

    print(f"========== Trend list, count: {len(trend_list)} ==========")
    trend_list.sort(key=lambda t: t.hurst, reverse=True)
    for ticker_metric in trend_list:
        print(ticker_metric.get_print_format())

def run():
    exchange = get_exchange()
    print("===== Large Cap Volume =====")
    find_coins(get_large_cap(exchange), exchange)
    print("===== Mid Cap Volume =====")
    find_coins(get_mid_cap(exchange), exchange)
    print("===== Small Cap Volume =====")
    find_coins(get_small_cap(exchange), exchange)
    print("===== Lowest Cap Volume =====")
    find_coins(get_lowest_cap(exchange), exchange)

def print_symbols_stats(symbols: list[str]):
    exchange = get_exchange()
    for symbol in symbols:
        print(calculate_metrics(symbol, exchange).get_print_format())


if __name__ == "__main__":
    print_symbols_stats(['BTR/USDT:USDT', 'BABA/USDT:USDT', 'CYS/USDT:USDT', 'SPX/USDT:USDT'])
    #run()
