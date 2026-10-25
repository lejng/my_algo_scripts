from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from ccxt import Exchange
from statsmodels.tsa.stattools import adfuller

from src.main.exchange_helper import (
    get_ohlc_data,
    get_bybit_swap_exchange, filter_symbols_by_volume, get_swap_symbols_by_quote
)

TIMEFRAME = '1h'
TOTAL_CANDLE_LIMIT = 600
LARGE_CAP_VOLUME_USDT = 10_000_000
ADF_WINDOW = 200 # сколько свечей использовать для ADF
HALF_LIFE_WINDOW = 100
MA_PERIOD = 48 # equivalent lookback for Bollinger Bands
MA_SLOPE_WINDOW = 100
MA_SLOPE_LOOKBACK = 50
MAX_MA_SLOPE = 100#0.03

BTC_ECO_SYSTEM = ['BTC/USDT:USDT', 'BCH/USDT:USDT', 'BSV/USDT:USDT', 'STX/USDT:USDT', 'ORDI/USDT:USDT']
ETH_ECO_SYSTEM = ['ETH/USDT:USDT', 'LINK/USDT:USDT', 'ARB/USDT:USDT', 'OP/USDT:USDT', 'STRK/USDT:USDT', 'AAVE/USDT:USDT']
# 1. AI & Compute (Инфраструктура ИИ и децентрализованные вычисления)
# Сильные корреляции на новостях сектора и трендах искусственного интеллекта.
AI_SECTOR = [
    'TAO/USDT:USDT',
    'RENDER/USDT:USDT',
    'IO/USDT:USDT',
    'NEAR/USDT:USDT'
]

# 2. Real-World Assets / RWA (Токенизация реальных активов)
# Институциональные монеты, которые часто движутся синхронно на макроданных.
RWA_SECTOR = [
    'ONDO/USDT:USDT',
    'PENDLE/USDT:USDT',
    'CFG/USDT:USDT'
]

# 3. DePIN (Децентрализованные физические сети)
# Инфраструктурные проекты реального железа и пропускной способности.
DePIN_SECTOR = [
    'RENDER/USDT:USDT',
    'AKT/USDT:USDT',
    'HNT/USDT:USDT',
    'FIL/USDT:USDT'
]

# 4. Layer-1 / Competitors (Основные конкуренты Ethereum)
# Крупные альтернативные блокчейны первого уровня. Между ними постоянно перетекает капитал.
L1_ALT_SECTOR = [
    'SOL/USDT:USDT',
    'AVAX/USDT:USDT',
    'NEAR/USDT:USDT',
    'SUI/USDT:USDT',
    'APT/USDT:USDT'
]

# 5. Perp DEXs (Децентрализованные деривативные биржи)
# Токен-экономика этих проектов завязана на объемы торгов и комиссии в DeFi.
PERP_DEX_SECTOR = [
    'DYDX/USDT:USDT',
    'GMX/USDT:USDT',
    'SNX/USDT:USDT',
    'JUP/USDT:USDT'
]

# 6. Meme Heavyweights (Крупные мемкоины)
# Чистая спекулятивная ликвидность. Часто коррелируют в моменты общего аппетита к риску.
MEME_SECTOR = [
    'DOGE/USDT:USDT',
    'SHIB1000/USDT:USDT',
    '1000PEPE/USDT:USDT',
    'WIF/USDT:USDT',
    '1000BONK/USDT:USDT'
]

SECTORS = {
    'btc eco system': BTC_ECO_SYSTEM,
    'eth eco system': ETH_ECO_SYSTEM,
    'AI sector': AI_SECTOR,
    'RWA Sector': RWA_SECTOR,
    'DePin sector': DePIN_SECTOR,
    'L1 altcoins sector': L1_ALT_SECTOR,
    'perp dex sector': PERP_DEX_SECTOR,
    'meme sector': MEME_SECTOR,
}


@dataclass
class PairProperty:
    symbol_a: str
    symbol_b: str
    correlation: float
    adf_p_value: float
    half_life: float
    ma_slope: float
    zscore: float

    def get_print_format(self) -> str:
        return (
            f"pair: {self.symbol_a} / {self.symbol_b} | "
            f"corr: {self.correlation:.4f} | "
            f"ADF: {self.adf_p_value:.4f} | "
            f"half-life: {self.half_life:.1f} candles | "
            f"MA slope: {self.ma_slope:.4f}% | "
            f"Z-score: {self.zscore:.2f}"
        )

def calculate_adf_p_value(ratio, window=ADF_WINDOW):
    """ADF p-value для ratio."""
    ratio = ratio.dropna().tail(window)

    if len(ratio) < 50:
        return np.nan

    try:
        return adfuller(ratio, autolag="AIC")[1]
    except Exception:
        return np.nan

def calculate_correlation(closes_a: pd.Series, closes_b: pd.Series, use_returns: bool = True) -> float:
    if use_returns:
        s1 = closes_a.pct_change().dropna()
        s2 = closes_b.pct_change().dropna()
        return float(s1.corr(s2))
    return float(closes_a.corr(closes_b))

def calculate_half_life(ratio, window=HALF_LIFE_WINDOW):
    """Оценка half-life mean reversion для ratio."""
    ratio = ratio.dropna().tail(window)

    if len(ratio) < 30:
        return np.nan

    lagged = ratio.shift(1)
    delta = ratio - lagged

    data = pd.concat([delta, lagged], axis=1).dropna()

    if len(data) < 20:
        return np.nan

    y = data.iloc[:, 0].values
    x = data.iloc[:, 1].values

    beta = np.polyfit(x, y, 1)[0]

    if beta >= 0:
        return np.inf

    half_life = -np.log(2) / beta

    return half_life

def calculate_zscore(ratio, window=MA_PERIOD):
    """Текущее отклонение ratio от локального среднего."""
    ratio = ratio.dropna()

    if len(ratio) < window:
        return np.nan

    recent = ratio.tail(window)

    mean = recent.mean()
    std = recent.std()

    if std == 0:
        return np.nan

    return (recent.iloc[-1] - mean) / std

# good range when slope is near 0
def calculate_ma_slope(
    ratio,
    ma_window=MA_SLOPE_WINDOW,
    lookback=MA_SLOPE_LOOKBACK
) -> float:
    """Наклон MA за последние lookback свечей через линейную регрессию."""
    ratio = ratio.dropna()

    if len(ratio) < ma_window + lookback:
        return np.nan

    ma = ratio.rolling(ma_window).mean().dropna()
    recent_ma = ma.tail(lookback)

    if len(recent_ma) < lookback:
        return np.nan

    x = np.arange(len(recent_ma))
    y = recent_ma.values

    slope = np.polyfit(x, y, 1)[0]

    # Нормализуем относительно текущего значения MA,
    # чтобы результат был в процентах на одну свечу.
    current_ma = recent_ma.iloc[-1]

    if current_ma == 0:
        return np.nan

    return (slope / current_ma) * 100

def calculate_pair_property(
        symbol_a: str,
        symbol_b: str,
        closes_a: pd.Series,
        closes_b: pd.Series
) -> PairProperty:
    # Вычисляем всё напрямую без concat
    ratio = closes_a / closes_b
    correlation = calculate_correlation(closes_a, closes_b)
    adf_p_value = calculate_adf_p_value(ratio)
    half_life = calculate_half_life(ratio)
    zscore = calculate_zscore(ratio)
    ma_slope = calculate_ma_slope(ratio)

    return PairProperty(
            symbol_a=symbol_a,
            symbol_b=symbol_b,
            correlation=correlation,
            adf_p_value=adf_p_value,
            half_life=half_life,
            ma_slope=ma_slope,
            zscore=zscore,
        )

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

def find_possible_pairs(symbols: list[str], exchange: Exchange) -> list[PairProperty]:
    # 1. Скачиваем свечи один раз
    print(f"Total {len(symbols)} symbols...")
    closes_dict: dict[str, pd.Series] = fetch_all_closes(symbols, exchange)
    valid_symbols: list[str] = list(closes_dict.keys())

    all_combinations = list(combinations(valid_symbols, 2))
    total_pairs = len(all_combinations)
    print(f"Analyzing {total_pairs} pairs combinations...")

    results: list[PairProperty] = []

    # 2. Быстрый перебор комбинаций из памяти
    for i, (symbol_a, symbol_b) in enumerate(all_combinations, start=1):
        pair_prop = calculate_pair_property(
            symbol_a,
            symbol_b,
            closes_dict[symbol_a],
            closes_dict[symbol_b]
        )
        results.append(pair_prop)

        if i % 10 == 0 or i == total_pairs:
            print(f"Processed {i}/{total_pairs} pairs")

    return results

def pair_sort_key(x: PairProperty):
    half_life = x.half_life

    if not np.isfinite(half_life):
        half_life = float("inf")

    return (
        abs(x.ma_slope),
        -abs(x.zscore),
        half_life,
        x.adf_p_value,
    )

def print_all_possible_pairs(symbols: list[str], exchange: Exchange):
    pair_properties = find_possible_pairs(symbols, exchange)
    pair_properties.sort(key=pair_sort_key)
    print("\n===== RESULTS =====")
    for pair in pair_properties:
        if abs(pair.ma_slope) <= MAX_MA_SLOPE:
            print(pair.get_print_format())

def relation_strength_to_symbol(exchange: Exchange, symbol_main: str = 'BTC/USDT:USDT', ma_fast_period = 20, ma_slow_period = 50):
    symbols = get_swap_symbols_by_quote(exchange)
    symbols = filter_symbols_by_volume(exchange, symbols, LARGE_CAP_VOLUME_USDT)
    symbols.remove(symbol_main)
    closes_symbol_to_compare = fetch_all_closes([symbol_main], exchange)[symbol_main]
    closes_dict: dict[str, pd.Series] = fetch_all_closes(symbols, exchange)
    valid_symbols: list[str] = list(closes_dict.keys())
    strong: list[str] = []
    weak: list[str] = []
    for symbol in valid_symbols:
        ratio = closes_dict[symbol] / closes_symbol_to_compare
        current_ratio_price = ratio.iloc[-1]
        ma_fast = ratio.rolling(ma_fast_period).mean()
        ma_slow = ratio.rolling(ma_slow_period).mean()
        if ma_slow.iloc[-1] < ma_fast.iloc[-1] < current_ratio_price:
            strong.append(symbol)
        if ma_slow.iloc[-1] > ma_fast.iloc[-1] > current_ratio_price:
            weak.append(symbol)
    print(f"Strong symbols to {symbol_main}:")
    print(strong)
    print(f"Weak symbols to {symbol_main}:")
    print(weak)

def print_all_combination_to_symbol(symbol_a: str, exchange: Exchange):
    symbols = get_swap_symbols_by_quote(exchange)
    symbols = filter_symbols_by_volume(exchange, symbols, LARGE_CAP_VOLUME_USDT)
    symbols.remove(symbol_a)
    closes_symbol_a = fetch_all_closes([symbol_a], exchange)[symbol_a]
    closes_dict: dict[str, pd.Series] = fetch_all_closes(symbols, exchange)
    valid_symbols: list[str] = list(closes_dict.keys())
    results: list[PairProperty] = []
    for symbol_b in valid_symbols:
        if symbol_a == symbol_b:
            continue
        pair_prop = calculate_pair_property(
            symbol_a,
            symbol_b,
            closes_symbol_a,
            closes_dict[symbol_b],
        )
        results.append(pair_prop)
    results.sort(key=pair_sort_key)
    print(f"===== All combinations to {symbol_a}=====")
    for pair in results:
        print(pair.get_print_format())

def print_large_cap_pairs(exchange: Exchange):
    symbols = get_swap_symbols_by_quote(exchange)
    symbols = filter_symbols_by_volume(exchange, symbols, LARGE_CAP_VOLUME_USDT)
    print("========= large cap pairs ============")
    print_all_possible_pairs(symbols, exchange)

def print_pairs_per_sectors(exchange: Exchange):
    for sector_name, sector_symbols in SECTORS.items():
        print(f"===== {sector_name} =====")
        print_all_possible_pairs(sector_symbols, exchange)

def run():
    exchange = get_bybit_swap_exchange()
    # ====== Basic screener ========== #
    relation_strength_to_symbol(exchange)
    print_pairs_per_sectors(exchange)
    print_large_cap_pairs(exchange)
    # example how to check some concreate situations
    #print_all_combination_to_symbol("PUFFER/USDT:USDT", exchange)
    # print_all_possible_pairs(['BTC/USDT:USDT', 'BTC/USDC:USDC'], exchange)
    # print_all_possible_pairs(['ETH/USDT:USDT', 'ETH/USDC:USDC'], exchange)


if __name__ == "__main__":
    run()