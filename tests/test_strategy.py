import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from components.strategy import Strategy

@pytest.fixture
def sample_market_data():
    dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='D')
    data = pd.DataFrame({
        'date': dates,
        'open': [100] * len(dates),
        'high': [105] * len(dates),
        'low': [95] * len(dates),
        'close': [100] * len(dates),
        'volume': [1000] * len(dates)
    })
    return data

@pytest.fixture
def strategy_config():
    return {
        'name': 'Test Strategy',
        'symbols': ['AAPL', 'GOOGL'],
        'timeframe': '1D',
        'indicators': {
            'sma': [20, 50],
            'rsi': 14,
            'macd': {'fast': 12, 'slow': 26, 'signal': 9}
        },
        'parameters': {
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'volume_threshold': 1000
        }
    }

@pytest.fixture
def strategy(sample_market_data, strategy_config):
    return Strategy(
        data=sample_market_data,
        config=strategy_config
    )

def test_initialization(strategy):
    assert strategy.name == 'Test Strategy'
    assert len(strategy.symbols) == 2
    assert strategy.timeframe == '1D'

def test_calculate_indicators(strategy):
    indicators = strategy.calculate_indicators()
    assert isinstance(indicators, dict)
    assert 'sma_20' in indicators
    assert 'sma_50' in indicators
    assert 'rsi' in indicators
    assert 'macd' in indicators

def test_generate_signals(strategy):
    signals = strategy.generate_signals()
    assert isinstance(signals, dict)
    assert 'buy' in signals
    assert 'sell' in signals
    assert 'hold' in signals

def test_backtest(strategy):
    results = strategy.backtest(
        start_date='2024-01-01',
        end_date='2024-01-10',
        initial_capital=100000
    )
    assert isinstance(results, dict)
    assert 'returns' in results
    assert 'sharpe_ratio' in results
    assert 'max_drawdown' in results
    assert 'win_rate' in results

def test_optimize_parameters(strategy):
    param_grid = {
        'rsi_oversold': [20, 30, 40],
        'rsi_overbought': [60, 70, 80],
        'volume_threshold': [800, 1000, 1200]
    }
    best_params = strategy.optimize_parameters(param_grid)
    assert isinstance(best_params, dict)
    assert 'rsi_oversold' in best_params
    assert 'rsi_overbought' in best_params
    assert 'volume_threshold' in best_params

def test_calculate_performance_metrics(strategy):
    metrics = strategy.calculate_performance_metrics()
    assert isinstance(metrics, dict)
    assert 'total_return' in metrics
    assert 'annual_return' in metrics
    assert 'volatility' in metrics
    assert 'sharpe_ratio' in metrics
    assert 'sortino_ratio' in metrics
    assert 'max_drawdown' in metrics
    assert 'win_rate' in metrics 