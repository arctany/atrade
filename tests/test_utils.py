import pytest
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from atrade.core.utils import (
    calculate_returns,
    calculate_sharpe_ratio,
    calculate_max_drawdown,
    calculate_position_size,
    format_currency,
    format_percentage,
    validate_symbol,
    parse_date_range,
    calculate_volatility
)

def test_calculate_returns():
    # Test with simple price series
    prices = pd.Series([100, 102, 101, 103, 105])
    returns = calculate_returns(prices)
    assert len(returns) == len(prices) - 1
    assert returns[0] == 0.02  # (102 - 100) / 100
    assert returns[1] == -0.0098  # (101 - 102) / 102
    assert returns[2] == 0.0198  # (103 - 101) / 101
    assert returns[3] == 0.0194  # (105 - 103) / 103

def test_calculate_sharpe_ratio():
    # Test with sample returns
    returns = pd.Series([0.02, -0.01, 0.03, -0.02, 0.01])
    risk_free_rate = 0.02
    sharpe = calculate_sharpe_ratio(returns, risk_free_rate)
    assert isinstance(sharpe, float)
    assert sharpe > 0  # Should be positive for this sample data

def test_calculate_max_drawdown():
    # Test with sample price series
    prices = pd.Series([100, 102, 98, 103, 95, 105])
    max_dd = calculate_max_drawdown(prices)
    assert isinstance(max_dd, float)
    assert max_dd >= 0
    assert max_dd <= 1
    # Max drawdown should be (98 - 102) / 102 ≈ 0.0392
    assert abs(max_dd - 0.0392) < 0.0001

def test_calculate_position_size():
    # Test position size calculation
    portfolio_value = 100000
    risk_per_trade = 0.02  # 2% risk per trade
    stop_loss = 0.05  # 5% stop loss
    
    position_size = calculate_position_size(portfolio_value, risk_per_trade, stop_loss)
    assert isinstance(position_size, float)
    assert position_size > 0
    # Should be (100000 * 0.02) / 0.05 = 40000
    assert abs(position_size - 40000) < 0.01

def test_format_currency():
    # Test currency formatting
    assert format_currency(1234.56) == "$1,234.56"
    assert format_currency(1000000) == "$1,000,000.00"
    assert format_currency(0) == "$0.00"
    assert format_currency(-1234.56) == "-$1,234.56"

def test_format_percentage():
    # Test percentage formatting
    assert format_percentage(0.1234) == "12.34%"
    assert format_percentage(0.05) == "5.00%"
    assert format_percentage(0) == "0.00%"
    assert format_percentage(-0.1234) == "-12.34%"

def test_validate_symbol():
    # Test symbol validation
    assert validate_symbol("AAPL") is True
    assert validate_symbol("GOOGL") is True
    assert validate_symbol("MSFT") is True
    assert validate_symbol("") is False
    assert validate_symbol("123") is False
    assert validate_symbol("A") is False
    assert validate_symbol("AAPL123") is False

def test_parse_date_range():
    # Test date range parsing
    start_date = "2023-01-01"
    end_date = "2023-12-31"
    
    start, end = parse_date_range(start_date, end_date)
    assert isinstance(start, datetime)
    assert isinstance(end, datetime)
    assert start.year == 2023
    assert start.month == 1
    assert start.day == 1
    assert end.year == 2023
    assert end.month == 12
    assert end.day == 31
    
    # Test with relative dates
    start, end = parse_date_range("1d", "now")
    assert isinstance(start, datetime)
    assert isinstance(end, datetime)
    assert end > start
    assert (end - start).days == 1

def test_calculate_volatility():
    # Test volatility calculation
    returns = pd.Series([0.02, -0.01, 0.03, -0.02, 0.01])
    volatility = calculate_volatility(returns)
    assert isinstance(volatility, float)
    assert volatility > 0
    
    # Test with different window sizes
    volatility_3d = calculate_volatility(returns, window=3)
    assert isinstance(volatility_3d, float)
    assert volatility_3d > 0
    assert volatility_3d != volatility  # Should be different from default window

def test_error_handling():
    # Test error handling for invalid inputs
    with pytest.raises(ValueError):
        calculate_returns(pd.Series([]))
    
    with pytest.raises(ValueError):
        calculate_sharpe_ratio(pd.Series([]), 0.02)
    
    with pytest.raises(ValueError):
        calculate_max_drawdown(pd.Series([]))
    
    with pytest.raises(ValueError):
        calculate_position_size(-100000, 0.02, 0.05)
    
    with pytest.raises(ValueError):
        parse_date_range("invalid", "2023-12-31")
    
    with pytest.raises(ValueError):
        calculate_volatility(pd.Series([])) 