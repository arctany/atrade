import pytest
import pandas as pd
from datetime import datetime, timedelta
from components.risk_manager import RiskManager

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
def sample_positions():
    return {
        'AAPL': {
            'quantity': 10,
            'entry_price': 100,
            'current_price': 105,
            'profit_loss': 50,
            'profit_loss_pct': 0.05
        },
        'GOOGL': {
            'quantity': 5,
            'entry_price': 200,
            'current_price': 190,
            'profit_loss': -50,
            'profit_loss_pct': -0.05
        }
    }

@pytest.fixture
def risk_config():
    return {
        'max_position_size': 0.1,
        'max_drawdown': 0.1,
        'max_leverage': 2.0,
        'max_correlation': 0.7,
        'stop_loss_pct': 0.02,
        'take_profit_pct': 0.05
    }

@pytest.fixture
def risk_manager(sample_market_data, sample_positions, risk_config):
    return RiskManager(
        data=sample_market_data,
        positions=sample_positions,
        capital=100000,
        config=risk_config
    )

def test_initialization(risk_manager):
    assert risk_manager.capital == 100000
    assert len(risk_manager.positions) == 2
    assert risk_manager.config['max_position_size'] == 0.1

def test_calculate_position_risk(risk_manager):
    risk = risk_manager.calculate_position_risk('AAPL')
    assert isinstance(risk, dict)
    assert 'value_at_risk' in risk
    assert 'beta' in risk
    assert 'correlation' in risk

def test_check_position_limits(risk_manager):
    # Test position size limit
    assert risk_manager.check_position_limits('AAPL')  # 10% position size is within limit
    
    # Test with oversized position
    risk_manager.positions['AAPL']['quantity'] = 1000
    assert not risk_manager.check_position_limits('AAPL')

def test_calculate_portfolio_risk(risk_manager):
    portfolio_risk = risk_manager.calculate_portfolio_risk()
    assert isinstance(portfolio_risk, dict)
    assert 'total_value' in portfolio_risk
    assert 'total_risk' in portfolio_risk
    assert 'diversification_score' in portfolio_risk

def test_check_stop_loss(risk_manager):
    # Test stop loss trigger
    risk_manager.positions['AAPL']['current_price'] = 98  # 2% below entry
    assert risk_manager.check_stop_loss('AAPL')

    # Test stop loss not triggered
    risk_manager.positions['AAPL']['current_price'] = 101
    assert not risk_manager.check_stop_loss('AAPL')

def test_check_take_profit(risk_manager):
    # Test take profit trigger
    risk_manager.positions['AAPL']['current_price'] = 105  # 5% above entry
    assert risk_manager.check_take_profit('AAPL')

    # Test take profit not triggered
    risk_manager.positions['AAPL']['current_price'] = 103
    assert not risk_manager.check_take_profit('AAPL') 