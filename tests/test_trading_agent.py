import pytest
import pandas as pd
from datetime import datetime, timedelta
from components.trading_agent import TradingAgent

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
        }
    }

@pytest.fixture
def trading_agent(sample_market_data, sample_positions):
    return TradingAgent(
        data=sample_market_data,
        positions=sample_positions,
        capital=100000
    )

def test_initialization(trading_agent):
    assert trading_agent.capital == 100000
    assert len(trading_agent.positions) == 1
    assert 'AAPL' in trading_agent.positions

def test_update_data(trading_agent):
    new_data = pd.DataFrame({
        'date': [datetime.now()],
        'open': [110],
        'high': [115],
        'low': [105],
        'close': [110],
        'volume': [2000]
    })
    trading_agent.update_data(new_data)
    assert len(trading_agent.data) > 1

def test_update_positions(trading_agent):
    new_positions = {
        'GOOGL': {
            'quantity': 5,
            'entry_price': 200,
            'current_price': 210,
            'profit_loss': 50,
            'profit_loss_pct': 0.05
        }
    }
    trading_agent.update_positions(new_positions)
    assert len(trading_agent.positions) == 1
    assert 'GOOGL' in trading_agent.positions

def test_calculate_position_value(trading_agent):
    position_value = trading_agent.calculate_position_value('AAPL')
    assert position_value == 1050  # 10 shares * 105 current price

def test_calculate_total_value(trading_agent):
    total_value = trading_agent.calculate_total_value()
    assert total_value == 101050  # 100000 capital + 1050 position value 