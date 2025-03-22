import pytest
from datetime import datetime, timedelta
from sqlalchemy.orm import Session

from atrade.models.trade import Trade
from atrade.models.position import Position
from atrade.models.portfolio import Portfolio
from atrade.core.trading import TradingEngine
from atrade.core.risk import RiskManager
from atrade.core.strategy import Strategy

def test_trading_engine_initialization(db_session: Session):
    engine = TradingEngine(db_session)
    assert engine is not None
    assert engine.risk_manager is not None
    assert engine.strategy is not None

def test_risk_manager_validation(db_session: Session):
    risk_manager = RiskManager()
    
    # Test position size validation
    assert risk_manager.validate_position_size(1000, 10000) is True
    assert risk_manager.validate_position_size(20000, 10000) is False
    
    # Test drawdown validation
    assert risk_manager.validate_drawdown(0.05) is True
    assert risk_manager.validate_drawdown(0.15) is False
    
    # Test leverage validation
    assert risk_manager.validate_leverage(1.5) is True
    assert risk_manager.validate_leverage(3.0) is False

def test_strategy_signals(db_session: Session):
    strategy = Strategy()
    
    # Test moving average crossover
    prices = [100, 102, 101, 103, 105, 104, 106, 107]
    signals = strategy.generate_signals(prices)
    assert len(signals) > 0
    
    # Test RSI signals
    rsi_signals = strategy.calculate_rsi_signals(prices)
    assert len(rsi_signals) > 0

def test_trade_execution(db_session: Session):
    engine = TradingEngine(db_session)
    
    # Create test trade
    trade_data = {
        "symbol": "AAPL",
        "side": "buy",
        "quantity": 10,
        "price": 150.0
    }
    
    # Execute trade
    trade = engine.execute_trade(**trade_data)
    assert trade is not None
    assert trade.symbol == trade_data["symbol"]
    assert trade.side == trade_data["side"]
    assert trade.quantity == trade_data["quantity"]
    assert trade.price == trade_data["price"]
    
    # Verify position update
    position = db_session.query(Position).filter_by(symbol=trade_data["symbol"]).first()
    assert position is not None
    assert position.quantity == trade_data["quantity"]
    assert position.average_price == trade_data["price"]

def test_portfolio_update(db_session: Session):
    engine = TradingEngine(db_session)
    
    # Create initial portfolio
    portfolio = Portfolio(
        total_value=100000,
        cash_balance=100000
    )
    db_session.add(portfolio)
    db_session.commit()
    
    # Execute trade
    trade_data = {
        "symbol": "AAPL",
        "side": "buy",
        "quantity": 10,
        "price": 150.0
    }
    trade = engine.execute_trade(**trade_data)
    
    # Verify portfolio update
    updated_portfolio = db_session.query(Portfolio).first()
    assert updated_portfolio.total_value == portfolio.total_value
    assert updated_portfolio.cash_balance == portfolio.cash_balance - (trade_data["quantity"] * trade_data["price"])

def test_performance_metrics(db_session: Session):
    engine = TradingEngine(db_session)
    
    # Create test trades
    trades = [
        Trade(symbol="AAPL", side="buy", quantity=10, price=150.0, timestamp=datetime.now() - timedelta(days=2)),
        Trade(symbol="AAPL", side="sell", quantity=5, price=160.0, timestamp=datetime.now() - timedelta(days=1)),
        Trade(symbol="AAPL", side="sell", quantity=5, price=170.0, timestamp=datetime.now())
    ]
    db_session.add_all(trades)
    db_session.commit()
    
    # Calculate performance metrics
    metrics = engine.calculate_performance_metrics()
    assert metrics["total_return"] > 0
    assert metrics["sharpe_ratio"] is not None
    assert metrics["max_drawdown"] >= 0

def test_risk_management(db_session: Session):
    engine = TradingEngine(db_session)
    
    # Test position size limits
    large_trade = {
        "symbol": "AAPL",
        "side": "buy",
        "quantity": 1000,
        "price": 150.0
    }
    
    with pytest.raises(ValueError):
        engine.execute_trade(**large_trade)
    
    # Test drawdown limits
    engine.risk_manager.current_drawdown = 0.15
    with pytest.raises(ValueError):
        engine.execute_trade(symbol="AAPL", side="buy", quantity=10, price=150.0)

def test_market_data_integration(db_session: Session):
    engine = TradingEngine(db_session)
    
    # Test market data retrieval
    market_data = engine.get_market_data("AAPL")
    assert market_data is not None
    assert "price" in market_data
    assert "volume" in market_data
    assert "timestamp" in market_data 