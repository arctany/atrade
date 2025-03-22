import pytest
from datetime import datetime
from sqlalchemy.orm import Session

from atrade.models.user import User
from atrade.models.trade import Trade
from atrade.models.position import Position
from atrade.models.portfolio import Portfolio
from atrade.models.watchlist import Watchlist
from atrade.models.alert import Alert
from atrade.core.security import get_password_hash

def test_user_model(db_session: Session):
    # Create test user
    user = User(
        username="testuser",
        email="test@example.com",
        hashed_password=get_password_hash("testpassword123")
    )
    db_session.add(user)
    db_session.commit()
    
    # Test user retrieval
    retrieved_user = db_session.query(User).filter_by(username="testuser").first()
    assert retrieved_user is not None
    assert retrieved_user.username == "testuser"
    assert retrieved_user.email == "test@example.com"
    assert retrieved_user.is_active is True
    assert retrieved_user.is_superuser is False

def test_trade_model(db_session: Session):
    # Create test trade
    trade = Trade(
        symbol="AAPL",
        side="buy",
        quantity=10,
        price=150.0,
        timestamp=datetime.now()
    )
    db_session.add(trade)
    db_session.commit()
    
    # Test trade retrieval
    retrieved_trade = db_session.query(Trade).filter_by(symbol="AAPL").first()
    assert retrieved_trade is not None
    assert retrieved_trade.symbol == "AAPL"
    assert retrieved_trade.side == "buy"
    assert retrieved_trade.quantity == 10
    assert retrieved_trade.price == 150.0

def test_position_model(db_session: Session):
    # Create test position
    position = Position(
        symbol="AAPL",
        quantity=100,
        average_price=150.0,
        current_price=160.0,
        unrealized_pnl=1000.0
    )
    db_session.add(position)
    db_session.commit()
    
    # Test position retrieval
    retrieved_position = db_session.query(Position).filter_by(symbol="AAPL").first()
    assert retrieved_position is not None
    assert retrieved_position.symbol == "AAPL"
    assert retrieved_position.quantity == 100
    assert retrieved_position.average_price == 150.0
    assert retrieved_position.current_price == 160.0
    assert retrieved_position.unrealized_pnl == 1000.0

def test_portfolio_model(db_session: Session):
    # Create test portfolio
    portfolio = Portfolio(
        total_value=100000.0,
        cash_balance=50000.0,
        margin_balance=100000.0,
        margin_used=50000.0
    )
    db_session.add(portfolio)
    db_session.commit()
    
    # Test portfolio retrieval
    retrieved_portfolio = db_session.query(Portfolio).first()
    assert retrieved_portfolio is not None
    assert retrieved_portfolio.total_value == 100000.0
    assert retrieved_portfolio.cash_balance == 50000.0
    assert retrieved_portfolio.margin_balance == 100000.0
    assert retrieved_portfolio.margin_used == 50000.0

def test_watchlist_model(db_session: Session):
    # Create test user
    user = User(
        username="testuser",
        email="test@example.com",
        hashed_password=get_password_hash("testpassword123")
    )
    db_session.add(user)
    db_session.commit()
    
    # Create test watchlist
    watchlist = Watchlist(
        user_id=user.id,
        name="My Watchlist",
        symbols=["AAPL", "GOOGL", "MSFT"]
    )
    db_session.add(watchlist)
    db_session.commit()
    
    # Test watchlist retrieval
    retrieved_watchlist = db_session.query(Watchlist).filter_by(user_id=user.id).first()
    assert retrieved_watchlist is not None
    assert retrieved_watchlist.name == "My Watchlist"
    assert "AAPL" in retrieved_watchlist.symbols
    assert "GOOGL" in retrieved_watchlist.symbols
    assert "MSFT" in retrieved_watchlist.symbols

def test_alert_model(db_session: Session):
    # Create test user
    user = User(
        username="testuser",
        email="test@example.com",
        hashed_password=get_password_hash("testpassword123")
    )
    db_session.add(user)
    db_session.commit()
    
    # Create test alert
    alert = Alert(
        user_id=user.id,
        symbol="AAPL",
        condition="price_above",
        threshold=200.0,
        is_active=True,
        created_at=datetime.now()
    )
    db_session.add(alert)
    db_session.commit()
    
    # Test alert retrieval
    retrieved_alert = db_session.query(Alert).filter_by(user_id=user.id).first()
    assert retrieved_alert is not None
    assert retrieved_alert.symbol == "AAPL"
    assert retrieved_alert.condition == "price_above"
    assert retrieved_alert.threshold == 200.0
    assert retrieved_alert.is_active is True

def test_model_relationships(db_session: Session):
    # Create test user
    user = User(
        username="testuser",
        email="test@example.com",
        hashed_password=get_password_hash("testpassword123")
    )
    db_session.add(user)
    db_session.commit()
    
    # Create related models
    portfolio = Portfolio(
        user_id=user.id,
        total_value=100000.0,
        cash_balance=50000.0
    )
    db_session.add(portfolio)
    
    trade = Trade(
        user_id=user.id,
        symbol="AAPL",
        side="buy",
        quantity=10,
        price=150.0
    )
    db_session.add(trade)
    
    watchlist = Watchlist(
        user_id=user.id,
        name="My Watchlist",
        symbols=["AAPL", "GOOGL"]
    )
    db_session.add(watchlist)
    
    db_session.commit()
    
    # Test relationships
    assert user.portfolio is not None
    assert user.portfolio.total_value == 100000.0
    assert len(user.trades) == 1
    assert user.trades[0].symbol == "AAPL"
    assert len(user.watchlists) == 1
    assert user.watchlists[0].name == "My Watchlist" 