from pydantic import BaseModel
from datetime import datetime
from typing import Optional, Dict, Any

class MarketDataBase(BaseModel):
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

class MarketDataCreate(MarketDataBase):
    pass

class MarketData(MarketDataBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True

class PositionBase(BaseModel):
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    profit_loss: float
    profit_loss_pct: float

class PositionCreate(PositionBase):
    pass

class Position(PositionBase):
    id: int
    updated_at: datetime

    class Config:
        from_attributes = True

class TradeBase(BaseModel):
    symbol: str
    type: str
    quantity: float
    price: float
    timestamp: datetime
    strategy: str
    trade_metadata: Optional[Dict[str, Any]] = None

class TradeCreate(TradeBase):
    pass

class Trade(TradeBase):
    id: int

    class Config:
        from_attributes = True

class PerformanceMetricsBase(BaseModel):
    timestamp: datetime
    total_return: float
    annual_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    performance_metadata: Optional[Dict[str, Any]] = None

class PerformanceMetricsCreate(PerformanceMetricsBase):
    pass

class PerformanceMetrics(PerformanceMetricsBase):
    id: int

    class Config:
        from_attributes = True

class RiskMetricsBase(BaseModel):
    timestamp: datetime
    var: float
    volatility: float
    beta: float
    correlation: float
    risk_metadata: Optional[Dict[str, Any]] = None

class RiskMetricsCreate(RiskMetricsBase):
    pass

class RiskMetrics(RiskMetricsBase):
    id: int

    class Config:
        from_attributes = True 