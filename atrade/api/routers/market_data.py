from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from datetime import datetime

from atrade.api.schemas.base import MarketData, MarketDataCreate
from atrade.database import SessionLocal
import atrade.database as database

router = APIRouter()

# 依赖项
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/", response_model=MarketData)
def create_market_data(market_data: MarketDataCreate, db: Session = Depends(get_db)):
    """创建市场数据"""
    db_market_data = database.MarketData(
        symbol=market_data.symbol,
        timestamp=market_data.timestamp,
        open=market_data.open,
        high=market_data.high,
        low=market_data.low,
        close=market_data.close,
        volume=market_data.volume
    )
    db.add(db_market_data)
    db.commit()
    db.refresh(db_market_data)
    return db_market_data

@router.get("/{symbol}", response_model=List[MarketData])
def get_market_data(
    symbol: str,
    start_time: datetime,
    end_time: datetime,
    db: Session = Depends(get_db)
):
    """获取指定股票的市场数据"""
    market_data = db.query(database.MarketData).filter(
        database.MarketData.symbol == symbol,
        database.MarketData.timestamp >= start_time,
        database.MarketData.timestamp <= end_time
    ).all()
    return market_data

@router.get("/latest/{symbol}", response_model=MarketData)
def get_latest_market_data(symbol: str, db: Session = Depends(get_db)):
    """获取指定股票的最新市场数据"""
    market_data = db.query(database.MarketData).filter(
        database.MarketData.symbol == symbol
    ).order_by(database.MarketData.timestamp.desc()).first()
    
    if market_data is None:
        raise HTTPException(status_code=404, detail="Market data not found")
    return market_data 