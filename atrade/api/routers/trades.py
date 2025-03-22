from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from datetime import datetime

from atrade.api.schemas.base import Trade, TradeCreate
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

@router.post("/", response_model=Trade)
def create_trade(trade: TradeCreate, db: Session = Depends(get_db)):
    """创建交易记录"""
    db_trade = database.Trade(
        symbol=trade.symbol,
        type=trade.type,
        quantity=trade.quantity,
        price=trade.price,
        timestamp=trade.timestamp,
        strategy=trade.strategy,
        trade_metadata=trade.trade_metadata
    )
    db.add(db_trade)
    db.commit()
    db.refresh(db_trade)
    return db_trade

@router.get("/", response_model=List[Trade])
def get_trades(
    start_time: datetime,
    end_time: datetime,
    symbol: str = None,
    db: Session = Depends(get_db)
):
    """获取交易记录"""
    query = db.query(database.Trade).filter(
        database.Trade.timestamp >= start_time,
        database.Trade.timestamp <= end_time
    )
    
    if symbol:
        query = query.filter(database.Trade.symbol == symbol)
    
    return query.all()

@router.get("/{trade_id}", response_model=Trade)
def get_trade(trade_id: int, db: Session = Depends(get_db)):
    """获取指定交易记录"""
    trade = db.query(database.Trade).filter(database.Trade.id == trade_id).first()
    if trade is None:
        raise HTTPException(status_code=404, detail="Trade not found")
    return trade 