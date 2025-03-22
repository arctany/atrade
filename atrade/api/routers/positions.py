from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from datetime import datetime

from atrade.api.schemas.base import Position, PositionCreate
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

@router.post("/", response_model=Position)
def create_position(position: PositionCreate, db: Session = Depends(get_db)):
    """创建持仓记录"""
    db_position = database.Position(
        symbol=position.symbol,
        quantity=position.quantity,
        entry_price=position.entry_price,
        current_price=position.current_price,
        profit_loss=position.profit_loss,
        profit_loss_pct=position.profit_loss_pct
    )
    db.add(db_position)
    db.commit()
    db.refresh(db_position)
    return db_position

@router.get("/", response_model=List[Position])
def get_positions(db: Session = Depends(get_db)):
    """获取所有持仓"""
    return db.query(database.Position).all()

@router.get("/{symbol}", response_model=Position)
def get_position(symbol: str, db: Session = Depends(get_db)):
    """获取指定股票的持仓"""
    position = db.query(database.Position).filter(database.Position.symbol == symbol).first()
    if position is None:
        raise HTTPException(status_code=404, detail="Position not found")
    return position

@router.put("/{symbol}", response_model=Position)
def update_position(
    symbol: str,
    position: PositionCreate,
    db: Session = Depends(get_db)
):
    """更新持仓信息"""
    db_position = db.query(database.Position).filter(database.Position.symbol == symbol).first()
    if db_position is None:
        raise HTTPException(status_code=404, detail="Position not found")
    
    for field, value in position.dict().items():
        setattr(db_position, field, value)
    
    db.commit()
    db.refresh(db_position)
    return db_position 