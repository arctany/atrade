from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from datetime import datetime

from atrade.api.schemas.base import PerformanceMetrics, PerformanceMetricsCreate
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

@router.post("/", response_model=PerformanceMetrics)
def create_performance_metrics(
    metrics: PerformanceMetricsCreate,
    db: Session = Depends(get_db)
):
    """创建性能指标"""
    db_metrics = database.PerformanceMetrics(
        timestamp=metrics.timestamp,
        total_return=metrics.total_return,
        annual_return=metrics.annual_return,
        sharpe_ratio=metrics.sharpe_ratio,
        max_drawdown=metrics.max_drawdown,
        win_rate=metrics.win_rate,
        profit_factor=metrics.profit_factor,
        performance_metadata=metrics.performance_metadata
    )
    db.add(db_metrics)
    db.commit()
    db.refresh(db_metrics)
    return db_metrics

@router.get("/", response_model=List[PerformanceMetrics])
def get_performance_metrics(
    start_time: datetime,
    end_time: datetime,
    db: Session = Depends(get_db)
):
    """获取性能指标"""
    return db.query(database.PerformanceMetrics).filter(
        database.PerformanceMetrics.timestamp >= start_time,
        database.PerformanceMetrics.timestamp <= end_time
    ).all()

@router.get("/latest", response_model=PerformanceMetrics)
def get_latest_performance_metrics(db: Session = Depends(get_db)):
    """获取最新性能指标"""
    metrics = db.query(database.PerformanceMetrics).order_by(
        database.PerformanceMetrics.timestamp.desc()
    ).first()
    
    if metrics is None:
        raise HTTPException(status_code=404, detail="Performance metrics not found")
    return metrics 