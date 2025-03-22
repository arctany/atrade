from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from datetime import datetime

from atrade.api.schemas.base import RiskMetrics, RiskMetricsCreate
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

@router.post("/", response_model=RiskMetrics)
def create_risk_metrics(
    metrics: RiskMetricsCreate,
    db: Session = Depends(get_db)
):
    """创建风险指标"""
    db_metrics = database.RiskMetrics(
        timestamp=metrics.timestamp,
        var=metrics.var,
        volatility=metrics.volatility,
        beta=metrics.beta,
        correlation=metrics.correlation,
        risk_metadata=metrics.risk_metadata
    )
    db.add(db_metrics)
    db.commit()
    db.refresh(db_metrics)
    return db_metrics

@router.get("/", response_model=List[RiskMetrics])
def get_risk_metrics(
    start_time: datetime,
    end_time: datetime,
    db: Session = Depends(get_db)
):
    """获取风险指标"""
    return db.query(database.RiskMetrics).filter(
        database.RiskMetrics.timestamp >= start_time,
        database.RiskMetrics.timestamp <= end_time
    ).all()

@router.get("/latest", response_model=RiskMetrics)
def get_latest_risk_metrics(db: Session = Depends(get_db)):
    """获取最新风险指标"""
    metrics = db.query(database.RiskMetrics).order_by(
        database.RiskMetrics.timestamp.desc()
    ).first()
    
    if metrics is None:
        raise HTTPException(status_code=404, detail="Risk metrics not found")
    return metrics 