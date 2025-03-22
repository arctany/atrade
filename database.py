import os
import logging
from typing import Dict, List, Optional
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from atrade.config.settings import load_config

# 设置日志
logger = logging.getLogger(__name__)

# 加载配置
config = load_config()

# 创建数据库引擎
DATABASE_URL = f"postgresql://{config.database.user}:{config.database.password}@{config.database.host}:{config.database.port}/{config.database.name}"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class MarketData(Base):
    """市场数据模型"""
    __tablename__ = "market_data"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True)
    timestamp = Column(DateTime, index=True)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

class Position(Base):
    """持仓数据模型"""
    __tablename__ = "positions"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True)
    quantity = Column(Float)
    entry_price = Column(Float)
    current_price = Column(Float)
    profit_loss = Column(Float)
    profit_loss_pct = Column(Float)
    updated_at = Column(DateTime, default=datetime.utcnow)

class Trade(Base):
    """交易记录模型"""
    __tablename__ = "trades"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True)
    type = Column(String)  # buy/sell
    quantity = Column(Float)
    price = Column(Float)
    timestamp = Column(DateTime, index=True)
    strategy = Column(String)
    trade_metadata = Column(JSON)

class PerformanceMetrics(Base):
    """性能指标模型"""
    __tablename__ = "performance_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, index=True)
    total_return = Column(Float)
    annual_return = Column(Float)
    sharpe_ratio = Column(Float)
    max_drawdown = Column(Float)
    win_rate = Column(Float)
    profit_factor = Column(Float)
    performance_metadata = Column(JSON)

class RiskMetrics(Base):
    """风险指标模型"""
    __tablename__ = "risk_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, index=True)
    var = Column(Float)
    volatility = Column(Float)
    beta = Column(Float)
    correlation = Column(Float)
    risk_metadata = Column(JSON)

class Database:
    """数据库管理类"""
    
    def __init__(self):
        """初始化数据库连接"""
        try:
            self.engine = engine
            self.SessionLocal = SessionLocal
            self.Base = Base
            self.create_tables()
            logger.info("数据库连接初始化成功")
        except Exception as e:
            logger.error(f"数据库连接初始化失败: {str(e)}")
            raise
    
    def create_tables(self):
        """创建数据库表"""
        try:
            self.Base.metadata.create_all(bind=self.engine)
            logger.info("数据库表创建成功")
        except Exception as e:
            logger.error(f"数据库表创建失败: {str(e)}")
            raise
    
    def get_session(self):
        """获取数据库会话"""
        return self.SessionLocal()
    
    def save_market_data(self, data: Dict):
        """保存市场数据"""
        try:
            session = self.get_session()
            market_data = MarketData(**data)
            session.add(market_data)
            session.commit()
            session.close()
            logger.info(f"市场数据保存成功: {data['symbol']}")
        except Exception as e:
            logger.error(f"市场数据保存失败: {str(e)}")
            raise
    
    def save_position(self, data: Dict):
        """保存持仓数据"""
        try:
            session = self.get_session()
            position = Position(**data)
            session.add(position)
            session.commit()
            session.close()
            logger.info(f"持仓数据保存成功: {data['symbol']}")
        except Exception as e:
            logger.error(f"持仓数据保存失败: {str(e)}")
            raise
    
    def save_trade(self, data: Dict):
        """保存交易记录"""
        try:
            session = self.get_session()
            trade = Trade(**data)
            session.add(trade)
            session.commit()
            session.close()
            logger.info(f"交易记录保存成功: {data['symbol']}")
        except Exception as e:
            logger.error(f"交易记录保存失败: {str(e)}")
            raise
    
    def save_performance_metrics(self, data: Dict):
        """保存性能指标"""
        try:
            session = self.get_session()
            metrics = PerformanceMetrics(**data)
            session.add(metrics)
            session.commit()
            session.close()
            logger.info("性能指标保存成功")
        except Exception as e:
            logger.error(f"性能指标保存失败: {str(e)}")
            raise
    
    def save_risk_metrics(self, data: Dict):
        """保存风险指标"""
        try:
            session = self.get_session()
            metrics = RiskMetrics(**data)
            session.add(metrics)
            session.commit()
            session.close()
            logger.info("风险指标保存成功")
        except Exception as e:
            logger.error(f"风险指标保存失败: {str(e)}")
            raise
    
    def get_market_data(self, symbol: str, start_time: datetime, end_time: datetime) -> List[Dict]:
        """获取市场数据"""
        try:
            session = self.get_session()
            data = session.query(MarketData).filter(
                MarketData.symbol == symbol,
                MarketData.timestamp >= start_time,
                MarketData.timestamp <= end_time
            ).all()
            session.close()
            return [item.__dict__ for item in data]
        except Exception as e:
            logger.error(f"获取市场数据失败: {str(e)}")
            raise
    
    def get_positions(self) -> List[Dict]:
        """获取持仓数据"""
        try:
            session = self.get_session()
            positions = session.query(Position).all()
            session.close()
            return [item.__dict__ for item in positions]
        except Exception as e:
            logger.error(f"获取持仓数据失败: {str(e)}")
            raise
    
    def get_trades(self, start_time: datetime, end_time: datetime) -> List[Dict]:
        """获取交易记录"""
        try:
            session = self.get_session()
            trades = session.query(Trade).filter(
                Trade.timestamp >= start_time,
                Trade.timestamp <= end_time
            ).all()
            session.close()
            return [item.__dict__ for item in trades]
        except Exception as e:
            logger.error(f"获取交易记录失败: {str(e)}")
            raise
    
    def get_performance_metrics(self, start_time: datetime, end_time: datetime) -> List[Dict]:
        """获取性能指标"""
        try:
            session = self.get_session()
            metrics = session.query(PerformanceMetrics).filter(
                PerformanceMetrics.timestamp >= start_time,
                PerformanceMetrics.timestamp <= end_time
            ).all()
            session.close()
            return [item.__dict__ for item in metrics]
        except Exception as e:
            logger.error(f"获取性能指标失败: {str(e)}")
            raise
    
    def get_risk_metrics(self, start_time: datetime, end_time: datetime) -> List[Dict]:
        """获取风险指标"""
        try:
            session = self.get_session()
            metrics = session.query(RiskMetrics).filter(
                RiskMetrics.timestamp >= start_time,
                RiskMetrics.timestamp <= end_time
            ).all()
            session.close()
            return [item.__dict__ for item in metrics]
        except Exception as e:
            logger.error(f"获取风险指标失败: {str(e)}")
            raise 