import os
import logging
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional

from config import (
    DATABASE_CONFIG, LOG_CONFIG, API_CONFIG,
    TRADING_CONFIG, RISK_CONFIG, STRATEGY_CONFIG,
    MONITOR_CONFIG, REPORT_CONFIG
)
from database import Database
from trading_agent_llm import TradingAgentLLM
from risk_manager import RiskManager
from backtest import Backtest
from monitor import Monitor

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(
    level=getattr(logging, LOG_CONFIG['level']),
    format=LOG_CONFIG['format'],
    filename=LOG_CONFIG['file']
)
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title="智能交易助手系统",
    description="基于大模型的智能交易助手系统",
    version="1.0.0"
)

# 全局变量
market_data = None
positions = {}
capital = TRADING_CONFIG['initial_capital']
db = None
trading_agent = None
risk_manager = None
backtest = None
monitor = None

class MarketData(BaseModel):
    """市场数据模型"""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

class Position(BaseModel):
    """持仓数据模型"""
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    profit_loss: float
    profit_loss_pct: float

class Message(BaseModel):
    """消息模型"""
    message: str

def initialize_components():
    """初始化系统组件"""
    global market_data, db, trading_agent, risk_manager, backtest, monitor
    
    try:
        # 加载市场数据
        market_data = load_market_data()
        
        # 初始化数据库
        db = Database()
        
        # 初始化交易助手
        trading_agent = TradingAgentLLM(
            data=market_data,
            positions=positions,
            capital=capital
        )
        
        # 初始化风险管理器
        risk_manager = RiskManager(
            data=market_data,
            positions=positions,
            capital=capital,
            config=RISK_CONFIG
        )
        
        # 初始化回测系统
        backtest = Backtest(
            data=market_data,
            config=STRATEGY_CONFIG
        )
        
        # 初始化监控系统
        monitor = Monitor(
            data=market_data,
            positions=positions,
            capital=capital,
            config=MONITOR_CONFIG
        )
        
        logger.info("系统组件初始化成功")
    except Exception as e:
        logger.error(f"系统组件初始化失败: {str(e)}")
        raise

def load_market_data() -> pd.DataFrame:
    """加载市场数据"""
    try:
        data = pd.read_csv('data/market_data.csv')
        data['date'] = pd.to_datetime(data['date'])
        return data
    except Exception as e:
        logger.error(f"加载市场数据失败: {str(e)}")
        raise

@app.on_event("startup")
async def startup_event():
    """系统启动事件"""
    initialize_components()

@app.post("/api/update_market_data")
async def update_market_data(data: MarketData):
    """更新市场数据"""
    try:
        # 保存到数据库
        db.save_market_data(data.dict())
        
        # 更新全局市场数据
        global market_data
        new_data = pd.DataFrame([data.dict()])
        market_data = pd.concat([market_data, new_data], ignore_index=True)
        
        # 更新各个组件
        trading_agent.update_data(market_data)
        risk_manager.update_data(market_data)
        backtest.update_data(market_data)
        monitor.update_data(market_data)
        
        return {"status": "success", "message": "市场数据更新成功"}
    except Exception as e:
        logger.error(f"更新市场数据失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/update_position")
async def update_position(position: Position):
    """更新持仓信息"""
    try:
        # 保存到数据库
        db.save_position(position.dict())
        
        # 更新全局持仓
        positions[position.symbol] = position.dict()
        
        # 更新各个组件
        trading_agent.update_positions(positions)
        risk_manager.update_positions(positions)
        monitor.update_positions(positions)
        
        return {"status": "success", "message": "持仓信息更新成功"}
    except Exception as e:
        logger.error(f"更新持仓信息失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
async def chat(message: Message):
    """处理用户消息"""
    try:
        response = trading_agent.process_message(message.message)
        return {"response": response}
    except Exception as e:
        logger.error(f"处理用户消息失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/risk_analysis")
async def get_risk_analysis():
    """获取风险分析"""
    try:
        analysis = risk_manager.analyze_risk()
        return analysis
    except Exception as e:
        logger.error(f"获取风险分析失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/performance_metrics")
async def get_performance_metrics():
    """获取性能指标"""
    try:
        metrics = monitor.get_performance_metrics()
        return metrics
    except Exception as e:
        logger.error(f"获取性能指标失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/backtest_results")
async def get_backtest_results():
    """获取回测结果"""
    try:
        results = backtest.run()
        return results
    except Exception as e:
        logger.error(f"获取回测结果失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "database": "connected" if db else "disconnected",
            "trading_agent": "initialized" if trading_agent else "not_initialized",
            "risk_manager": "initialized" if risk_manager else "not_initialized",
            "backtest": "initialized" if backtest else "not_initialized",
            "monitor": "initialized" if monitor else "not_initialized"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=API_CONFIG['host'],
        port=API_CONFIG['port'],
        reload=API_CONFIG['debug']
    ) 