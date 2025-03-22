from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List

from atrade.config.settings import load_config
from atrade.database import SessionLocal, engine
import atrade.database as database

# 加载配置
config = load_config()

# 创建 FastAPI 应用
app = FastAPI(
    title="ATrade API",
    description="ATrade 交易系统 API",
    version="1.0.0"
)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该设置具体的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 依赖项
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# 导入路由
from atrade.api.routers import market_data, trades, positions, performance, risk

# 注册路由
app.include_router(market_data.router, prefix="/api/v1/market-data", tags=["market-data"])
app.include_router(trades.router, prefix="/api/v1/trades", tags=["trades"])
app.include_router(positions.router, prefix="/api/v1/positions", tags=["positions"])
app.include_router(performance.router, prefix="/api/v1/performance", tags=["performance"])
app.include_router(risk.router, prefix="/api/v1/risk", tags=["risk"])

@app.get("/")
async def root():
    return {"message": "Welcome to ATrade API"} 