import os
import logging
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class TradingAgentLLM:
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key
        self.model = model
        self.market_data = None
        self.positions = {}
        self.capital = 100000.0  # 初始资金
        self.commission_rate = 0.0003  # 手续费率
        self.slippage = 0.0001  # 滑点

    def update_data(self, market_data: pd.DataFrame):
        """更新市场数据"""
        self.market_data = market_data
        logger.info("交易助手市场数据更新成功")

    def update_positions(self, positions: Dict):
        """更新持仓信息"""
        self.positions = positions
        logger.info("交易助手持仓信息更新成功")

    def analyze_market(self) -> Dict:
        """分析市场状况"""
        if self.market_data is None or self.market_data.empty:
            return {"error": "没有可用的市场数据"}

        try:
            latest_data = self.market_data.iloc[-1]
            analysis = {
                "timestamp": datetime.now().isoformat(),
                "market_status": "正常",
                "price": latest_data["close"],
                "volume": latest_data["volume"],
                "trend": "上升" if latest_data["close"] > latest_data["open"] else "下降",
                "volatility": self._calculate_volatility(),
                "support_levels": self._find_support_levels(),
                "resistance_levels": self._find_resistance_levels()
            }
            return analysis
        except Exception as e:
            logger.error(f"市场分析失败: {str(e)}")
            return {"error": str(e)}

    def generate_trading_signal(self) -> Dict:
        """生成交易信号"""
        if self.market_data is None or self.market_data.empty:
            return {"error": "没有可用的市场数据"}

        try:
            analysis = self.analyze_market()
            signal = {
                "timestamp": datetime.now().isoformat(),
                "action": "持有",  # 默认持有
                "confidence": 0.5,
                "price_target": None,
                "stop_loss": None,
                "analysis": analysis
            }

            # 基于技术分析生成信号
            if self._is_oversold():
                signal["action"] = "买入"
                signal["confidence"] = 0.7
            elif self._is_overbought():
                signal["action"] = "卖出"
                signal["confidence"] = 0.7

            return signal
        except Exception as e:
            logger.error(f"生成交易信号失败: {str(e)}")
            return {"error": str(e)}

    def _calculate_volatility(self) -> float:
        """计算波动率"""
        if len(self.market_data) < 2:
            return 0.0
        returns = self.market_data["close"].pct_change().dropna()
        return returns.std() * (252 ** 0.5)  # 年化波动率

    def _find_support_levels(self) -> List[float]:
        """寻找支撑位"""
        if len(self.market_data) < 20:
            return []
        # 使用最近20个交易日的最低点作为支撑位
        return sorted(self.market_data["low"].tail(20).unique())[:3]

    def _find_resistance_levels(self) -> List[float]:
        """寻找阻力位"""
        if len(self.market_data) < 20:
            return []
        # 使用最近20个交易日的最高点作为阻力位
        return sorted(self.market_data["high"].tail(20).unique(), reverse=True)[:3]

    def _is_oversold(self) -> bool:
        """判断是否超卖"""
        if len(self.market_data) < 14:
            return False
        # 使用RSI指标判断超卖
        delta = self.market_data["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] < 30

    def _is_overbought(self) -> bool:
        """判断是否超买"""
        if len(self.market_data) < 14:
            return False
        # 使用RSI指标判断超买
        delta = self.market_data["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] > 70 