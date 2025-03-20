import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class Backtest:
    def __init__(self, config: Dict):
        self.config = config
        self.market_data = None
        self.positions = {}
        self.trades = []
        self.performance_metrics = {}
        self.initial_capital = config.get("initial_capital", 100000.0)
        self.commission_rate = config.get("commission_rate", 0.0003)
        self.slippage = config.get("slippage", 0.0001)

    def update_data(self, market_data: pd.DataFrame):
        """更新市场数据"""
        self.market_data = market_data
        logger.info("回测系统市场数据更新成功")

    def update_positions(self, positions: Dict):
        """更新持仓信息"""
        self.positions = positions
        logger.info("回测系统持仓信息更新成功")

    def run_backtest(self, strategy: str = "default") -> Dict:
        """运行回测"""
        if self.market_data is None or self.market_data.empty:
            return {"error": "没有可用的市场数据"}

        try:
            # 初始化回测参数
            capital = self.initial_capital
            position = 0
            trades = []
            equity_curve = [capital]

            # 遍历市场数据
            for i in range(1, len(self.market_data)):
                current_price = self.market_data.iloc[i]["close"]
                prev_price = self.market_data.iloc[i-1]["close"]
                
                # 根据策略生成信号
                signal = self._generate_signal(strategy, i)
                
                # 执行交易
                if signal == "buy" and position <= 0:
                    # 买入
                    shares = capital / current_price
                    cost = shares * current_price * (1 + self.commission_rate + self.slippage)
                    if cost <= capital:
                        position = shares
                        capital -= cost
                        trades.append({
                            "timestamp": self.market_data.iloc[i]["date"],
                            "type": "buy",
                            "price": current_price,
                            "shares": shares,
                            "cost": cost
                        })
                elif signal == "sell" and position >= 0:
                    # 卖出
                    if position > 0:
                        revenue = position * current_price * (1 - self.commission_rate - self.slippage)
                        capital += revenue
                        trades.append({
                            "timestamp": self.market_data.iloc[i]["date"],
                            "type": "sell",
                            "price": current_price,
                            "shares": position,
                            "revenue": revenue
                        })
                        position = 0

                # 更新权益曲线
                current_value = capital + position * current_price
                equity_curve.append(current_value)

            # 计算性能指标
            self.trades = trades
            self.performance_metrics = self._calculate_performance_metrics(equity_curve)
            return self.performance_metrics

        except Exception as e:
            logger.error(f"回测执行失败: {str(e)}")
            return {"error": str(e)}

    def _generate_signal(self, strategy: str, index: int) -> str:
        """生成交易信号"""
        if strategy == "default":
            # 简单的均线策略
            if index < 20:
                return "hold"
            
            ma5 = self.market_data["close"].rolling(window=5).mean()
            ma20 = self.market_data["close"].rolling(window=20).mean()
            
            if ma5.iloc[index] > ma20.iloc[index] and ma5.iloc[index-1] <= ma20.iloc[index-1]:
                return "buy"
            elif ma5.iloc[index] < ma20.iloc[index] and ma5.iloc[index-1] >= ma20.iloc[index-1]:
                return "sell"
            return "hold"
        
        return "hold"  # 默认持有

    def _calculate_performance_metrics(self, equity_curve: List[float]) -> Dict:
        """计算性能指标"""
        if not equity_curve:
            return {}

        equity_series = pd.Series(equity_curve)
        returns = equity_series.pct_change().dropna()

        # 计算年化收益率
        total_return = (equity_curve[-1] / self.initial_capital - 1)
        annual_return = (1 + total_return) ** (252 / len(equity_curve)) - 1

        # 计算夏普比率
        risk_free_rate = 0.02  # 假设无风险利率为2%
        excess_returns = returns - risk_free_rate/252
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std()

        # 计算最大回撤
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdowns = cumulative / rolling_max - 1
        max_drawdown = abs(drawdowns.min())

        # 计算胜率
        winning_trades = len([t for t in self.trades if t["type"] == "sell" and t["revenue"] > t["cost"]])
        total_trades = len([t for t in self.trades if t["type"] == "sell"])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        return {
            "timestamp": datetime.now().isoformat(),
            "total_return": total_return,
            "annual_return": annual_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "total_trades": total_trades,
            "equity_curve": equity_curve
        }

    def get_trade_history(self) -> List[Dict]:
        """获取交易历史"""
        return self.trades

    def get_performance_metrics(self) -> Dict:
        """获取性能指标"""
        return self.performance_metrics 