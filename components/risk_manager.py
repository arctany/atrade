import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class RiskManager:
    def __init__(self, config: Dict):
        self.config = config
        self.market_data = None
        self.positions = {}
        self.risk_metrics = {}
        self.alerts = []

    def update_data(self, market_data: pd.DataFrame):
        """更新市场数据"""
        self.market_data = market_data
        logger.info("风险管理器市场数据更新成功")

    def update_positions(self, positions: Dict):
        """更新持仓信息"""
        self.positions = positions
        logger.info("风险管理器持仓信息更新成功")

    def analyze_risk(self) -> Dict:
        """分析风险状况"""
        if self.market_data is None or self.market_data.empty:
            return {"error": "没有可用的市场数据"}

        try:
            risk_metrics = {
                "timestamp": datetime.now().isoformat(),
                "var": self._calculate_var(),
                "volatility": self._calculate_volatility(),
                "beta": self._calculate_beta(),
                "correlation": self._calculate_correlation(),
                "drawdown": self._calculate_drawdown(),
                "position_risk": self._analyze_position_risk()
            }
            self.risk_metrics = risk_metrics
            return risk_metrics
        except Exception as e:
            logger.error(f"风险分析失败: {str(e)}")
            return {"error": str(e)}

    def check_risk_limits(self) -> List[Dict]:
        """检查风险限制"""
        if not self.risk_metrics:
            return []

        alerts = []
        metrics = self.risk_metrics

        # 检查最大回撤
        if metrics["drawdown"] > self.config["max_drawdown"]:
            alerts.append({
                "type": "drawdown",
                "level": "high",
                "message": f"当前回撤 {metrics['drawdown']:.2%} 超过限制 {self.config['max_drawdown']:.2%}"
            })

        # 检查VaR
        if metrics["var"] > self.config["var_limit"]:
            alerts.append({
                "type": "var",
                "level": "high",
                "message": f"当前VaR {metrics['var']:.2%} 超过限制 {self.config['var_limit']:.2%}"
            })

        # 检查波动率
        if metrics["volatility"] > self.config["volatility_limit"]:
            alerts.append({
                "type": "volatility",
                "level": "medium",
                "message": f"当前波动率 {metrics['volatility']:.2%} 超过限制 {self.config['volatility_limit']:.2%}"
            })

        # 检查Beta
        if abs(metrics["beta"]) > self.config["beta_limit"]:
            alerts.append({
                "type": "beta",
                "level": "medium",
                "message": f"当前Beta {metrics['beta']:.2f} 超过限制 {self.config['beta_limit']:.2f}"
            })

        # 检查相关性
        if abs(metrics["correlation"]) > self.config["correlation_limit"]:
            alerts.append({
                "type": "correlation",
                "level": "medium",
                "message": f"当前相关性 {metrics['correlation']:.2f} 超过限制 {self.config['correlation_limit']:.2f}"
            })

        self.alerts = alerts
        return alerts

    def _calculate_var(self, confidence: float = 0.95) -> float:
        """计算VaR"""
        if len(self.market_data) < 2:
            return 0.0
        returns = self.market_data["close"].pct_change().dropna()
        return np.percentile(returns, (1 - confidence) * 100)

    def _calculate_volatility(self) -> float:
        """计算波动率"""
        if len(self.market_data) < 2:
            return 0.0
        returns = self.market_data["close"].pct_change().dropna()
        return returns.std() * (252 ** 0.5)  # 年化波动率

    def _calculate_beta(self) -> float:
        """计算Beta"""
        if len(self.market_data) < 2:
            return 1.0
        returns = self.market_data["close"].pct_change().dropna()
        market_returns = pd.Series(returns).rolling(window=20).mean()
        covariance = returns.cov(market_returns)
        market_variance = market_returns.var()
        return covariance / market_variance if market_variance != 0 else 1.0

    def _calculate_correlation(self) -> float:
        """计算相关性"""
        if len(self.market_data) < 2:
            return 0.0
        returns = self.market_data["close"].pct_change().dropna()
        market_returns = pd.Series(returns).rolling(window=20).mean()
        return returns.corr(market_returns)

    def _calculate_drawdown(self) -> float:
        """计算最大回撤"""
        if len(self.market_data) < 2:
            return 0.0
        cumulative = (1 + self.market_data["close"].pct_change()).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdowns = cumulative / rolling_max - 1
        return abs(drawdowns.min())

    def _analyze_position_risk(self) -> Dict:
        """分析持仓风险"""
        if not self.positions:
            return {"total_exposure": 0.0, "concentration_risk": 0.0}

        total_value = sum(pos["quantity"] * pos["current_price"] for pos in self.positions.values())
        if total_value == 0:
            return {"total_exposure": 0.0, "concentration_risk": 0.0}

        # 计算集中度风险
        position_values = [pos["quantity"] * pos["current_price"] for pos in self.positions.values()]
        concentration = max(position_values) / total_value

        return {
            "total_exposure": total_value,
            "concentration_risk": concentration
        } 