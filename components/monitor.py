import logging
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class Monitor:
    def __init__(self, config: Dict):
        self.config = config
        self.market_data = None
        self.positions = {}
        self.alerts = []
        self.check_interval = config.get("check_interval", 300)  # 默认5分钟
        self.alert_threshold = config.get("alert_threshold", 0.02)  # 默认2%
        self.max_alerts = config.get("max_alerts", 100)
        self.alert_retention_days = config.get("alert_retention_days", 7)

    def update_data(self, market_data: pd.DataFrame):
        """更新市场数据"""
        self.market_data = market_data
        logger.info("监控系统市场数据更新成功")

    def update_positions(self, positions: Dict):
        """更新持仓信息"""
        self.positions = positions
        logger.info("监控系统持仓信息更新成功")

    def check_market_conditions(self) -> List[Dict]:
        """检查市场状况"""
        if self.market_data is None or self.market_data.empty:
            return []

        try:
            alerts = []
            latest_data = self.market_data.iloc[-1]
            prev_data = self.market_data.iloc[-2]

            # 检查价格变化
            price_change = (latest_data["close"] - prev_data["close"]) / prev_data["close"]
            if abs(price_change) > self.alert_threshold:
                alerts.append({
                    "timestamp": datetime.now().isoformat(),
                    "type": "price_change",
                    "level": "high" if abs(price_change) > self.alert_threshold * 2 else "medium",
                    "message": f"价格变化 {price_change:.2%} 超过阈值 {self.alert_threshold:.2%}"
                })

            # 检查成交量变化
            volume_change = (latest_data["volume"] - prev_data["volume"]) / prev_data["volume"]
            if abs(volume_change) > self.alert_threshold * 2:
                alerts.append({
                    "timestamp": datetime.now().isoformat(),
                    "type": "volume_change",
                    "level": "high" if abs(volume_change) > self.alert_threshold * 4 else "medium",
                    "message": f"成交量变化 {volume_change:.2%} 超过阈值 {self.alert_threshold * 2:.2%}"
                })

            # 检查持仓变化
            if self.positions:
                for symbol, position in self.positions.items():
                    if position["quantity"] > 0:
                        # 检查持仓盈亏
                        profit_loss_pct = position["profit_loss_pct"]
                        if abs(profit_loss_pct) > self.alert_threshold:
                            alerts.append({
                                "timestamp": datetime.now().isoformat(),
                                "type": "position_pnl",
                                "level": "high" if abs(profit_loss_pct) > self.alert_threshold * 2 else "medium",
                                "message": f"持仓 {symbol} 盈亏 {profit_loss_pct:.2%} 超过阈值 {self.alert_threshold:.2%}"
                            })

            # 更新告警列表
            self._update_alerts(alerts)
            return alerts

        except Exception as e:
            logger.error(f"市场状况检查失败: {str(e)}")
            return []

    def check_system_health(self) -> Dict:
        """检查系统健康状态"""
        try:
            health_status = {
                "timestamp": datetime.now().isoformat(),
                "status": "healthy",
                "components": {
                    "market_data": "ok" if self.market_data is not None and not self.market_data.empty else "error",
                    "positions": "ok" if self.positions is not None else "error",
                    "alerts": "ok" if len(self.alerts) < self.max_alerts else "warning"
                },
                "metrics": {
                    "data_points": len(self.market_data) if self.market_data is not None else 0,
                    "active_positions": len(self.positions),
                    "active_alerts": len(self.alerts)
                }
            }

            # 检查数据时效性
            if self.market_data is not None and not self.market_data.empty:
                latest_time = pd.to_datetime(self.market_data.iloc[-1]["date"])
                time_diff = (datetime.now() - latest_time).total_seconds()
                if time_diff > self.check_interval * 2:
                    health_status["status"] = "warning"
                    health_status["components"]["market_data"] = "stale"

            return health_status

        except Exception as e:
            logger.error(f"系统健康检查失败: {str(e)}")
            return {
                "timestamp": datetime.now().isoformat(),
                "status": "error",
                "error": str(e)
            }

    def _update_alerts(self, new_alerts: List[Dict]):
        """更新告警列表"""
        # 添加新告警
        self.alerts.extend(new_alerts)

        # 限制告警数量
        if len(self.alerts) > self.max_alerts:
            self.alerts = self.alerts[-self.max_alerts:]

        # 清理过期告警
        current_time = datetime.now()
        self.alerts = [
            alert for alert in self.alerts
            if (current_time - datetime.fromisoformat(alert["timestamp"])).days <= self.alert_retention_days
        ]

    def get_active_alerts(self) -> List[Dict]:
        """获取活动告警"""
        return self.alerts

    def clear_alerts(self):
        """清除所有告警"""
        self.alerts = []
        logger.info("所有告警已清除") 