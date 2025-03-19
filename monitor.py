import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from datetime import datetime, timedelta
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from config import LOG_CONFIG, TRADING_STRATEGY, MONITORING_CONFIG, MONITOR_CONFIG
import asyncio
import aiohttp
import websockets
import json
import os
import time
from dataclasses import dataclass
from enum import Enum
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

class AlertLevel(Enum):
    """告警级别"""
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

@dataclass
class Alert:
    """告警信息"""
    level: AlertLevel
    message: str
    timestamp: datetime
    data: Optional[Dict] = None

class Monitor:
    def __init__(self, email_config: Dict = None):
        self.alerts: List[Dict] = []
        self.alert_history: List[Dict] = []
        self.email_config = email_config
        self.alert_thresholds = {
            'price_change': 0.05,  # 5%价格变动
            'volume_spike': 3.0,    # 3倍成交量
            'loss_threshold': 0.1,  # 10%亏损
            'drawdown_limit': 0.2,  # 20%回撤
            'position_limit': 0.1   # 10%持仓限制
        }

    def check_price_alerts(self, symbol: str, current_price: float, 
                         historical_data: pd.DataFrame) -> List[Dict]:
        """检查价格告警"""
        try:
            alerts = []
            
            # 计算价格变动
            price_change = (current_price - historical_data['Close'].iloc[-2]) / historical_data['Close'].iloc[-2]
            
            if abs(price_change) > self.alert_thresholds['price_change']:
                alerts.append({
                    'type': 'PRICE_ALERT',
                    'symbol': symbol,
                    'message': f"Price {price_change:.2%} change detected",
                    'severity': 'HIGH' if abs(price_change) > 0.1 else 'MEDIUM',
                    'timestamp': datetime.now()
                })
            
            return alerts
        except Exception as e:
            logger.error(f"Error checking price alerts: {str(e)}")
            return []

    def check_volume_alerts(self, symbol: str, current_volume: float,
                          historical_data: pd.DataFrame) -> List[Dict]:
        """检查成交量告警"""
        try:
            alerts = []
            
            # 计算成交量变化
            avg_volume = historical_data['Volume'].rolling(window=20).mean().iloc[-1]
            volume_ratio = current_volume / avg_volume
            
            if volume_ratio > self.alert_thresholds['volume_spike']:
                alerts.append({
                    'type': 'VOLUME_ALERT',
                    'symbol': symbol,
                    'message': f"Volume spike detected: {volume_ratio:.1f}x average",
                    'severity': 'HIGH' if volume_ratio > 5 else 'MEDIUM',
                    'timestamp': datetime.now()
                })
            
            return alerts
        except Exception as e:
            logger.error(f"Error checking volume alerts: {str(e)}")
            return []

    def check_position_alerts(self, positions: Dict) -> List[Dict]:
        """检查持仓告警"""
        try:
            alerts = []
            
            for symbol, position in positions.items():
                # 检查持仓规模
                if position['risk_metrics']['weight'] > self.alert_thresholds['position_limit']:
                    alerts.append({
                        'type': 'POSITION_ALERT',
                        'symbol': symbol,
                        'message': f"Position size exceeds limit: {position['risk_metrics']['weight']:.1%}",
                        'severity': 'HIGH',
                        'timestamp': datetime.now()
                    })
                
                # 检查未实现盈亏
                if position['unrealized_pnl'] < -self.alert_thresholds['loss_threshold'] * position['value']:
                    alerts.append({
                        'type': 'LOSS_ALERT',
                        'symbol': symbol,
                        'message': f"Large unrealized loss: {position['unrealized_pnl']:.2f}",
                        'severity': 'HIGH',
                        'timestamp': datetime.now()
                    })
            
            return alerts
        except Exception as e:
            logger.error(f"Error checking position alerts: {str(e)}")
            return []

    def check_system_alerts(self, system_status: Dict) -> List[Dict]:
        """检查系统告警"""
        try:
            alerts = []
            
            # 检查连接状态
            if not system_status.get('ibkr_connected', False):
                alerts.append({
                    'type': 'SYSTEM_ALERT',
                    'message': "IBKR connection lost",
                    'severity': 'HIGH',
                    'timestamp': datetime.now()
                })
            
            # 检查数据库连接
            if not system_status.get('database_connected', False):
                alerts.append({
                    'type': 'SYSTEM_ALERT',
                    'message': "Database connection lost",
                    'severity': 'HIGH',
                    'timestamp': datetime.now()
                })
            
            # 检查系统资源
            if system_status.get('cpu_usage', 0) > 90:
                alerts.append({
                    'type': 'SYSTEM_ALERT',
                    'message': "High CPU usage detected",
                    'severity': 'MEDIUM',
                    'timestamp': datetime.now()
                })
            
            if system_status.get('memory_usage', 0) > 90:
                alerts.append({
                    'type': 'SYSTEM_ALERT',
                    'message': "High memory usage detected",
                    'severity': 'MEDIUM',
                    'timestamp': datetime.now()
                })
            
            return alerts
        except Exception as e:
            logger.error(f"Error checking system alerts: {str(e)}")
            return []

    def process_alerts(self, alerts: List[Dict]):
        """处理告警"""
        try:
            for alert in alerts:
                # 记录告警
                self.alerts.append(alert)
                self.alert_history.append(alert)
                
                # 记录日志
                logger.warning(f"Alert: {alert['message']}")
                
                # 发送邮件通知
                if self.email_config and alert['severity'] == 'HIGH':
                    self._send_email_alert(alert)
        except Exception as e:
            logger.error(f"Error processing alerts: {str(e)}")

    def _send_email_alert(self, alert: Dict):
        """发送邮件告警"""
        try:
            if not self.email_config:
                return

            msg = MIMEMultipart()
            msg['From'] = self.email_config['sender']
            msg['To'] = self.email_config['recipient']
            msg['Subject'] = f"Trading System Alert: {alert['type']}"

            body = f"""
            Alert Type: {alert['type']}
            Severity: {alert['severity']}
            Message: {alert['message']}
            Time: {alert['timestamp']}
            """

            msg.attach(MIMEText(body, 'plain'))

            with smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port']) as server:
                server.starttls()
                server.login(self.email_config['username'], self.email_config['password'])
                server.send_message(msg)

            logger.info(f"Email alert sent for {alert['type']}")
        except Exception as e:
            logger.error(f"Error sending email alert: {str(e)}")

    def get_alert_summary(self) -> Dict:
        """获取告警摘要"""
        try:
            # 统计告警类型
            alert_types = pd.DataFrame(self.alert_history)['type'].value_counts()
            
            # 统计告警严重程度
            alert_severities = pd.DataFrame(self.alert_history)['severity'].value_counts()
            
            # 获取最近的告警
            recent_alerts = sorted(
                self.alert_history,
                key=lambda x: x['timestamp'],
                reverse=True
            )[:10]
            
            return {
                'alert_types': alert_types.to_dict(),
                'alert_severities': alert_severities.to_dict(),
                'recent_alerts': recent_alerts,
                'total_alerts': len(self.alert_history)
            }
        except Exception as e:
            logger.error(f"Error getting alert summary: {str(e)}")
            return {}

    def clear_alerts(self):
        """清除当前告警"""
        self.alerts = []

class StrategyMonitor:
    def __init__(
        self,
        strategy_name: str,
        data_source: str,
        update_interval: int = 60,
        alert_thresholds: Optional[Dict] = None
    ):
        """
        初始化策略监控器
        
        Args:
            strategy_name: 策略名称
            data_source: 数据源（例如：交易所API、数据库等）
            update_interval: 更新间隔（秒）
            alert_thresholds: 告警阈值
        """
        self.strategy_name = strategy_name
        self.data_source = data_source
        self.update_interval = update_interval
        self.alert_thresholds = alert_thresholds or MONITOR_CONFIG['alert_thresholds']
        self.monitor_config = MONITOR_CONFIG
        
        # 初始化状态
        self.is_running = False
        self.last_update = None
        self.alerts = []
        self.performance_metrics = {}
        self.position_data = {}
        
        # 创建输出目录
        self.output_dir = os.path.join('monitor_output', strategy_name)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 设置日志
        self._setup_logging()
    
    def _setup_logging(self):
        """设置日志"""
        log_file = os.path.join(self.output_dir, f'monitor_{datetime.now().strftime("%Y%m%d")}.log')
        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(handler)
    
    async def start_monitoring(self):
        """启动监控"""
        try:
            self.is_running = True
            logger.info(f"Starting monitoring for strategy: {self.strategy_name}")
            
            while self.is_running:
                try:
                    # 获取最新数据
                    await self._fetch_latest_data()
                    
                    # 更新性能指标
                    await self._update_performance_metrics()
                    
                    # 检查告警条件
                    await self._check_alerts()
                    
                    # 生成报告
                    await self._generate_report()
                    
                    # 等待下一次更新
                    await asyncio.sleep(self.update_interval)
                    
                except Exception as e:
                    logger.error(f"Error in monitoring loop: {str(e)}")
                    await self._handle_error(e)
            
        except Exception as e:
            logger.error(f"Error starting monitoring: {str(e)}")
            raise
    
    async def stop_monitoring(self):
        """停止监控"""
        self.is_running = False
        logger.info(f"Stopping monitoring for strategy: {self.strategy_name}")
    
    async def _fetch_latest_data(self):
        """获取最新数据"""
        try:
            # 根据数据源类型获取数据
            if self.data_source.startswith('http'):
                async with aiohttp.ClientSession() as session:
                    async with session.get(self.data_source) as response:
                        data = await response.json()
            elif self.data_source.startswith('ws'):
                async with websockets.connect(self.data_source) as websocket:
                    data = await websocket.recv()
                    data = json.loads(data)
            else:
                # 从文件或数据库读取数据
                data = pd.read_csv(self.data_source)
            
            # 更新数据
            self.last_update = datetime.now()
            self._process_data(data)
            
        except Exception as e:
            logger.error(f"Error fetching data: {str(e)}")
            raise
    
    def _process_data(self, data: Union[Dict, pd.DataFrame]):
        """处理数据"""
        try:
            if isinstance(data, pd.DataFrame):
                # 处理DataFrame数据
                self.position_data = data.to_dict('records')
            else:
                # 处理字典数据
                self.position_data = data
            
            logger.info("Data processed successfully")
            
        except Exception as e:
            logger.error(f"Error processing data: {str(e)}")
            raise
    
    async def _update_performance_metrics(self):
        """更新性能指标"""
        try:
            if not self.position_data:
                return
            
            # 计算性能指标
            returns = self._calculate_returns()
            self.performance_metrics.update({
                'total_return': self._calculate_total_return(returns),
                'sharpe_ratio': self._calculate_sharpe_ratio(returns),
                'max_drawdown': self._calculate_max_drawdown(returns),
                'win_rate': self._calculate_win_rate(returns),
                'profit_factor': self._calculate_profit_factor(returns)
            })
            
            logger.info("Performance metrics updated")
            
        except Exception as e:
            logger.error(f"Error updating performance metrics: {str(e)}")
            raise
    
    def _calculate_returns(self) -> np.ndarray:
        """计算收益率"""
        try:
            if isinstance(self.position_data, pd.DataFrame):
                returns = self.position_data['close'].pct_change().dropna().values
            else:
                prices = [pos['close'] for pos in self.position_data]
                returns = np.diff(prices) / prices[:-1]
            
            return returns
            
        except Exception as e:
            logger.error(f"Error calculating returns: {str(e)}")
            raise
    
    def _calculate_total_return(self, returns: np.ndarray) -> float:
        """计算总收益率"""
        return (1 + returns).prod() - 1
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
        """计算夏普比率"""
        if len(returns) < 2:
            return 0.0
        return np.mean(returns) / np.std(returns) * np.sqrt(252)
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """计算最大回撤"""
        cum_returns = (1 + returns).cumprod()
        rolling_max = np.maximum.accumulate(cum_returns)
        drawdowns = (cum_returns - rolling_max) / rolling_max
        return abs(drawdowns.min())
    
    def _calculate_win_rate(self, returns: np.ndarray) -> float:
        """计算胜率"""
        return np.mean(returns > 0)
    
    def _calculate_profit_factor(self, returns: np.ndarray) -> float:
        """计算盈亏比"""
        positive_returns = returns[returns > 0].sum()
        negative_returns = abs(returns[returns < 0].sum())
        return positive_returns / negative_returns if negative_returns != 0 else float('inf')
    
    async def _check_alerts(self):
        """检查告警条件"""
        try:
            # 检查性能指标告警
            for metric, threshold in self.alert_thresholds.items():
                if metric in self.performance_metrics:
                    value = self.performance_metrics[metric]
                    if value < threshold['min']:
                        await self._create_alert(
                            AlertLevel.WARNING,
                            f"{metric} below threshold: {value:.2f} < {threshold['min']:.2f}"
                        )
                    elif value > threshold['max']:
                        await self._create_alert(
                            AlertLevel.WARNING,
                            f"{metric} above threshold: {value:.2f} > {threshold['max']:.2f}"
                        )
            
            # 检查数据更新告警
            if self.last_update:
                time_since_update = (datetime.now() - self.last_update).total_seconds()
                if time_since_update > self.monitor_config['data_update_timeout']:
                    await self._create_alert(
                        AlertLevel.ERROR,
                        f"Data not updated for {time_since_update:.0f} seconds"
                    )
            
        except Exception as e:
            logger.error(f"Error checking alerts: {str(e)}")
            raise
    
    async def _create_alert(self, level: AlertLevel, message: str, data: Optional[Dict] = None):
        """创建告警"""
        try:
            alert = Alert(
                level=level,
                message=message,
                timestamp=datetime.now(),
                data=data
            )
            
            self.alerts.append(alert)
            
            # 记录告警
            logger.warning(f"Alert: {level.value} - {message}")
            
            # 发送告警通知
            await self._send_alert_notification(alert)
            
        except Exception as e:
            logger.error(f"Error creating alert: {str(e)}")
            raise
    
    async def _send_alert_notification(self, alert: Alert):
        """发送告警通知"""
        try:
            # 根据配置发送通知
            if self.monitor_config['notifications']['email']:
                await self._send_email_notification(alert)
            
            if self.monitor_config['notifications']['webhook']:
                await self._send_webhook_notification(alert)
            
            if self.monitor_config['notifications']['telegram']:
                await self._send_telegram_notification(alert)
            
        except Exception as e:
            logger.error(f"Error sending alert notification: {str(e)}")
            raise
    
    async def _send_email_notification(self, alert: Alert):
        """发送邮件通知"""
        # TODO: 实现邮件通知
        pass
    
    async def _send_webhook_notification(self, alert: Alert):
        """发送Webhook通知"""
        try:
            webhook_url = self.monitor_config['notifications']['webhook_url']
            payload = {
                'strategy': self.strategy_name,
                'level': alert.level.value,
                'message': alert.message,
                'timestamp': alert.timestamp.isoformat(),
                'data': alert.data
            }
            
            async with aiohttp.ClientSession() as session:
                await session.post(webhook_url, json=payload)
            
        except Exception as e:
            logger.error(f"Error sending webhook notification: {str(e)}")
            raise
    
    async def _send_telegram_notification(self, alert: Alert):
        """发送Telegram通知"""
        try:
            bot_token = self.monitor_config['notifications']['telegram_bot_token']
            chat_id = self.monitor_config['notifications']['telegram_chat_id']
            
            message = (
                f"🚨 *{alert.level.value} Alert*\n"
                f"Strategy: {self.strategy_name}\n"
                f"Message: {alert.message}\n"
                f"Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
            )
            
            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            payload = {
                'chat_id': chat_id,
                'text': message,
                'parse_mode': 'Markdown'
            }
            
            async with aiohttp.ClientSession() as session:
                await session.post(url, json=payload)
            
        except Exception as e:
            logger.error(f"Error sending telegram notification: {str(e)}")
            raise
    
    async def _handle_error(self, error: Exception):
        """处理错误"""
        try:
            # 创建错误告警
            await self._create_alert(
                AlertLevel.ERROR,
                f"Error in monitoring: {str(error)}"
            )
            
            # 等待一段时间后继续
            await asyncio.sleep(self.monitor_config['error_retry_interval'])
            
        except Exception as e:
            logger.error(f"Error handling error: {str(e)}")
            raise
    
    async def _generate_report(self):
        """生成报告"""
        try:
            # 生成HTML报告
            html_content = self._generate_html_report()
            
            # 保存报告
            report_path = os.path.join(
                self.output_dir,
                f'monitor_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html'
            )
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"Report generated: {report_path}")
            
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            raise
    
    def _generate_html_report(self) -> str:
        """生成HTML报告"""
        try:
            html_content = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Strategy Monitor Report</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    .section { margin-bottom: 30px; }
                    table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
                    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                    th { background-color: #f2f2f2; }
                    .chart { margin-bottom: 30px; }
                    .alert { padding: 10px; margin-bottom: 10px; border-radius: 4px; }
                    .alert-info { background-color: #e3f2fd; }
                    .alert-warning { background-color: #fff3e0; }
                    .alert-error { background-color: #ffebee; }
                    .alert-critical { background-color: #f44336; color: white; }
                </style>
            </head>
            <body>
                <h1>Strategy Monitor Report</h1>
                <p>Generated at: {timestamp}</p>
            """.format(timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            
            # 添加性能指标
            html_content += self._generate_performance_section()
            
            # 添加图表
            html_content += self._generate_charts()
            
            # 添加告警
            html_content += self._generate_alerts_section()
            
            html_content += """
            </body>
            </html>
            """
            
            return html_content
            
        except Exception as e:
            logger.error(f"Error generating HTML report: {str(e)}")
            raise
    
    def _generate_performance_section(self) -> str:
        """生成性能指标部分"""
        try:
            metrics_table = pd.DataFrame({
                'Metric': list(self.performance_metrics.keys()),
                'Value': list(self.performance_metrics.values())
            }).to_html(index=False)
            
            return f"""
                <div class="section">
                    <h2>Performance Metrics</h2>
                    {metrics_table}
                </div>
            """
            
        except Exception as e:
            logger.error(f"Error generating performance section: {str(e)}")
            raise
    
    def _generate_charts(self) -> str:
        """生成图表"""
        try:
            # 创建收益率图表
            returns_fig = go.Figure()
            returns_fig.add_trace(go.Scatter(
                y=self._calculate_returns(),
                mode='lines',
                name='Returns'
            ))
            returns_fig.update_layout(
                title='Strategy Returns',
                xaxis_title='Time',
                yaxis_title='Return'
            )
            returns_html = returns_fig.to_html(full_html=False)
            
            # 创建回撤图表
            drawdown_fig = go.Figure()
            cum_returns = (1 + self._calculate_returns()).cumprod()
            rolling_max = np.maximum.accumulate(cum_returns)
            drawdowns = (cum_returns - rolling_max) / rolling_max
            drawdown_fig.add_trace(go.Scatter(
                y=drawdowns,
                mode='lines',
                name='Drawdown'
            ))
            drawdown_fig.update_layout(
                title='Strategy Drawdown',
                xaxis_title='Time',
                yaxis_title='Drawdown'
            )
            drawdown_html = drawdown_fig.to_html(full_html=False)
            
            return f"""
                <div class="section">
                    <h2>Charts</h2>
                    <div class="chart">
                        {returns_html}
                    </div>
                    <div class="chart">
                        {drawdown_html}
                    </div>
                </div>
            """
            
        except Exception as e:
            logger.error(f"Error generating charts: {str(e)}")
            raise
    
    def _generate_alerts_section(self) -> str:
        """生成告警部分"""
        try:
            alerts_html = ""
            for alert in self.alerts[-10:]:  # 只显示最近10条告警
                alerts_html += f"""
                    <div class="alert alert-{alert.level.value.lower()}">
                        <strong>{alert.level.value}</strong> - {alert.message}<br>
                        Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
                    </div>
                """
            
            return f"""
                <div class="section">
                    <h2>Recent Alerts</h2>
                    {alerts_html}
                </div>
            """
            
        except Exception as e:
            logger.error(f"Error generating alerts section: {str(e)}")
            raise 