import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.figure_factory as ff
from scipy import stats
import json
import os
from config import CHART_CONFIG

logger = logging.getLogger(__name__)

class ReportGenerator:
    def __init__(
        self,
        data: pd.DataFrame,
        signals: pd.DataFrame,
        performance_metrics: pd.DataFrame,
        output_dir: str = 'reports'
    ):
        """
        初始化报告生成器
        
        Args:
            data: 市场数据
            signals: 交易信号
            performance_metrics: 绩效指标
            output_dir: 输出目录
        """
        self.data = data
        self.signals = signals
        self.performance_metrics = performance_metrics
        self.output_dir = output_dir
        self.chart_config = CHART_CONFIG
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_full_report(
        self,
        title: str = 'Trading Strategy Report',
        include_charts: bool = True,
        include_tables: bool = True
    ) -> str:
        """
        生成完整报告
        
        Args:
            title: 报告标题
            include_charts: 是否包含图表
            include_tables: 是否包含表格
            
        Returns:
            报告文件路径
        """
        try:
            # 创建HTML报告
            html_content = self._generate_html_report(title)
            
            # 生成图表
            if include_charts:
                self._generate_charts()
            
            # 生成表格
            if include_tables:
                self._generate_tables()
            
            # 保存报告
            report_path = os.path.join(self.output_dir, f'report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html')
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            return report_path
            
        except Exception as e:
            logger.error(f"Error generating full report: {str(e)}")
            raise
    
    def _generate_html_report(self, title: str) -> str:
        """生成HTML报告"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .section {{ margin-bottom: 30px; }}
                .chart {{ margin-bottom: 20px; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .metric {{ margin-bottom: 10px; }}
            </style>
        </head>
        <body>
            <h1>{title}</h1>
            <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            
            <div class="section">
                <h2>Performance Summary</h2>
                {self._generate_performance_summary()}
            </div>
            
            <div class="section">
                <h2>Charts</h2>
                <div class="chart">
                    <h3>Equity Curve</h3>
                    <img src="equity_curve.png" alt="Equity Curve">
                </div>
                <div class="chart">
                    <h3>Drawdown</h3>
                    <img src="drawdown.png" alt="Drawdown">
                </div>
                <div class="chart">
                    <h3>Monthly Returns</h3>
                    <img src="monthly_returns.png" alt="Monthly Returns">
                </div>
            </div>
            
            <div class="section">
                <h2>Performance Metrics</h2>
                {self._generate_metrics_table()}
            </div>
            
            <div class="section">
                <h2>Trade History</h2>
                {self._generate_trade_history()}
            </div>
            
            <div class="section">
                <h2>Risk Analysis</h2>
                {self._generate_risk_analysis()}
            </div>
        </body>
        </html>
        """
        return html_content
    
    def _generate_charts(self):
        """生成图表"""
        try:
            # 生成权益曲线
            self._generate_equity_curve()
            
            # 生成回撤图
            self._generate_drawdown_chart()
            
            # 生成月度收益热力图
            self._generate_monthly_returns_heatmap()
            
            # 生成交易分布图
            self._generate_trade_distribution()
            
            # 生成风险指标图
            self._generate_risk_metrics_chart()
            
        except Exception as e:
            logger.error(f"Error generating charts: {str(e)}")
            raise
    
    def _generate_equity_curve(self):
        """生成权益曲线"""
        try:
            # 计算累积收益
            returns = self.data['close'].pct_change()
            cumulative_returns = (1 + returns).cumprod()
            
            # 创建图表
            fig = go.Figure()
            
            # 添加权益曲线
            fig.add_trace(go.Scatter(
                x=self.data.index,
                y=cumulative_returns,
                name='Equity Curve',
                line=dict(color='blue')
            ))
            
            # 添加交易信号点
            for date, signal in self.signals.iterrows():
                if signal['direction'] == 'buy':
                    fig.add_trace(go.Scatter(
                        x=[date],
                        y=[cumulative_returns[date]],
                        mode='markers',
                        name='Buy Signal',
                        marker=dict(color='green', size=10)
                    ))
                elif signal['direction'] == 'sell':
                    fig.add_trace(go.Scatter(
                        x=[date],
                        y=[cumulative_returns[date]],
                        mode='markers',
                        name='Sell Signal',
                        marker=dict(color='red', size=10)
                    ))
            
            # 更新布局
            fig.update_layout(
                title='Equity Curve with Trade Signals',
                xaxis_title='Date',
                yaxis_title='Cumulative Returns',
                height=self.chart_config['chart_height'],
                width=self.chart_config['chart_width'],
                template=self.chart_config['theme']
            )
            
            # 保存图表
            fig.write_image(os.path.join(self.output_dir, 'equity_curve.png'))
            
        except Exception as e:
            logger.error(f"Error generating equity curve: {str(e)}")
            raise
    
    def _generate_drawdown_chart(self):
        """生成回撤图"""
        try:
            # 计算回撤
            returns = self.data['close'].pct_change()
            cumulative_returns = (1 + returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - rolling_max) / rolling_max
            
            # 创建图表
            fig = go.Figure()
            
            # 添加回撤曲线
            fig.add_trace(go.Scatter(
                x=self.data.index,
                y=drawdown,
                name='Drawdown',
                line=dict(color='red')
            ))
            
            # 更新布局
            fig.update_layout(
                title='Drawdown',
                xaxis_title='Date',
                yaxis_title='Drawdown',
                height=self.chart_config['chart_height'],
                width=self.chart_config['chart_width'],
                template=self.chart_config['theme']
            )
            
            # 保存图表
            fig.write_image(os.path.join(self.output_dir, 'drawdown.png'))
            
        except Exception as e:
            logger.error(f"Error generating drawdown chart: {str(e)}")
            raise
    
    def _generate_monthly_returns_heatmap(self):
        """生成月度收益热力图"""
        try:
            # 计算月度收益
            returns = self.data['close'].pct_change()
            monthly_returns = returns.resample('M').agg(lambda x: (1 + x).prod() - 1)
            
            # 创建月度收益矩阵
            monthly_matrix = monthly_returns.groupby([
                monthly_returns.index.year,
                monthly_returns.index.month
            ]).mean().unstack()
            
            # 创建热力图
            fig = px.imshow(
                monthly_matrix,
                labels=dict(
                    x='Month',
                    y='Year',
                    color='Return'
                ),
                aspect='auto',
                color_continuous_scale='RdYlGn'
            )
            
            # 更新布局
            fig.update_layout(
                title='Monthly Returns Heatmap',
                height=self.chart_config['chart_height'],
                width=self.chart_config['chart_width'],
                template=self.chart_config['theme']
            )
            
            # 保存图表
            fig.write_image(os.path.join(self.output_dir, 'monthly_returns.png'))
            
        except Exception as e:
            logger.error(f"Error generating monthly returns heatmap: {str(e)}")
            raise
    
    def _generate_trade_distribution(self):
        """生成交易分布图"""
        try:
            # 计算交易收益
            trade_returns = []
            for date, signal in self.signals.iterrows():
                if signal['direction'] == 'sell':
                    entry_price = signal['entry_price']
                    exit_price = signal['exit_price']
                    trade_return = (exit_price - entry_price) / entry_price
                    trade_returns.append(trade_return)
            
            # 创建直方图
            fig = go.Figure()
            
            fig.add_trace(go.Histogram(
                x=trade_returns,
                name='Trade Returns Distribution',
                nbinsx=50
            ))
            
            # 更新布局
            fig.update_layout(
                title='Trade Returns Distribution',
                xaxis_title='Return',
                yaxis_title='Frequency',
                height=self.chart_config['chart_height'],
                width=self.chart_config['chart_width'],
                template=self.chart_config['theme']
            )
            
            # 保存图表
            fig.write_image(os.path.join(self.output_dir, 'trade_distribution.png'))
            
        except Exception as e:
            logger.error(f"Error generating trade distribution: {str(e)}")
            raise
    
    def _generate_risk_metrics_chart(self):
        """生成风险指标图"""
        try:
            # 创建子图
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Volatility', 'Beta', 'Sharpe Ratio', 'Sortino Ratio')
            )
            
            # 添加波动率
            volatility = self.data['close'].pct_change().rolling(window=20).std() * np.sqrt(252)
            fig.add_trace(
                go.Scatter(x=self.data.index, y=volatility, name='Volatility'),
                row=1, col=1
            )
            
            # 添加Beta
            market_returns = self.data['close'].pct_change()
            beta = market_returns.rolling(window=20).cov(market_returns) / market_returns.rolling(window=20).var()
            fig.add_trace(
                go.Scatter(x=self.data.index, y=beta, name='Beta'),
                row=1, col=2
            )
            
            # 添加夏普比率
            returns = self.data['close'].pct_change()
            sharpe = (returns.rolling(window=20).mean() * 252) / (returns.rolling(window=20).std() * np.sqrt(252))
            fig.add_trace(
                go.Scatter(x=self.data.index, y=sharpe, name='Sharpe Ratio'),
                row=2, col=1
            )
            
            # 添加索提诺比率
            downside_returns = returns[returns < 0]
            sortino = (returns.rolling(window=20).mean() * 252) / (downside_returns.rolling(window=20).std() * np.sqrt(252))
            fig.add_trace(
                go.Scatter(x=self.data.index, y=sortino, name='Sortino Ratio'),
                row=2, col=2
            )
            
            # 更新布局
            fig.update_layout(
                title='Risk Metrics',
                height=self.chart_config['chart_height'] * 2,
                width=self.chart_config['chart_width'] * 2,
                template=self.chart_config['theme']
            )
            
            # 保存图表
            fig.write_image(os.path.join(self.output_dir, 'risk_metrics.png'))
            
        except Exception as e:
            logger.error(f"Error generating risk metrics chart: {str(e)}")
            raise
    
    def _generate_tables(self):
        """生成表格"""
        try:
            # 生成绩效指标表格
            self._generate_metrics_table()
            
            # 生成交易历史表格
            self._generate_trade_history()
            
            # 生成风险分析表格
            self._generate_risk_analysis()
            
        except Exception as e:
            logger.error(f"Error generating tables: {str(e)}")
            raise
    
    def _generate_performance_summary(self) -> str:
        """生成绩效摘要"""
        try:
            # 计算关键指标
            returns = self.data['close'].pct_change()
            cumulative_returns = (1 + returns).cumprod()
            
            summary = {
                'Total Return': f"{cumulative_returns.iloc[-1] - 1:.2%}",
                'Annual Return': f"{(1 + returns.mean()) ** 252 - 1:.2%}",
                'Annual Volatility': f"{returns.std() * np.sqrt(252):.2%}",
                'Sharpe Ratio': f"{(returns.mean() * 252) / (returns.std() * np.sqrt(252)):.2f}",
                'Max Drawdown': f"{(cumulative_returns / cumulative_returns.cummax() - 1).min():.2%}",
                'Win Rate': f"{(returns > 0).mean():.2%}",
                'Profit Factor': f"{abs(returns[returns > 0].sum() / returns[returns < 0].sum()):.2f}"
            }
            
            # 生成HTML表格
            html = '<table>'
            html += '<tr><th>Metric</th><th>Value</th></tr>'
            for metric, value in summary.items():
                html += f'<tr><td>{metric}</td><td>{value}</td></tr>'
            html += '</table>'
            
            return html
            
        except Exception as e:
            logger.error(f"Error generating performance summary: {str(e)}")
            raise
    
    def _generate_metrics_table(self) -> str:
        """生成指标表格"""
        try:
            # 获取所有指标
            metrics = self.performance_metrics.columns.tolist()
            
            # 生成HTML表格
            html = '<table>'
            html += '<tr><th>Date</th>'
            for metric in metrics:
                html += f'<th>{metric}</th>'
            html += '</tr>'
            
            for date, row in self.performance_metrics.iterrows():
                html += f'<tr><td>{date.strftime("%Y-%m-%d")}</td>'
                for metric in metrics:
                    html += f'<td>{row[metric]:.2%}</td>'
                html += '</tr>'
            
            html += '</table>'
            return html
            
        except Exception as e:
            logger.error(f"Error generating metrics table: {str(e)}")
            raise
    
    def _generate_trade_history(self) -> str:
        """生成交易历史"""
        try:
            # 生成HTML表格
            html = '<table>'
            html += '<tr><th>Date</th><th>Direction</th><th>Price</th><th>Size</th><th>Return</th></tr>'
            
            for date, signal in self.signals.iterrows():
                html += f'<tr>'
                html += f'<td>{date.strftime("%Y-%m-%d")}</td>'
                html += f'<td>{signal["direction"]}</td>'
                html += f'<td>{signal["price"]:.2f}</td>'
                html += f'<td>{signal["size"]:.2f}</td>'
                html += f'<td>{signal.get("return", "N/A")}</td>'
                html += '</tr>'
            
            html += '</table>'
            return html
            
        except Exception as e:
            logger.error(f"Error generating trade history: {str(e)}")
            raise
    
    def _generate_risk_analysis(self) -> str:
        """生成风险分析"""
        try:
            # 计算风险指标
            returns = self.data['close'].pct_change()
            
            risk_metrics = {
                'VaR (95%)': f"{np.percentile(returns, 5):.2%}",
                'CVaR (95%)': f"{returns[returns <= np.percentile(returns, 5)].mean():.2%}",
                'Beta': f"{returns.cov(returns) / returns.var():.2f}",
                'Correlation': f"{returns.corr(returns):.2f}",
                'Skewness': f"{returns.skew():.2f}",
                'Kurtosis': f"{returns.kurtosis():.2f}"
            }
            
            # 生成HTML表格
            html = '<table>'
            html += '<tr><th>Risk Metric</th><th>Value</th></tr>'
            for metric, value in risk_metrics.items():
                html += f'<tr><td>{metric}</td><td>{value}</td></tr>'
            html += '</table>'
            
            return html
            
        except Exception as e:
            logger.error(f"Error generating risk analysis: {str(e)}")
            raise 