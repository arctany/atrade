import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import os
from config import CHART_CONFIG

logger = logging.getLogger(__name__)

class StrategyComparator:
    def __init__(
        self,
        strategies: Dict[str, Dict],
        output_dir: str = 'comparison_reports'
    ):
        """
        初始化策略比较器
        
        Args:
            strategies: 策略字典，格式为 {strategy_name: {data, signals, performance_metrics}}
            output_dir: 输出目录
        """
        self.strategies = strategies
        self.output_dir = output_dir
        self.chart_config = CHART_CONFIG
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
    
    def compare_strategies(
        self,
        title: str = 'Strategy Comparison Report',
        include_charts: bool = True,
        include_tables: bool = True
    ) -> str:
        """
        比较策略
        
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
            report_path = os.path.join(self.output_dir, f'comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html')
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            return report_path
            
        except Exception as e:
            logger.error(f"Error comparing strategies: {str(e)}")
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
                    <h3>Equity Curves</h3>
                    <img src="equity_curves.png" alt="Equity Curves">
                </div>
                <div class="chart">
                    <h3>Drawdown Comparison</h3>
                    <img src="drawdown_comparison.png" alt="Drawdown Comparison">
                </div>
                <div class="chart">
                    <h3>Monthly Returns Heatmap</h3>
                    <img src="monthly_returns.png" alt="Monthly Returns">
                </div>
            </div>
            
            <div class="section">
                <h2>Performance Metrics</h2>
                {self._generate_metrics_table()}
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
            # 生成权益曲线对比图
            self._generate_equity_curves()
            
            # 生成回撤对比图
            self._generate_drawdown_comparison()
            
            # 生成月度收益热力图
            self._generate_monthly_returns_heatmap()
            
            # 生成风险指标对比图
            self._generate_risk_metrics_chart()
            
        except Exception as e:
            logger.error(f"Error generating charts: {str(e)}")
            raise
    
    def _generate_equity_curves(self):
        """生成权益曲线对比图"""
        try:
            # 创建图表
            fig = go.Figure()
            
            # 添加每个策略的权益曲线
            for name, strategy in self.strategies.items():
                returns = strategy['data']['close'].pct_change()
                cumulative_returns = (1 + returns).cumprod()
                
                fig.add_trace(go.Scatter(
                    x=strategy['data'].index,
                    y=cumulative_returns,
                    name=name,
                    line=dict(width=2)
                ))
            
            # 更新布局
            fig.update_layout(
                title='Equity Curves Comparison',
                xaxis_title='Date',
                yaxis_title='Cumulative Returns',
                height=self.chart_config['chart_height'],
                width=self.chart_config['chart_width'],
                template=self.chart_config['theme']
            )
            
            # 保存图表
            fig.write_image(os.path.join(self.output_dir, 'equity_curves.png'))
            
        except Exception as e:
            logger.error(f"Error generating equity curves: {str(e)}")
            raise
    
    def _generate_drawdown_comparison(self):
        """生成回撤对比图"""
        try:
            # 创建图表
            fig = go.Figure()
            
            # 添加每个策略的回撤曲线
            for name, strategy in self.strategies.items():
                returns = strategy['data']['close'].pct_change()
                cumulative_returns = (1 + returns).cumprod()
                rolling_max = cumulative_returns.expanding().max()
                drawdown = (cumulative_returns - rolling_max) / rolling_max
                
                fig.add_trace(go.Scatter(
                    x=strategy['data'].index,
                    y=drawdown,
                    name=name,
                    line=dict(width=2)
                ))
            
            # 更新布局
            fig.update_layout(
                title='Drawdown Comparison',
                xaxis_title='Date',
                yaxis_title='Drawdown',
                height=self.chart_config['chart_height'],
                width=self.chart_config['chart_width'],
                template=self.chart_config['theme']
            )
            
            # 保存图表
            fig.write_image(os.path.join(self.output_dir, 'drawdown_comparison.png'))
            
        except Exception as e:
            logger.error(f"Error generating drawdown comparison: {str(e)}")
            raise
    
    def _generate_monthly_returns_heatmap(self):
        """生成月度收益热力图"""
        try:
            # 创建子图
            fig = make_subplots(
                rows=len(self.strategies),
                cols=1,
                subplot_titles=list(self.strategies.keys())
            )
            
            # 添加每个策略的月度收益热力图
            for i, (name, strategy) in enumerate(self.strategies.items(), 1):
                returns = strategy['data']['close'].pct_change()
                monthly_returns = returns.resample('M').agg(lambda x: (1 + x).prod() - 1)
                
                monthly_matrix = monthly_returns.groupby([
                    monthly_returns.index.year,
                    monthly_returns.index.month
                ]).mean().unstack()
                
                fig.add_trace(
                    go.Heatmap(
                        z=monthly_matrix.values,
                        x=monthly_matrix.columns,
                        y=monthly_matrix.index,
                        colorscale='RdYlGn',
                        name=name,
                        showscale=True
                    ),
                    row=i, col=1
                )
            
            # 更新布局
            fig.update_layout(
                title='Monthly Returns Heatmap Comparison',
                height=self.chart_config['chart_height'] * len(self.strategies),
                width=self.chart_config['chart_width'],
                template=self.chart_config['theme']
            )
            
            # 保存图表
            fig.write_image(os.path.join(self.output_dir, 'monthly_returns.png'))
            
        except Exception as e:
            logger.error(f"Error generating monthly returns heatmap: {str(e)}")
            raise
    
    def _generate_risk_metrics_chart(self):
        """生成风险指标对比图"""
        try:
            # 创建子图
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Volatility', 'Beta', 'Sharpe Ratio', 'Sortino Ratio')
            )
            
            # 计算每个策略的风险指标
            for name, strategy in self.strategies.items():
                returns = strategy['data']['close'].pct_change()
                
                # 波动率
                volatility = returns.rolling(window=20).std() * np.sqrt(252)
                fig.add_trace(
                    go.Scatter(
                        x=strategy['data'].index,
                        y=volatility,
                        name=name,
                        line=dict(width=2)
                    ),
                    row=1, col=1
                )
                
                # Beta
                market_returns = returns
                beta = market_returns.rolling(window=20).cov(market_returns) / market_returns.rolling(window=20).var()
                fig.add_trace(
                    go.Scatter(
                        x=strategy['data'].index,
                        y=beta,
                        name=name,
                        line=dict(width=2)
                    ),
                    row=1, col=2
                )
                
                # 夏普比率
                sharpe = (returns.rolling(window=20).mean() * 252) / (returns.rolling(window=20).std() * np.sqrt(252))
                fig.add_trace(
                    go.Scatter(
                        x=strategy['data'].index,
                        y=sharpe,
                        name=name,
                        line=dict(width=2)
                    ),
                    row=2, col=1
                )
                
                # 索提诺比率
                downside_returns = returns[returns < 0]
                sortino = (returns.rolling(window=20).mean() * 252) / (downside_returns.rolling(window=20).std() * np.sqrt(252))
                fig.add_trace(
                    go.Scatter(
                        x=strategy['data'].index,
                        y=sortino,
                        name=name,
                        line=dict(width=2)
                    ),
                    row=2, col=2
                )
            
            # 更新布局
            fig.update_layout(
                title='Risk Metrics Comparison',
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
            
            # 生成风险分析表格
            self._generate_risk_analysis()
            
        except Exception as e:
            logger.error(f"Error generating tables: {str(e)}")
            raise
    
    def _generate_performance_summary(self) -> str:
        """生成绩效摘要"""
        try:
            # 计算每个策略的关键指标
            summary = {}
            for name, strategy in self.strategies.items():
                returns = strategy['data']['close'].pct_change()
                cumulative_returns = (1 + returns).cumprod()
                
                summary[name] = {
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
            html += '<tr><th>Strategy</th>'
            metrics = list(next(iter(summary.values())).keys())
            for metric in metrics:
                html += f'<th>{metric}</th>'
            html += '</tr>'
            
            for name, metrics in summary.items():
                html += f'<tr><td>{name}</td>'
                for metric in metrics.values():
                    html += f'<td>{metric}</td>'
                html += '</tr>'
            
            html += '</table>'
            return html
            
        except Exception as e:
            logger.error(f"Error generating performance summary: {str(e)}")
            raise
    
    def _generate_metrics_table(self) -> str:
        """生成指标表格"""
        try:
            # 获取所有策略的指标
            metrics = {}
            for name, strategy in self.strategies.items():
                metrics[name] = strategy['performance_metrics']
            
            # 生成HTML表格
            html = '<table>'
            html += '<tr><th>Strategy</th><th>Date</th>'
            metric_names = list(next(iter(metrics.values())).columns)
            for metric in metric_names:
                html += f'<th>{metric}</th>'
            html += '</tr>'
            
            for name, df in metrics.items():
                for date, row in df.iterrows():
                    html += f'<tr><td>{name}</td><td>{date.strftime("%Y-%m-%d")}</td>'
                    for metric in metric_names:
                        html += f'<td>{row[metric]:.2%}</td>'
                    html += '</tr>'
            
            html += '</table>'
            return html
            
        except Exception as e:
            logger.error(f"Error generating metrics table: {str(e)}")
            raise
    
    def _generate_risk_analysis(self) -> str:
        """生成风险分析"""
        try:
            # 计算每个策略的风险指标
            risk_metrics = {}
            for name, strategy in self.strategies.items():
                returns = strategy['data']['close'].pct_change()
                
                risk_metrics[name] = {
                    'VaR (95%)': f"{np.percentile(returns, 5):.2%}",
                    'CVaR (95%)': f"{returns[returns <= np.percentile(returns, 5)].mean():.2%}",
                    'Beta': f"{returns.cov(returns) / returns.var():.2f}",
                    'Correlation': f"{returns.corr(returns):.2f}",
                    'Skewness': f"{returns.skew():.2f}",
                    'Kurtosis': f"{returns.kurtosis():.2f}"
                }
            
            # 生成HTML表格
            html = '<table>'
            html += '<tr><th>Strategy</th>'
            metrics = list(next(iter(risk_metrics.values())).keys())
            for metric in metrics:
                html += f'<th>{metric}</th>'
            html += '</tr>'
            
            for name, metrics in risk_metrics.items():
                html += f'<tr><td>{name}</td>'
                for metric in metrics.values():
                    html += f'<td>{metric}</td>'
                html += '</tr>'
            
            html += '</table>'
            return html
            
        except Exception as e:
            logger.error(f"Error generating risk analysis: {str(e)}")
            raise
    
    def get_best_strategy(self, metric: str) -> Tuple[str, float]:
        """
        获取最佳策略
        
        Args:
            metric: 评估指标
            
        Returns:
            最佳策略名称和指标值
        """
        try:
            best_strategy = None
            best_value = float('-inf')
            
            for name, strategy in self.strategies.items():
                returns = strategy['data']['close'].pct_change()
                cumulative_returns = (1 + returns).cumprod()
                
                if metric == 'Total Return':
                    value = cumulative_returns.iloc[-1] - 1
                elif metric == 'Annual Return':
                    value = (1 + returns.mean()) ** 252 - 1
                elif metric == 'Sharpe Ratio':
                    value = (returns.mean() * 252) / (returns.std() * np.sqrt(252))
                elif metric == 'Max Drawdown':
                    value = -(cumulative_returns / cumulative_returns.cummax() - 1).min()
                elif metric == 'Win Rate':
                    value = (returns > 0).mean()
                elif metric == 'Profit Factor':
                    value = abs(returns[returns > 0].sum() / returns[returns < 0].sum())
                else:
                    raise ValueError(f"Unknown metric: {metric}")
                
                if value > best_value:
                    best_value = value
                    best_strategy = name
            
            return best_strategy, best_value
            
        except Exception as e:
            logger.error(f"Error getting best strategy: {str(e)}")
            raise
    
    def get_strategy_rankings(self, metric: str) -> List[Tuple[str, float]]:
        """
        获取策略排名
        
        Args:
            metric: 评估指标
            
        Returns:
            策略排名列表，按指标值降序排列
        """
        try:
            rankings = []
            
            for name, strategy in self.strategies.items():
                returns = strategy['data']['close'].pct_change()
                cumulative_returns = (1 + returns).cumprod()
                
                if metric == 'Total Return':
                    value = cumulative_returns.iloc[-1] - 1
                elif metric == 'Annual Return':
                    value = (1 + returns.mean()) ** 252 - 1
                elif metric == 'Sharpe Ratio':
                    value = (returns.mean() * 252) / (returns.std() * np.sqrt(252))
                elif metric == 'Max Drawdown':
                    value = -(cumulative_returns / cumulative_returns.cummax() - 1).min()
                elif metric == 'Win Rate':
                    value = (returns > 0).mean()
                elif metric == 'Profit Factor':
                    value = abs(returns[returns > 0].sum() / returns[returns < 0].sum())
                else:
                    raise ValueError(f"Unknown metric: {metric}")
                
                rankings.append((name, value))
            
            # 按指标值降序排列
            rankings.sort(key=lambda x: x[1], reverse=True)
            return rankings
            
        except Exception as e:
            logger.error(f"Error getting strategy rankings: {str(e)}")
            raise 