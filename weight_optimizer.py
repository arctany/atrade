import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
import os
from scipy.optimize import minimize
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from config import OPTIMIZATION_CONFIG

logger = logging.getLogger(__name__)

class WeightOptimizer:
    def __init__(
        self,
        returns: pd.DataFrame,
        risk_free_rate: float = 0.02,
        rebalance_period: str = 'M'
    ):
        """
        初始化权重优化器
        
        Args:
            returns: 策略收益率数据
            risk_free_rate: 无风险利率
            rebalance_period: 再平衡周期（D/W/M/Q/Y）
        """
        self.returns = returns
        self.risk_free_rate = risk_free_rate
        self.rebalance_period = rebalance_period
        self.optimization_config = OPTIMIZATION_CONFIG
        
        # 计算年化无风险利率
        self.annual_rf = (1 + risk_free_rate) ** (252 / self._get_period_days()) - 1
        
        # 初始化结果存储
        self.weights_history = pd.DataFrame()
        self.performance_history = pd.DataFrame()
    
    def optimize_weights(
        self,
        method: str = 'sharpe',
        constraints: Optional[Dict] = None,
        target_return: Optional[float] = None
    ) -> Dict:
        """
        优化权重
        
        Args:
            method: 优化方法（sharpe/min_variance/max_return）
            constraints: 约束条件
            target_return: 目标收益率
            
        Returns:
            优化结果
        """
        try:
            # 获取优化参数
            params = self.optimization_config['optimization_params'][method]
            
            # 设置约束条件
            if constraints is None:
                constraints = self.optimization_config['default_constraints']
            
            # 设置目标函数
            if method == 'sharpe':
                objective = self._sharpe_ratio
            elif method == 'min_variance':
                objective = self._portfolio_variance
            elif method == 'max_return':
                objective = self._portfolio_return
            else:
                raise ValueError(f"Unsupported optimization method: {method}")
            
            # 设置边界条件
            bounds = [(0, 1) for _ in range(len(self.returns.columns))]
            
            # 设置等式约束
            constraints_list = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
            ]
            
            # 添加目标收益率约束
            if target_return is not None:
                constraints_list.append({
                    'type': 'eq',
                    'fun': lambda x: self._portfolio_return(x) - target_return
                })
            
            # 添加自定义约束
            for constraint in constraints.values():
                if constraint['type'] == 'ineq':
                    constraints_list.append({
                        'type': 'ineq',
                        'fun': constraint['fun']
                    })
            
            # 优化
            result = minimize(
                objective,
                x0=np.array([1/len(self.returns.columns)] * len(self.returns.columns)),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints_list,
                options={'maxiter': params['max_iterations']}
            )
            
            # 计算优化结果
            optimal_weights = result.x
            portfolio_return = self._portfolio_return(optimal_weights)
            portfolio_volatility = np.sqrt(self._portfolio_variance(optimal_weights))
            sharpe_ratio = self._sharpe_ratio(optimal_weights)
            
            # 更新历史记录
            self._update_history(optimal_weights, portfolio_return, portfolio_volatility, sharpe_ratio)
            
            return {
                'weights': dict(zip(self.returns.columns, optimal_weights)),
                'return': portfolio_return,
                'volatility': portfolio_volatility,
                'sharpe_ratio': sharpe_ratio,
                'status': result.status,
                'message': result.message
            }
            
        except Exception as e:
            logger.error(f"Error optimizing weights: {str(e)}")
            raise
    
    def rebalance_portfolio(
        self,
        current_weights: Dict[str, float],
        method: str = 'sharpe',
        constraints: Optional[Dict] = None,
        target_return: Optional[float] = None
    ) -> Dict:
        """
        再平衡投资组合
        
        Args:
            current_weights: 当前权重
            method: 优化方法
            constraints: 约束条件
            target_return: 目标收益率
            
        Returns:
            再平衡结果
        """
        try:
            # 优化权重
            optimization_result = self.optimize_weights(
                method=method,
                constraints=constraints,
                target_return=target_return
            )
            
            # 计算需要调整的权重
            weight_changes = {
                asset: optimization_result['weights'][asset] - current_weights[asset]
                for asset in current_weights.keys()
            }
            
            # 计算交易成本
            transaction_costs = self._calculate_transaction_costs(weight_changes)
            
            return {
                'new_weights': optimization_result['weights'],
                'weight_changes': weight_changes,
                'transaction_costs': transaction_costs,
                'performance_metrics': {
                    'return': optimization_result['return'],
                    'volatility': optimization_result['volatility'],
                    'sharpe_ratio': optimization_result['sharpe_ratio']
                }
            }
            
        except Exception as e:
            logger.error(f"Error rebalancing portfolio: {str(e)}")
            raise
    
    def generate_weight_report(
        self,
        output_dir: str = 'weight_reports',
        include_charts: bool = True,
        include_tables: bool = True
    ) -> str:
        """
        生成权重报告
        
        Args:
            output_dir: 输出目录
            include_charts: 是否包含图表
            include_tables: 是否包含表格
            
        Returns:
            报告文件路径
        """
        try:
            # 创建输出目录
            os.makedirs(output_dir, exist_ok=True)
            
            # 生成HTML报告
            html_content = self._generate_html_report(include_charts, include_tables)
            
            # 保存报告
            report_path = os.path.join(output_dir, f'weight_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html')
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            return report_path
            
        except Exception as e:
            logger.error(f"Error generating weight report: {str(e)}")
            raise
    
    def _get_period_days(self) -> int:
        """获取周期天数"""
        period_days = {
            'D': 1,
            'W': 5,
            'M': 21,
            'Q': 63,
            'Y': 252
        }
        return period_days.get(self.rebalance_period, 252)
    
    def _portfolio_return(self, weights: np.ndarray) -> float:
        """计算投资组合收益率"""
        return np.sum(self.returns.mean() * weights) * self._get_period_days()
    
    def _portfolio_variance(self, weights: np.ndarray) -> float:
        """计算投资组合方差"""
        return np.dot(weights.T, np.dot(self.returns.cov() * self._get_period_days(), weights))
    
    def _sharpe_ratio(self, weights: np.ndarray) -> float:
        """计算夏普比率"""
        portfolio_return = self._portfolio_return(weights)
        portfolio_volatility = np.sqrt(self._portfolio_variance(weights))
        return (portfolio_return - self.annual_rf) / portfolio_volatility
    
    def _calculate_transaction_costs(self, weight_changes: Dict[str, float]) -> float:
        """计算交易成本"""
        transaction_costs = 0
        for asset, change in weight_changes.items():
            if abs(change) > self.optimization_config['transaction_cost_threshold']:
                transaction_costs += abs(change) * self.optimization_config['transaction_costs'][asset]
        return transaction_costs
    
    def _update_history(
        self,
        weights: np.ndarray,
        portfolio_return: float,
        portfolio_volatility: float,
        sharpe_ratio: float
    ):
        """更新历史记录"""
        timestamp = datetime.now()
        
        # 更新权重历史
        weights_df = pd.DataFrame(
            [dict(zip(self.returns.columns, weights))],
            index=[timestamp]
        )
        self.weights_history = pd.concat([self.weights_history, weights_df])
        
        # 更新性能历史
        performance_df = pd.DataFrame(
            [{
                'return': portfolio_return,
                'volatility': portfolio_volatility,
                'sharpe_ratio': sharpe_ratio
            }],
            index=[timestamp]
        )
        self.performance_history = pd.concat([self.performance_history, performance_df])
    
    def _generate_html_report(self, include_charts: bool, include_tables: bool) -> str:
        """生成HTML报告"""
        try:
            html_content = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Weight Optimization Report</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    .section { margin-bottom: 30px; }
                    table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
                    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                    th { background-color: #f2f2f2; }
                    .chart { margin-bottom: 30px; }
                </style>
            </head>
            <body>
                <h1>Weight Optimization Report</h1>
                <p>Generated at: {timestamp}</p>
            """.format(timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            
            if include_charts:
                html_content += self._generate_charts()
            
            if include_tables:
                html_content += self._generate_tables()
            
            html_content += """
            </body>
            </html>
            """
            
            return html_content
            
        except Exception as e:
            logger.error(f"Error generating HTML report: {str(e)}")
            raise
    
    def _generate_charts(self) -> str:
        """生成图表"""
        try:
            # 创建权重历史图表
            weights_fig = go.Figure()
            for column in self.weights_history.columns:
                weights_fig.add_trace(go.Scatter(
                    x=self.weights_history.index,
                    y=self.weights_history[column],
                    name=column,
                    mode='lines+markers'
                ))
            weights_fig.update_layout(
                title='Weight History',
                xaxis_title='Date',
                yaxis_title='Weight',
                yaxis=dict(tickformat='.2%')
            )
            weights_html = weights_fig.to_html(full_html=False)
            
            # 创建性能历史图表
            performance_fig = go.Figure()
            for column in self.performance_history.columns:
                performance_fig.add_trace(go.Scatter(
                    x=self.performance_history.index,
                    y=self.performance_history[column],
                    name=column,
                    mode='lines+markers'
                ))
            performance_fig.update_layout(
                title='Performance History',
                xaxis_title='Date',
                yaxis_title='Value'
            )
            performance_html = performance_fig.to_html(full_html=False)
            
            return f"""
                <div class="section">
                    <h2>Charts</h2>
                    <div class="chart">
                        {weights_html}
                    </div>
                    <div class="chart">
                        {performance_html}
                    </div>
                </div>
            """
            
        except Exception as e:
            logger.error(f"Error generating charts: {str(e)}")
            raise
    
    def _generate_tables(self) -> str:
        """生成表格"""
        try:
            # 生成权重历史表格
            weights_table = self.weights_history.to_html(
                float_format=lambda x: '{:.2%}'.format(x)
            )
            
            # 生成性能历史表格
            performance_table = self.performance_history.to_html(
                float_format=lambda x: '{:.4f}'.format(x)
            )
            
            return f"""
                <div class="section">
                    <h2>Tables</h2>
                    <h3>Weight History</h3>
                    {weights_table}
                    <h3>Performance History</h3>
                    {performance_table}
                </div>
            """
            
        except Exception as e:
            logger.error(f"Error generating tables: {str(e)}")
            raise 