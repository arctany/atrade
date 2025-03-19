import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
from scipy.optimize import minimize
from strategy_optimizer import StrategyOptimizer
from config import TRADING_STRATEGY, BACKTEST_CONFIG

logger = logging.getLogger(__name__)

class PortfolioOptimizer:
    def __init__(self, data: pd.DataFrame):
        """
        初始化投资组合优化器
        
        Args:
            data: 包含OHLCV数据的DataFrame
        """
        self.data = data.copy()
        self.strategy_optimizer = StrategyOptimizer(data)
        self.strategies = {}
        self.returns = {}
        self.weights = {}
        
    def optimize_portfolio(
        self,
        strategy_names: List[str],
        risk_free_rate: float = 0.02,
        target_return: Optional[float] = None,
        risk_tolerance: float = 0.5
    ) -> Dict:
        """
        优化策略组合
        
        Args:
            strategy_names: 策略名称列表
            risk_free_rate: 无风险利率
            target_return: 目标收益率
            risk_tolerance: 风险承受度 (0-1)
            
        Returns:
            优化结果
        """
        try:
            # 获取每个策略的收益序列
            self._get_strategy_returns(strategy_names)
            
            # 计算收益矩阵
            returns_matrix = pd.DataFrame(self.returns)
            
            # 计算协方差矩阵
            cov_matrix = returns_matrix.cov()
            
            # 计算平均收益
            mean_returns = returns_matrix.mean()
            
            # 定义优化目标函数
            def objective(weights):
                portfolio_return = np.sum(mean_returns * weights)
                portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                
                if target_return is not None:
                    # 最小化风险，约束条件为达到目标收益
                    return portfolio_risk
                else:
                    # 最大化夏普比率
                    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_risk
                    return -sharpe_ratio
            
            # 定义约束条件
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # 权重和为1
                {'type': 'ineq', 'fun': lambda x: x}  # 权重非负
            ]
            
            if target_return is not None:
                constraints.append({
                    'type': 'eq',
                    'fun': lambda x: np.sum(mean_returns * x) - target_return
                })
            
            # 设置初始权重
            n_strategies = len(strategy_names)
            initial_weights = np.array([1/n_strategies] * n_strategies)
            
            # 优化
            result = minimize(
                objective,
                initial_weights,
                method='SLSQP',
                constraints=constraints,
                bounds=[(0, 1) for _ in range(n_strategies)]
            )
            
            # 记录优化结果
            self.weights = dict(zip(strategy_names, result.x))
            
            # 计算优化后的组合指标
            portfolio_metrics = self._calculate_portfolio_metrics(
                returns_matrix,
                result.x,
                risk_free_rate
            )
            
            return {
                'weights': self.weights,
                'metrics': portfolio_metrics,
                'optimization_status': result.message,
                'optimization_success': result.success
            }
            
        except Exception as e:
            logger.error(f"Error in portfolio optimization: {str(e)}")
            return {}
    
    def generate_portfolio_report(
        self,
        strategy_names: List[str],
        weights: Optional[Dict[str, float]] = None
    ) -> Dict:
        """
        生成投资组合报告
        
        Args:
            strategy_names: 策略名称列表
            weights: 策略权重
            
        Returns:
            投资组合报告
        """
        try:
            # 使用提供的权重或优化后的权重
            weights = weights or self.weights
            
            if not weights:
                return {}
            
            # 获取每个策略的性能指标
            strategy_metrics = {}
            for strategy_name in strategy_names:
                metrics = self.strategy_optimizer.analyze_performance(strategy_name)
                if metrics:
                    strategy_metrics[strategy_name] = metrics
            
            # 计算组合整体指标
            portfolio_metrics = self._calculate_portfolio_metrics(
                pd.DataFrame(self.returns),
                np.array([weights[name] for name in strategy_names])
            )
            
            # 生成报告
            report = {
                'portfolio_weights': weights,
                'strategy_metrics': strategy_metrics,
                'portfolio_metrics': portfolio_metrics,
                'generated_at': datetime.now().isoformat()
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error in portfolio report generation: {str(e)}")
            return {}
    
    def _get_strategy_returns(self, strategy_names: List[str]):
        """获取策略收益序列"""
        try:
            for strategy_name in strategy_names:
                if strategy_name not in self.returns:
                    # 获取策略回测结果
                    backtest_results = self.strategy_optimizer.analyze_performance(strategy_name)
                    
                    if backtest_results and 'backtest_results' in backtest_results:
                        trades = backtest_results['backtest_results']['trades']
                        
                        # 计算每日收益
                        daily_returns = pd.Series(index=self.data.index, dtype=float)
                        for trade in trades:
                            if trade['type'] == 'SELL':
                                daily_returns[trade['time']] = trade['profit']
                        
                        # 填充空值
                        daily_returns = daily_returns.fillna(0)
                        
                        self.returns[strategy_name] = daily_returns
                        
        except Exception as e:
            logger.error(f"Error getting strategy returns: {str(e)}")
    
    def _calculate_portfolio_metrics(
        self,
        returns_matrix: pd.DataFrame,
        weights: np.ndarray,
        risk_free_rate: float = 0.02
    ) -> Dict:
        """计算投资组合指标"""
        try:
            metrics = {}
            
            # 计算组合收益
            portfolio_returns = returns_matrix.dot(weights)
            
            # 计算年化收益率
            annual_return = portfolio_returns.mean() * 252
            
            # 计算年化波动率
            annual_volatility = portfolio_returns.std() * np.sqrt(252)
            
            # 计算夏普比率
            sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility
            
            # 计算最大回撤
            cumulative_returns = (1 + portfolio_returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdowns = cumulative_returns / rolling_max - 1
            max_drawdown = drawdowns.min()
            
            # 计算索提诺比率
            downside_returns = portfolio_returns[portfolio_returns < 0]
            downside_std = downside_returns.std() * np.sqrt(252)
            sortino_ratio = (annual_return - risk_free_rate) / downside_std
            
            # 计算信息比率
            excess_returns = portfolio_returns - risk_free_rate/252
            tracking_error = excess_returns.std() * np.sqrt(252)
            information_ratio = (annual_return - risk_free_rate) / tracking_error
            
            metrics.update({
                'annual_return': annual_return,
                'annual_volatility': annual_volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'sortino_ratio': sortino_ratio,
                'information_ratio': information_ratio,
                'risk_free_rate': risk_free_rate
            })
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating portfolio metrics: {str(e)}")
            return {} 