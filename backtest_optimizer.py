import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
from config import BACKTEST_CONFIG, TRADING_STRATEGY, RISK_MANAGEMENT_CONFIG
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os

logger = logging.getLogger(__name__)

class BacktestOptimizer:
    def __init__(self, data: pd.DataFrame):
        """
        初始化回测优化器
        
        Args:
            data: 包含OHLCV数据的DataFrame
        """
        self.data = data.copy()
        self.initial_capital = BACKTEST_CONFIG['initial_capital']
        self.commission_rate = BACKTEST_CONFIG['commission_rate']
        self.slippage = BACKTEST_CONFIG['slippage']
        self.positions = {}
        self.trades = []
        self.equity_curve = []
        self.performance_metrics = {}
        
    def run_backtest(
        self,
        strategy_name: str,
        params: Dict = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict:
        """
        运行回测
        
        Args:
            strategy_name: 策略名称
            params: 策略参数
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            回测结果
        """
        try:
            # 设置回测时间范围
            if start_date:
                self.data = self.data[self.data.index >= start_date]
            if end_date:
                self.data = self.data[self.data.index <= end_date]
            
            # 初始化回测状态
            self.positions = {}
            self.trades = []
            self.equity_curve = []
            current_capital = self.initial_capital
            
            # 运行回测
            for date, row in self.data.iterrows():
                # 获取策略信号
                signal = self._get_strategy_signal(strategy_name, row, params)
                
                # 更新持仓
                current_capital = self._update_positions(signal, row, current_capital)
                
                # 记录权益曲线
                self.equity_curve.append({
                    'date': date,
                    'equity': current_capital
                })
            
            # 计算性能指标
            self._calculate_performance_metrics()
            
            # 生成回测报告
            report = self._generate_backtest_report()
            
            return report
            
        except Exception as e:
            logger.error(f"Error running backtest: {str(e)}")
            return {}
    
    def optimize_parameters(
        self,
        strategy_name: str,
        param_grid: Dict[str, List],
        metric: str = 'sharpe_ratio',
        n_jobs: int = -1
    ) -> Dict:
        """
        优化策略参数
        
        Args:
            strategy_name: 策略名称
            param_grid: 参数网格
            metric: 优化指标
            n_jobs: 并行任务数
            
        Returns:
            优化结果
        """
        try:
            from itertools import product
            from concurrent.futures import ProcessPoolExecutor
            
            # 生成参数组合
            param_combinations = list(product(*param_grid.values()))
            param_names = list(param_grid.keys())
            
            # 并行运行回测
            results = []
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                futures = []
                for params in param_combinations:
                    param_dict = dict(zip(param_names, params))
                    futures.append(
                        executor.submit(self.run_backtest, strategy_name, param_dict)
                    )
                
                for future in futures:
                    results.append(future.result())
            
            # 根据优化指标排序结果
            sorted_results = sorted(
                results,
                key=lambda x: x['performance_metrics'][metric],
                reverse=True
            )
            
            return {
                'best_params': sorted_results[0]['parameters'],
                'best_metrics': sorted_results[0]['performance_metrics'],
                'all_results': sorted_results
            }
            
        except Exception as e:
            logger.error(f"Error optimizing parameters: {str(e)}")
            return {}
    
    def walk_forward_analysis(
        self,
        strategy_name: str,
        params: Dict,
        train_size: int = 252,
        test_size: int = 63,
        step_size: int = 21
    ) -> Dict:
        """
        进行前向分析
        
        Args:
            strategy_name: 策略名称
            params: 策略参数
            train_size: 训练集大小
            test_size: 测试集大小
            step_size: 步长
            
        Returns:
            分析结果
        """
        try:
            results = []
            total_days = len(self.data)
            
            for i in range(0, total_days - train_size - test_size, step_size):
                # 划分训练集和测试集
                train_data = self.data.iloc[i:i+train_size]
                test_data = self.data.iloc[i+train_size:i+train_size+test_size]
                
                # 在训练集上优化参数
                train_optimizer = BacktestOptimizer(train_data)
                train_results = train_optimizer.optimize_parameters(
                    strategy_name,
                    params,
                    metric='sharpe_ratio'
                )
                
                # 使用最优参数在测试集上运行回测
                test_optimizer = BacktestOptimizer(test_data)
                test_results = test_optimizer.run_backtest(
                    strategy_name,
                    train_results['best_params']
                )
                
                results.append({
                    'period': {
                        'train_start': train_data.index[0],
                        'train_end': train_data.index[-1],
                        'test_start': test_data.index[0],
                        'test_end': test_data.index[-1]
                    },
                    'train_metrics': train_results['best_metrics'],
                    'test_metrics': test_results['performance_metrics']
                })
            
            return {
                'periods': results,
                'summary': self._summarize_walk_forward_results(results)
            }
            
        except Exception as e:
            logger.error(f"Error in walk forward analysis: {str(e)}")
            return {}
    
    def monte_carlo_simulation(
        self,
        strategy_name: str,
        params: Dict,
        n_simulations: int = 1000,
        confidence_level: float = 0.95
    ) -> Dict:
        """
        进行蒙特卡洛模拟
        
        Args:
            strategy_name: 策略名称
            params: 策略参数
            n_simulations: 模拟次数
            confidence_level: 置信水平
            
        Returns:
            模拟结果
        """
        try:
            # 计算收益率和波动率
            returns = self.data.pct_change().dropna()
            mu = returns.mean()
            sigma = returns.std()
            
            # 生成模拟数据
            simulated_returns = np.random.normal(
                mu,
                sigma,
                (n_simulations, len(self.data))
            )
            
            # 计算模拟路径
            simulated_paths = []
            for i in range(n_simulations):
                path = self.initial_capital * (1 + simulated_returns[i]).cumprod()
                simulated_paths.append(path)
            
            # 计算置信区间
            sorted_paths = np.sort(simulated_paths, axis=0)
            lower_idx = int((1 - confidence_level) / 2 * n_simulations)
            upper_idx = int((1 + confidence_level) / 2 * n_simulations)
            
            lower_bound = sorted_paths[lower_idx]
            upper_bound = sorted_paths[upper_idx]
            
            return {
                'simulated_paths': simulated_paths,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'mean_path': np.mean(simulated_paths, axis=0),
                'confidence_level': confidence_level
            }
            
        except Exception as e:
            logger.error(f"Error in Monte Carlo simulation: {str(e)}")
            return {}
    
    def _get_strategy_signal(
        self,
        strategy_name: str,
        data: pd.Series,
        params: Dict
    ) -> Dict:
        """获取策略信号"""
        try:
            # 这里应该调用策略模块获取信号
            # 示例实现
            signal = {
                'action': 'HOLD',
                'symbol': data.name,
                'price': data['Close'],
                'size': 0
            }
            return signal
            
        except Exception as e:
            logger.error(f"Error getting strategy signal: {str(e)}")
            return {}
    
    def _update_positions(
        self,
        signal: Dict,
        data: pd.Series,
        current_capital: float
    ) -> float:
        """更新持仓"""
        try:
            if signal['action'] == 'BUY':
                # 计算交易成本
                commission = signal['price'] * signal['size'] * self.commission_rate
                slippage = signal['price'] * signal['size'] * self.slippage
                total_cost = signal['price'] * signal['size'] + commission + slippage
                
                # 检查是否有足够资金
                if total_cost <= current_capital:
                    # 更新持仓
                    self.positions[signal['symbol']] = {
                        'size': signal['size'],
                        'entry_price': signal['price'],
                        'entry_time': data.name
                    }
                    
                    # 记录交易
                    self.trades.append({
                        'type': 'BUY',
                        'symbol': signal['symbol'],
                        'price': signal['price'],
                        'size': signal['size'],
                        'cost': total_cost,
                        'time': data.name
                    })
                    
                    current_capital -= total_cost
                    
            elif signal['action'] == 'SELL' and signal['symbol'] in self.positions:
                position = self.positions[signal['symbol']]
                
                # 计算交易成本
                commission = signal['price'] * position['size'] * self.commission_rate
                slippage = signal['price'] * position['size'] * self.slippage
                total_cost = commission + slippage
                
                # 计算收益
                revenue = signal['price'] * position['size']
                profit = revenue - position['entry_price'] * position['size'] - total_cost
                
                # 记录交易
                self.trades.append({
                    'type': 'SELL',
                    'symbol': signal['symbol'],
                    'price': signal['price'],
                    'size': position['size'],
                    'revenue': revenue,
                    'cost': total_cost,
                    'profit': profit,
                    'time': data.name
                })
                
                # 更新资金
                current_capital += revenue - total_cost
                
                # 移除持仓
                del self.positions[signal['symbol']]
            
            return current_capital
            
        except Exception as e:
            logger.error(f"Error updating positions: {str(e)}")
            return current_capital
    
    def _calculate_performance_metrics(self):
        """计算性能指标"""
        try:
            # 计算收益率
            returns = pd.Series(self.equity_curve).pct_change().dropna()
            
            # 计算年化收益率
            annual_return = returns.mean() * 252
            
            # 计算年化波动率
            annual_volatility = returns.std() * np.sqrt(252)
            
            # 计算夏普比率
            risk_free_rate = RISK_MANAGEMENT_CONFIG['risk_free_rate']
            sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility
            
            # 计算最大回撤
            cumulative_returns = (1 + returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdowns = cumulative_returns / rolling_max - 1
            max_drawdown = drawdowns.min()
            
            # 计算胜率
            trade_returns = [t['profit'] for t in self.trades if t['type'] == 'SELL']
            win_rate = len([r for r in trade_returns if r > 0]) / len(trade_returns)
            
            # 计算盈亏比
            profits = [r for r in trade_returns if r > 0]
            losses = [r for r in trade_returns if r < 0]
            profit_factor = abs(sum(profits) / sum(losses)) if losses else float('inf')
            
            # 计算平均收益和损失
            avg_profit = np.mean(profits) if profits else 0
            avg_loss = np.mean(losses) if losses else 0
            
            # 计算最大连续盈利和亏损
            consecutive_wins = 0
            consecutive_losses = 0
            max_consecutive_wins = 0
            max_consecutive_losses = 0
            
            for r in trade_returns:
                if r > 0:
                    consecutive_wins += 1
                    consecutive_losses = 0
                    max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
                else:
                    consecutive_losses += 1
                    consecutive_wins = 0
                    max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
            
            self.performance_metrics = {
                'annual_return': annual_return,
                'annual_volatility': annual_volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'avg_profit': avg_profit,
                'avg_loss': avg_loss,
                'max_consecutive_wins': max_consecutive_wins,
                'max_consecutive_losses': max_consecutive_losses,
                'total_trades': len(self.trades),
                'total_profit': sum(trade_returns),
                'final_capital': self.equity_curve[-1]['equity']
            }
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {str(e)}")
    
    def _generate_backtest_report(self) -> Dict:
        """生成回测报告"""
        try:
            # 创建图表
            fig = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                subplot_titles=('Equity Curve', 'Drawdown', 'Monthly Returns')
            )
            
            # 添加权益曲线
            equity_dates = [e['date'] for e in self.equity_curve]
            equity_values = [e['equity'] for e in self.equity_curve]
            fig.add_trace(
                go.Scatter(
                    x=equity_dates,
                    y=equity_values,
                    name='Equity',
                    line=dict(color='blue')
                ),
                row=1, col=1
            )
            
            # 添加回撤曲线
            returns = pd.Series(equity_values).pct_change()
            cumulative_returns = (1 + returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdowns = cumulative_returns / rolling_max - 1
            fig.add_trace(
                go.Scatter(
                    x=equity_dates,
                    y=drawdowns,
                    name='Drawdown',
                    line=dict(color='red')
                ),
                row=2, col=1
            )
            
            # 添加月度收益热力图
            monthly_returns = returns.resample('M').agg(lambda x: (1 + x).prod() - 1)
            monthly_returns_matrix = monthly_returns.groupby(
                [monthly_returns.index.year, monthly_returns.index.month]
            ).mean().unstack()
            
            fig.add_trace(
                go.Heatmap(
                    z=monthly_returns_matrix.values,
                    x=monthly_returns_matrix.columns,
                    y=monthly_returns_matrix.index,
                    colorscale='RdYlGn',
                    name='Monthly Returns'
                ),
                row=3, col=1
            )
            
            # 更新布局
            fig.update_layout(
                height=1200,
                title_text="Backtest Results",
                showlegend=True
            )
            
            # 保存图表
            fig.write_html('backtest_results.html')
            
            return {
                'performance_metrics': self.performance_metrics,
                'trades': self.trades,
                'equity_curve': self.equity_curve,
                'parameters': self._get_strategy_parameters(),
                'generated_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating backtest report: {str(e)}")
            return {}
    
    def _get_strategy_parameters(self) -> Dict:
        """获取策略参数"""
        try:
            # 这里应该返回策略参数
            # 示例实现
            return {
                'initial_capital': self.initial_capital,
                'commission_rate': self.commission_rate,
                'slippage': self.slippage
            }
            
        except Exception as e:
            logger.error(f"Error getting strategy parameters: {str(e)}")
            return {}
    
    def _summarize_walk_forward_results(self, results: List[Dict]) -> Dict:
        """总结前向分析结果"""
        try:
            # 计算平均指标
            metrics = ['sharpe_ratio', 'annual_return', 'max_drawdown', 'win_rate']
            summary = {}
            
            for metric in metrics:
                train_values = [r['train_metrics'][metric] for r in results]
                test_values = [r['test_metrics'][metric] for r in results]
                
                summary[metric] = {
                    'train_mean': np.mean(train_values),
                    'train_std': np.std(train_values),
                    'test_mean': np.mean(test_values),
                    'test_std': np.std(test_values)
                }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error summarizing walk forward results: {str(e)}")
            return {} 