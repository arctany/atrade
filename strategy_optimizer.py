import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from itertools import product
from datetime import datetime
from strategy import TradingStrategy
from config import TRADING_STRATEGY, BACKTEST_CONFIG

logger = logging.getLogger(__name__)

class StrategyOptimizer:
    def __init__(self, data: pd.DataFrame):
        """
        初始化策略优化器
        
        Args:
            data: 包含OHLCV数据的DataFrame
        """
        self.data = data.copy()
        self.strategy = TradingStrategy(data)
        
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
            results = []
            
            # 生成参数组合
            param_combinations = [dict(zip(param_grid.keys(), v)) 
                                for v in product(*param_grid.values())]
            
            # 对每个参数组合进行回测
            for params in param_combinations:
                # 更新策略参数
                self._update_strategy_params(strategy_name, params)
                
                # 执行回测
                backtest_results = self.strategy.backtest_strategy(
                    strategy_name,
                    initial_capital=BACKTEST_CONFIG['initial_capital'],
                    commission_rate=BACKTEST_CONFIG['commission_rate'],
                    slippage=BACKTEST_CONFIG['slippage']
                )
                
                # 记录结果
                if backtest_results:
                    results.append({
                        'parameters': params,
                        'metrics': backtest_results,
                        'score': backtest_results.get(metric, 0)
                    })
            
            # 按优化指标排序
            results.sort(key=lambda x: x['score'], reverse=True)
            
            return {
                'best_parameters': results[0]['parameters'] if results else None,
                'best_metrics': results[0]['metrics'] if results else None,
                'all_results': results
            }
            
        except Exception as e:
            logger.error(f"Error in parameter optimization: {str(e)}")
            return {}
    
    def analyze_performance(
        self,
        strategy_name: str,
        parameters: Optional[Dict] = None
    ) -> Dict:
        """
        分析策略性能
        
        Args:
            strategy_name: 策略名称
            parameters: 策略参数
            
        Returns:
            性能分析结果
        """
        try:
            # 更新策略参数
            if parameters:
                self._update_strategy_params(strategy_name, parameters)
            
            # 执行回测
            backtest_results = self.strategy.backtest_strategy(
                strategy_name,
                initial_capital=BACKTEST_CONFIG['initial_capital'],
                commission_rate=BACKTEST_CONFIG['commission_rate'],
                slippage=BACKTEST_CONFIG['slippage']
            )
            
            if not backtest_results:
                return {}
            
            # 计算额外性能指标
            performance_metrics = self._calculate_performance_metrics(backtest_results)
            
            return {
                'backtest_results': backtest_results,
                'performance_metrics': performance_metrics
            }
            
        except Exception as e:
            logger.error(f"Error in performance analysis: {str(e)}")
            return {}
    
    def generate_report(
        self,
        strategy_name: str,
        parameters: Optional[Dict] = None
    ) -> Dict:
        """
        生成策略报告
        
        Args:
            strategy_name: 策略名称
            parameters: 策略参数
            
        Returns:
            策略报告
        """
        try:
            # 分析性能
            analysis_results = self.analyze_performance(strategy_name, parameters)
            
            if not analysis_results:
                return {}
            
            backtest_results = analysis_results['backtest_results']
            performance_metrics = analysis_results['performance_metrics']
            
            # 生成报告
            report = {
                'strategy_name': strategy_name,
                'parameters': parameters or TRADING_STRATEGY['strategies'][strategy_name],
                'summary': {
                    'initial_capital': backtest_results['initial_capital'],
                    'final_capital': backtest_results['final_capital'],
                    'total_return': (backtest_results['final_capital'] - backtest_results['initial_capital']) 
                                  / backtest_results['initial_capital'],
                    'total_trades': backtest_results['total_trades'],
                    'win_rate': backtest_results['win_rate'],
                    'max_drawdown': backtest_results['max_drawdown'],
                    'sharpe_ratio': backtest_results['sharpe_ratio']
                },
                'performance_metrics': performance_metrics,
                'trades': backtest_results['trades'],
                'generated_at': datetime.now().isoformat()
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error in report generation: {str(e)}")
            return {}
    
    def _update_strategy_params(self, strategy_name: str, parameters: Dict):
        """更新策略参数"""
        try:
            strategy_config = TRADING_STRATEGY['strategies'][strategy_name]
            strategy_config.update(parameters)
        except Exception as e:
            logger.error(f"Error updating strategy parameters: {str(e)}")
    
    def _calculate_performance_metrics(self, backtest_results: Dict) -> Dict:
        """计算性能指标"""
        try:
            metrics = {}
            
            # 计算年化收益率
            if backtest_results['trades']:
                first_trade_time = backtest_results['trades'][0]['time']
                last_trade_time = backtest_results['trades'][-1]['time']
                days = (last_trade_time - first_trade_time).days
                if days > 0:
                    total_return = (backtest_results['final_capital'] - backtest_results['initial_capital']) 
                                 / backtest_results['initial_capital']
                    metrics['annual_return'] = (1 + total_return) ** (365 / days) - 1
            
            # 计算盈亏比
            if backtest_results['losing_trades'] > 0:
                metrics['profit_factor'] = (
                    sum(trade['profit'] for trade in backtest_results['trades'] 
                        if trade['type'] == 'SELL' and trade['profit'] > 0) /
                    abs(sum(trade['profit'] for trade in backtest_results['trades'] 
                          if trade['type'] == 'SELL' and trade['profit'] < 0))
                )
            
            # 计算平均盈亏
            if backtest_results['total_trades'] > 0:
                profits = [trade['profit'] for trade in backtest_results['trades'] 
                          if trade['type'] == 'SELL']
                metrics['average_profit'] = sum(profits) / len(profits)
                metrics['average_win'] = sum(p for p in profits if p > 0) / len([p for p in profits if p > 0]) if any(p > 0 for p in profits) else 0
                metrics['average_loss'] = sum(p for p in profits if p < 0) / len([p for p in profits if p < 0]) if any(p < 0 for p in profits) else 0
            
            # 计算最大连续盈亏
            if backtest_results['trades']:
                current_streak = 0
                max_win_streak = 0
                max_loss_streak = 0
                
                for trade in backtest_results['trades']:
                    if trade['type'] == 'SELL':
                        if trade['profit'] > 0:
                            current_streak = max(0, current_streak + 1)
                            max_win_streak = max(max_win_streak, current_streak)
                        else:
                            current_streak = min(0, current_streak - 1)
                            max_loss_streak = min(max_loss_streak, current_streak)
                
                metrics['max_win_streak'] = max_win_streak
                metrics['max_loss_streak'] = abs(max_loss_streak)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {str(e)}")
            return {} 