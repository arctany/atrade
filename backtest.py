import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from strategies import BaseStrategy
from utils import calculate_returns, calculate_volatility, calculate_sharpe_ratio, calculate_max_drawdown
import os
from config import BACKTEST_CONFIG, TRADING_STRATEGY, RISK_MANAGEMENT_CONFIG

logger = logging.getLogger(__name__)

class Backtest:
    """回测类，用于评估交易策略的历史表现"""
    
    def __init__(self, data: pd.DataFrame, initial_capital: float = None):
        """
        初始化回测类
        
        Args:
            data: 历史数据，包含OHLCV
            initial_capital: 初始资金
        """
        self.data = data.copy()
        self.initial_capital = initial_capital or BACKTEST_CONFIG['initial_capital']
        self.current_capital = self.initial_capital
        self.positions: Dict[str, Dict] = {}  # 当前持仓
        self.trades: List[Dict] = []  # 交易历史
        self.equity_curve: List[float] = [self.initial_capital]  # 权益曲线
        self.trade_history: List[Dict] = []  # 交易记录
        self.performance_metrics: Dict = {}  # 性能指标
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        
    def run(self, strategy: Dict) -> Dict:
        """
        运行回测
        
        Args:
            strategy: 策略配置
            
        Returns:
            Dict: 回测结果
        """
        try:
            self.logger.info("开始回测...")
            
            # 遍历数据
            for i in range(len(self.data)):
                current_data = self.data.iloc[:i+1]
                current_price = current_data['close'].iloc[-1]
                current_time = current_data.index[-1]
                
                # 生成交易信号
                signal = self._generate_signal(current_data, strategy)
                
                # 执行交易
                if signal:
                    self._execute_trade(signal, current_price, current_time)
                
                # 更新持仓
                self._update_positions(current_price, current_time)
                
                # 更新权益曲线
                self._update_equity_curve(current_price)
            
            # 计算性能指标
            self._calculate_performance_metrics()
            
            # 生成回测报告
            report = self._generate_report()
            
            self.logger.info("回测完成")
            return report
            
        except Exception as e:
            self.logger.error(f"回测过程中发生错误: {str(e)}")
            raise
    
    def _generate_signal(self, data: pd.DataFrame, strategy: Dict) -> Optional[Dict]:
        """
        生成交易信号
        
        Args:
            data: 历史数据
            strategy: 策略配置
            
        Returns:
            Optional[Dict]: 交易信号
        """
        try:
            signals = []
            weights = []
            
            # 遍历所有启用的策略
            for name, config in strategy['strategies'].items():
                if config['enabled']:
                    # 计算技术指标
                    indicators = self._calculate_indicators(data, name)
                    
                    # 生成信号
                    signal = self._generate_strategy_signal(name, indicators, config)
                    if signal:
                        signals.append(signal)
                        weights.append(config['weight'])
            
            if not signals:
                return None
            
            # 加权组合信号
            combined_signal = self._combine_signals(signals, weights)
            return combined_signal
            
        except Exception as e:
            self.logger.error(f"生成信号时发生错误: {str(e)}")
            return None
    
    def _calculate_indicators(self, data: pd.DataFrame, strategy_name: str) -> Dict:
        """
        计算技术指标
        
        Args:
            data: 历史数据
            strategy_name: 策略名称
            
        Returns:
            Dict: 技术指标
        """
        try:
            indicators = {}
            
            if strategy_name == 'MA_CROSS':
                # 计算移动平均线
                indicators['ma_short'] = data['close'].rolling(window=20).mean()
                indicators['ma_long'] = data['close'].rolling(window=50).mean()
                
            elif strategy_name == 'RSI':
                # 计算RSI
                delta = data['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                indicators['rsi'] = 100 - (100 / (1 + rs))
                
            elif strategy_name == 'MACD':
                # 计算MACD
                exp1 = data['close'].ewm(span=12, adjust=False).mean()
                exp2 = data['close'].ewm(span=26, adjust=False).mean()
                indicators['macd'] = exp1 - exp2
                indicators['signal'] = indicators['macd'].ewm(span=9, adjust=False).mean()
                
            elif strategy_name == 'BOLLINGER_BANDS':
                # 计算布林带
                indicators['ma'] = data['close'].rolling(window=20).mean()
                indicators['std'] = data['close'].rolling(window=20).std()
                indicators['upper'] = indicators['ma'] + 2 * indicators['std']
                indicators['lower'] = indicators['ma'] - 2 * indicators['std']
                
            return indicators
            
        except Exception as e:
            self.logger.error(f"计算技术指标时发生错误: {str(e)}")
            return {}
    
    def _generate_strategy_signal(self, strategy_name: str, indicators: Dict, config: Dict) -> Optional[Dict]:
        """
        根据策略生成交易信号
        
        Args:
            strategy_name: 策略名称
            indicators: 技术指标
            config: 策略配置
            
        Returns:
            Optional[Dict]: 交易信号
        """
        try:
            if strategy_name == 'MA_CROSS':
                # 均线交叉策略
                if indicators['ma_short'].iloc[-1] > indicators['ma_long'].iloc[-1] and \
                   indicators['ma_short'].iloc[-2] <= indicators['ma_long'].iloc[-2]:
                    return {'action': 'buy', 'strength': 1.0}
                elif indicators['ma_short'].iloc[-1] < indicators['ma_long'].iloc[-1] and \
                     indicators['ma_short'].iloc[-2] >= indicators['ma_long'].iloc[-2]:
                    return {'action': 'sell', 'strength': 1.0}
                    
            elif strategy_name == 'RSI':
                # RSI策略
                rsi = indicators['rsi'].iloc[-1]
                if rsi < 30:
                    return {'action': 'buy', 'strength': (70 - rsi) / 40}
                elif rsi > 70:
                    return {'action': 'sell', 'strength': (rsi - 30) / 40}
                    
            elif strategy_name == 'MACD':
                # MACD策略
                if indicators['macd'].iloc[-1] > indicators['signal'].iloc[-1] and \
                   indicators['macd'].iloc[-2] <= indicators['signal'].iloc[-2]:
                    return {'action': 'buy', 'strength': 1.0}
                elif indicators['macd'].iloc[-1] < indicators['signal'].iloc[-1] and \
                     indicators['macd'].iloc[-2] >= indicators['signal'].iloc[-2]:
                    return {'action': 'sell', 'strength': 1.0}
                    
            elif strategy_name == 'BOLLINGER_BANDS':
                # 布林带策略
                price = self.data['close'].iloc[-1]
                if price < indicators['lower'].iloc[-1]:
                    return {'action': 'buy', 'strength': 1.0}
                elif price > indicators['upper'].iloc[-1]:
                    return {'action': 'sell', 'strength': 1.0}
            
            return None
            
        except Exception as e:
            self.logger.error(f"生成策略信号时发生错误: {str(e)}")
            return None
    
    def _combine_signals(self, signals: List[Dict], weights: List[float]) -> Dict:
        """
        组合多个策略的信号
        
        Args:
            signals: 信号列表
            weights: 权重列表
            
        Returns:
            Dict: 组合后的信号
        """
        try:
            # 归一化权重
            weights = np.array(weights) / sum(weights)
            
            # 计算加权信号强度
            buy_strength = 0
            sell_strength = 0
            
            for signal, weight in zip(signals, weights):
                if signal['action'] == 'buy':
                    buy_strength += signal['strength'] * weight
                else:
                    sell_strength += signal['strength'] * weight
            
            # 确定最终信号
            if buy_strength > sell_strength and buy_strength > 0.5:
                return {'action': 'buy', 'strength': buy_strength}
            elif sell_strength > buy_strength and sell_strength > 0.5:
                return {'action': 'sell', 'strength': sell_strength}
            
            return None
            
        except Exception as e:
            self.logger.error(f"组合信号时发生错误: {str(e)}")
            return None
    
    def _execute_trade(self, signal: Dict, price: float, time: datetime) -> None:
        """
        执行交易
        
        Args:
            signal: 交易信号
            price: 当前价格
            time: 当前时间
        """
        try:
            # 计算交易数量
            position_size = TRADING_STRATEGY['position_size']
            trade_amount = self.current_capital * position_size * signal['strength']
            quantity = trade_amount / price
            
            # 计算交易成本
            commission = trade_amount * BACKTEST_CONFIG['commission_rate']
            slippage = trade_amount * BACKTEST_CONFIG['slippage']
            total_cost = commission + slippage
            
            # 检查是否有足够资金
            if trade_amount + total_cost > self.current_capital:
                self.logger.warning("资金不足，无法执行交易")
                return
            
            # 执行交易
            if signal['action'] == 'buy':
                # 检查是否达到最大持仓数量
                if len(self.positions) >= TRADING_STRATEGY['max_positions']:
                    self.logger.warning("达到最大持仓数量限制")
                    return
                
                # 开仓
                self.positions[time] = {
                    'type': 'long',
                    'quantity': quantity,
                    'entry_price': price,
                    'entry_time': time,
                    'stop_loss': price * (1 - TRADING_STRATEGY['stop_loss']),
                    'take_profit': price * (1 + TRADING_STRATEGY['take_profit'])
                }
                
                # 更新资金
                self.current_capital -= (trade_amount + total_cost)
                
                # 记录交易
                self.trades.append({
                    'time': time,
                    'action': 'buy',
                    'price': price,
                    'quantity': quantity,
                    'amount': trade_amount,
                    'cost': total_cost
                })
                
            else:  # sell
                # 平仓
                for pos_time, position in list(self.positions.items()):
                    if position['type'] == 'long':
                        # 计算收益
                        profit = (price - position['entry_price']) * position['quantity']
                        profit -= total_cost
                        
                        # 更新资金
                        self.current_capital += (trade_amount + profit)
                        
                        # 记录交易
                        self.trades.append({
                            'time': time,
                            'action': 'sell',
                            'price': price,
                            'quantity': position['quantity'],
                            'amount': trade_amount,
                            'cost': total_cost,
                            'profit': profit
                        })
                        
                        # 移除持仓
                        del self.positions[pos_time]
            
        except Exception as e:
            self.logger.error(f"执行交易时发生错误: {str(e)}")
    
    def _update_positions(self, price: float, time: datetime) -> None:
        """
        更新持仓状态
        
        Args:
            price: 当前价格
            time: 当前时间
        """
        try:
            for pos_time, position in list(self.positions.items()):
                # 检查止损
                if price <= position['stop_loss']:
                    self._close_position(position, price, time, 'stop_loss')
                    continue
                
                # 检查止盈
                if price >= position['take_profit']:
                    self._close_position(position, price, time, 'take_profit')
                    continue
                
        except Exception as e:
            self.logger.error(f"更新持仓状态时发生错误: {str(e)}")
    
    def _close_position(self, position: Dict, price: float, time: datetime, reason: str) -> None:
        """
        平仓
        
        Args:
            position: 持仓信息
            price: 当前价格
            time: 当前时间
            reason: 平仓原因
        """
        try:
            # 计算收益
            profit = (price - position['entry_price']) * position['quantity']
            trade_amount = position['quantity'] * price
            total_cost = trade_amount * (BACKTEST_CONFIG['commission_rate'] + BACKTEST_CONFIG['slippage'])
            profit -= total_cost
            
            # 更新资金
            self.current_capital += (trade_amount + profit)
            
            # 记录交易
            self.trades.append({
                'time': time,
                'action': 'sell',
                'price': price,
                'quantity': position['quantity'],
                'amount': trade_amount,
                'cost': total_cost,
                'profit': profit,
                'reason': reason
            })
            
            # 移除持仓
            del self.positions[time]
            
        except Exception as e:
            self.logger.error(f"平仓时发生错误: {str(e)}")
    
    def _update_equity_curve(self, price: float) -> None:
        """
        更新权益曲线
        
        Args:
            price: 当前价格
        """
        try:
            # 计算当前持仓市值
            position_value = sum(
                (price - pos['entry_price']) * pos['quantity']
                for pos in self.positions.values()
            )
            
            # 更新权益
            current_equity = self.current_capital + position_value
            self.equity_curve.append(current_equity)
            
        except Exception as e:
            self.logger.error(f"更新权益曲线时发生错误: {str(e)}")
    
    def _calculate_performance_metrics(self) -> None:
        """计算性能指标"""
        try:
            # 计算收益率
            returns = pd.Series(self.equity_curve).pct_change().dropna()
            
            # 总收益率
            total_return = (self.equity_curve[-1] - self.initial_capital) / self.initial_capital
            
            # 年化收益率
            trading_days = len(returns)
            annual_return = (1 + total_return) ** (252 / trading_days) - 1
            
            # 年化波动率
            annual_volatility = returns.std() * np.sqrt(252)
            
            # 夏普比率
            risk_free_rate = RISK_MANAGEMENT_CONFIG['risk_free_rate']
            excess_returns = returns - risk_free_rate/252
            sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
            
            # 最大回撤
            cumulative_returns = (1 + returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdowns = cumulative_returns / rolling_max - 1
            max_drawdown = drawdowns.min()
            
            # 胜率
            winning_trades = len([t for t in self.trades if t.get('profit', 0) > 0])
            total_trades = len([t for t in self.trades if t.get('profit') is not None])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # 盈亏比
            total_profit = sum(t.get('profit', 0) for t in self.trades if t.get('profit', 0) > 0)
            total_loss = abs(sum(t.get('profit', 0) for t in self.trades if t.get('profit', 0) < 0))
            profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
            
            # 保存性能指标
            self.performance_metrics = {
                'total_return': total_return,
                'annual_return': annual_return,
                'annual_volatility': annual_volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': total_trades - winning_trades,
                'total_profit': total_profit,
                'total_loss': total_loss
            }
            
        except Exception as e:
            self.logger.error(f"计算性能指标时发生错误: {str(e)}")
    
    def _generate_report(self) -> Dict:
        """
        生成回测报告
        
        Returns:
            Dict: 回测报告
        """
        try:
            # 准备报告数据
            report = {
                'summary': {
                    'initial_capital': self.initial_capital,
                    'final_capital': self.equity_curve[-1],
                    'total_return': self.performance_metrics['total_return'],
                    'annual_return': self.performance_metrics['annual_return'],
                    'sharpe_ratio': self.performance_metrics['sharpe_ratio'],
                    'max_drawdown': self.performance_metrics['max_drawdown'],
                    'win_rate': self.performance_metrics['win_rate'],
                    'profit_factor': self.performance_metrics['profit_factor']
                },
                'trades': self.trades,
                'equity_curve': self.equity_curve,
                'performance_metrics': self.performance_metrics
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"生成回测报告时发生错误: {str(e)}")
            return {}

    def plot_results(self):
        """绘制回测结果"""
        try:
            import matplotlib.pyplot as plt
            
            # 创建图形
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # 绘制权益曲线
            ax1.plot(self.equity_curve)
            ax1.set_title('Equity Curve')
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Portfolio Value')
            ax1.grid(True)
            
            # 绘制回撤
            equity_series = pd.Series(self.equity_curve)
            rolling_max = equity_series.expanding().max()
            drawdowns = (equity_series - rolling_max) / rolling_max
            
            ax2.fill_between(equity_series.index, drawdowns, 0, color='red', alpha=0.3)
            ax2.set_title('Drawdown')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Drawdown')
            ax2.grid(True)
            
            plt.tight_layout()
            return fig
        except Exception as e:
            logger.error(f"Error plotting results: {str(e)}")
            return None

    def get_trade_summary(self) -> pd.DataFrame:
        """获取交易摘要"""
        try:
            if not self.trades:
                return pd.DataFrame()
            
            trades_df = pd.DataFrame(self.trades)
            trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
            trades_df = trades_df.sort_values('timestamp')
            
            # 计算每笔交易的收益
            trades_df['profit'] = 0
            for i in range(1, len(trades_df)):
                if trades_df.iloc[i-1]['type'] == 'BUY' and trades_df.iloc[i]['type'] == 'SELL':
                    profit = (trades_df.iloc[i]['price'] - trades_df.iloc[i-1]['price']) * trades_df.iloc[i-1]['quantity']
                    trades_df.iloc[i, trades_df.columns.get_loc('profit')] = profit
            
            return trades_df
        except Exception as e:
            logger.error(f"Error getting trade summary: {str(e)}")
            return pd.DataFrame() 