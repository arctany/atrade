import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
from indicators import TechnicalIndicators
from config import TECHNICAL_INDICATORS, TRADING_STRATEGY

logger = logging.getLogger(__name__)

class TradingStrategy:
    def __init__(self, data: pd.DataFrame):
        """
        初始化交易策略
        
        Args:
            data: 包含OHLCV数据的DataFrame
        """
        self.data = data.copy()
        self.indicators = TechnicalIndicators(data)
        self._validate_data()
        
    def _validate_data(self):
        """验证输入数据的完整性"""
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

    def ma_cross_strategy(self) -> List[Dict]:
        """
        移动平均线交叉策略
        
        Returns:
            交易信号列表
        """
        try:
            signals = []
            ma_config = TECHNICAL_INDICATORS['MA']
            
            # 计算移动平均线
            sma_short = self.indicators.calculate_trend_indicators()[f'SMA_{ma_config["short"]}']
            sma_long = self.indicators.calculate_trend_indicators()[f'SMA_{ma_config["long"]}']
            
            # 生成交易信号
            for i in range(1, len(self.data)):
                if sma_short[i] > sma_long[i] and sma_short[i-1] <= sma_long[i-1]:
                    signals.append({
                        'type': 'BUY',
                        'price': self.data['Close'].iloc[i],
                        'time': self.data.index[i],
                        'reason': 'Short MA crossed above Long MA'
                    })
                elif sma_short[i] < sma_long[i] and sma_short[i-1] >= sma_long[i-1]:
                    signals.append({
                        'type': 'SELL',
                        'price': self.data['Close'].iloc[i],
                        'time': self.data.index[i],
                        'reason': 'Short MA crossed below Long MA'
                    })
            
            return signals
        except Exception as e:
            logger.error(f"Error in MA cross strategy: {str(e)}")
            return []

    def rsi_strategy(self) -> List[Dict]:
        """
        RSI策略
        
        Returns:
            交易信号列表
        """
        try:
            signals = []
            rsi_config = TECHNICAL_INDICATORS['RSI']
            
            # 计算RSI
            rsi = self.indicators.calculate_momentum_indicators()['RSI_14']
            
            # 生成交易信号
            for i in range(1, len(self.data)):
                if rsi[i] < rsi_config['oversold'] and rsi[i-1] >= rsi_config['oversold']:
                    signals.append({
                        'type': 'BUY',
                        'price': self.data['Close'].iloc[i],
                        'time': self.data.index[i],
                        'reason': 'RSI oversold'
                    })
                elif rsi[i] > rsi_config['overbought'] and rsi[i-1] <= rsi_config['overbought']:
                    signals.append({
                        'type': 'SELL',
                        'price': self.data['Close'].iloc[i],
                        'time': self.data.index[i],
                        'reason': 'RSI overbought'
                    })
            
            return signals
        except Exception as e:
            logger.error(f"Error in RSI strategy: {str(e)}")
            return []

    def macd_strategy(self) -> List[Dict]:
        """
        MACD策略
        
        Returns:
            交易信号列表
        """
        try:
            signals = []
            macd_config = TECHNICAL_INDICATORS['MACD']
            
            # 计算MACD
            macd = self.indicators.calculate_trend_indicators()
            macd_line = macd['MACD']
            signal_line = macd['MACD_Signal']
            hist = macd['MACD_Hist']
            
            # 生成交易信号
            for i in range(1, len(self.data)):
                if macd_line[i] > signal_line[i] and macd_line[i-1] <= signal_line[i-1]:
                    signals.append({
                        'type': 'BUY',
                        'price': self.data['Close'].iloc[i],
                        'time': self.data.index[i],
                        'reason': 'MACD crossed above Signal line'
                    })
                elif macd_line[i] < signal_line[i] and macd_line[i-1] >= signal_line[i-1]:
                    signals.append({
                        'type': 'SELL',
                        'price': self.data['Close'].iloc[i],
                        'time': self.data.index[i],
                        'reason': 'MACD crossed below Signal line'
                    })
            
            return signals
        except Exception as e:
            logger.error(f"Error in MACD strategy: {str(e)}")
            return []

    def bollinger_bands_strategy(self) -> List[Dict]:
        """
        布林带策略
        
        Returns:
            交易信号列表
        """
        try:
            signals = []
            bb_config = TECHNICAL_INDICATORS['BB']
            
            # 计算布林带
            bb = self.indicators.calculate_volatility_indicators()
            upper = bb['BB_Upper']
            lower = bb['BB_Lower']
            middle = bb['BB_Middle']
            
            # 生成交易信号
            for i in range(1, len(self.data)):
                if self.data['Close'].iloc[i] < lower[i]:
                    signals.append({
                        'type': 'BUY',
                        'price': self.data['Close'].iloc[i],
                        'time': self.data.index[i],
                        'reason': 'Price below lower Bollinger Band'
                    })
                elif self.data['Close'].iloc[i] > upper[i]:
                    signals.append({
                        'type': 'SELL',
                        'price': self.data['Close'].iloc[i],
                        'time': self.data.index[i],
                        'reason': 'Price above upper Bollinger Band'
                    })
            
            return signals
        except Exception as e:
            logger.error(f"Error in Bollinger Bands strategy: {str(e)}")
            return []

    def stochastic_strategy(self) -> List[Dict]:
        """
        随机指标策略
        
        Returns:
            交易信号列表
        """
        try:
            signals = []
            stoch_config = TECHNICAL_INDICATORS['Stochastic']
            
            # 计算随机指标
            stoch = self.indicators.calculate_momentum_indicators()
            k_line = stoch['Stoch_K']
            d_line = stoch['Stoch_D']
            
            # 生成交易信号
            for i in range(1, len(self.data)):
                if k_line[i] < 20 and d_line[i] < 20:
                    signals.append({
                        'type': 'BUY',
                        'price': self.data['Close'].iloc[i],
                        'time': self.data.index[i],
                        'reason': 'Stochastic oversold'
                    })
                elif k_line[i] > 80 and d_line[i] > 80:
                    signals.append({
                        'type': 'SELL',
                        'price': self.data['Close'].iloc[i],
                        'time': self.data.index[i],
                        'reason': 'Stochastic overbought'
                    })
            
            return signals
        except Exception as e:
            logger.error(f"Error in Stochastic strategy: {str(e)}")
            return []

    def trend_following_strategy(self) -> List[Dict]:
        """
        趋势跟踪策略
        
        Returns:
            交易信号列表
        """
        try:
            signals = []
            ma_config = TECHNICAL_INDICATORS['MA']
            
            # 计算多个周期的移动平均线
            sma_short = self.indicators.calculate_trend_indicators()[f'SMA_{ma_config["short"]}']
            sma_long = self.indicators.calculate_trend_indicators()[f'SMA_{ma_config["long"]}']
            sma_longest = self.indicators.calculate_trend_indicators()[f'SMA_{ma_config["longest"]}']
            
            # 计算趋势强度
            atr = self.indicators.calculate_volatility_indicators()['ATR']
            atr_percent = atr / self.data['Close'] * 100
            
            # 生成交易信号
            for i in range(1, len(self.data)):
                # 判断趋势方向
                trend_up = sma_short[i] > sma_long[i] > sma_longest[i]
                trend_down = sma_short[i] < sma_long[i] < sma_longest[i]
                
                # 判断趋势强度
                strong_trend = atr_percent[i] > atr_percent[i-1]
                
                if trend_up and strong_trend:
                    signals.append({
                        'type': 'BUY',
                        'price': self.data['Close'].iloc[i],
                        'time': self.data.index[i],
                        'reason': 'Strong uptrend detected'
                    })
                elif trend_down and strong_trend:
                    signals.append({
                        'type': 'SELL',
                        'price': self.data['Close'].iloc[i],
                        'time': self.data.index[i],
                        'reason': 'Strong downtrend detected'
                    })
            
            return signals
        except Exception as e:
            logger.error(f"Error in trend following strategy: {str(e)}")
            return []

    def momentum_strategy(self) -> List[Dict]:
        """
        动量交易策略
        
        Returns:
            交易信号列表
        """
        try:
            signals = []
            
            # 计算动量指标
            rsi = self.indicators.calculate_momentum_indicators()['RSI_14']
            stoch = self.indicators.calculate_momentum_indicators()
            k_line = stoch['Stoch_K']
            d_line = stoch['Stoch_D']
            
            # 计算价格动量
            returns = self.data['Close'].pct_change()
            momentum = returns.rolling(window=10).mean()
            
            # 生成交易信号
            for i in range(1, len(self.data)):
                # 判断动量方向
                momentum_up = momentum[i] > 0 and momentum[i] > momentum[i-1]
                momentum_down = momentum[i] < 0 and momentum[i] < momentum[i-1]
                
                # 判断RSI和随机指标
                rsi_oversold = rsi[i] < 30
                rsi_overbought = rsi[i] > 70
                stoch_oversold = k_line[i] < 20 and d_line[i] < 20
                stoch_overbought = k_line[i] > 80 and d_line[i] > 80
                
                if momentum_up and (rsi_oversold or stoch_oversold):
                    signals.append({
                        'type': 'BUY',
                        'price': self.data['Close'].iloc[i],
                        'time': self.data.index[i],
                        'reason': 'Positive momentum with oversold conditions'
                    })
                elif momentum_down and (rsi_overbought or stoch_overbought):
                    signals.append({
                        'type': 'SELL',
                        'price': self.data['Close'].iloc[i],
                        'time': self.data.index[i],
                        'reason': 'Negative momentum with overbought conditions'
                    })
            
            return signals
        except Exception as e:
            logger.error(f"Error in momentum strategy: {str(e)}")
            return []

    def volatility_breakout_strategy(self) -> List[Dict]:
        """
        波动率突破策略
        
        Returns:
            交易信号列表
        """
        try:
            signals = []
            
            # 计算波动率指标
            bb = self.indicators.calculate_volatility_indicators()
            upper = bb['BB_Upper']
            lower = bb['BB_Lower']
            middle = bb['BB_Middle']
            atr = bb['ATR']
            
            # 计算波动率变化
            atr_change = atr.pct_change()
            
            # 生成交易信号
            for i in range(1, len(self.data)):
                # 判断波动率突破
                volatility_increase = atr_change[i] > 0.1  # 波动率增加超过10%
                price_above_upper = self.data['Close'].iloc[i] > upper[i]
                price_below_lower = self.data['Close'].iloc[i] < lower[i]
                
                if volatility_increase and price_above_upper:
                    signals.append({
                        'type': 'BUY',
                        'price': self.data['Close'].iloc[i],
                        'time': self.data.index[i],
                        'reason': 'Volatility breakout above upper band'
                    })
                elif volatility_increase and price_below_lower:
                    signals.append({
                        'type': 'SELL',
                        'price': self.data['Close'].iloc[i],
                        'time': self.data.index[i],
                        'reason': 'Volatility breakout below lower band'
                    })
            
            return signals
        except Exception as e:
            logger.error(f"Error in volatility breakout strategy: {str(e)}")
            return []

    def volume_price_strategy(self) -> List[Dict]:
        """
        成交量价格策略
        
        Returns:
            交易信号列表
        """
        try:
            signals = []
            
            # 计算成交量指标
            volume_ma = self.data['Volume'].rolling(window=20).mean()
            volume_std = self.data['Volume'].rolling(window=20).std()
            
            # 计算价格趋势
            price_ma = self.data['Close'].rolling(window=20).mean()
            price_std = self.data['Close'].rolling(window=20).std()
            
            # 生成交易信号
            for i in range(1, len(self.data)):
                # 判断成交量异常
                volume_spike = self.data['Volume'].iloc[i] > volume_ma[i] + 2 * volume_std[i]
                volume_drop = self.data['Volume'].iloc[i] < volume_ma[i] - 2 * volume_std[i]
                
                # 判断价格趋势
                price_up = self.data['Close'].iloc[i] > price_ma[i]
                price_down = self.data['Close'].iloc[i] < price_ma[i]
                
                if volume_spike and price_up:
                    signals.append({
                        'type': 'BUY',
                        'price': self.data['Close'].iloc[i],
                        'time': self.data.index[i],
                        'reason': 'Volume spike with upward price trend'
                    })
                elif volume_spike and price_down:
                    signals.append({
                        'type': 'SELL',
                        'price': self.data['Close'].iloc[i],
                        'time': self.data.index[i],
                        'reason': 'Volume spike with downward price trend'
                    })
            
            return signals
        except Exception as e:
            logger.error(f"Error in volume price strategy: {str(e)}")
            return []

    def support_resistance_strategy(self) -> List[Dict]:
        """
        支撑阻力位策略
        
        Returns:
            交易信号列表
        """
        try:
            signals = []
            
            # 计算支撑阻力位
            support_resistance = self.indicators.calculate_support_resistance()
            support_levels = support_resistance['support_levels']
            resistance_levels = support_resistance['resistance_levels']
            
            # 计算价格与支撑阻力位的距离
            current_price = self.data['Close']
            
            # 生成交易信号
            for i in range(1, len(self.data)):
                # 判断是否接近支撑位
                near_support = any(abs(current_price[i] - level) / level < 0.01 
                                 for level in support_levels)
                
                # 判断是否接近阻力位
                near_resistance = any(abs(current_price[i] - level) / level < 0.01 
                                    for level in resistance_levels)
                
                if near_support:
                    signals.append({
                        'type': 'BUY',
                        'price': current_price[i],
                        'time': self.data.index[i],
                        'reason': 'Price near support level'
                    })
                elif near_resistance:
                    signals.append({
                        'type': 'SELL',
                        'price': current_price[i],
                        'time': self.data.index[i],
                        'reason': 'Price near resistance level'
                    })
            
            return signals
        except Exception as e:
            logger.error(f"Error in support resistance strategy: {str(e)}")
            return []

    def pattern_recognition_strategy(self) -> List[Dict]:
        """
        价格形态识别策略
        
        Returns:
            交易信号列表
        """
        try:
            signals = []
            
            # 获取价格形态
            patterns = self.indicators.calculate_price_patterns()
            
            # 生成交易信号
            for i in range(1, len(self.data)):
                # 判断看涨形态
                bullish_patterns = [
                    pattern for pattern in patterns
                    if pattern['type'] == 'bullish' and pattern['time'] == self.data.index[i]
                ]
                
                # 判断看跌形态
                bearish_patterns = [
                    pattern for pattern in patterns
                    if pattern['type'] == 'bearish' and pattern['time'] == self.data.index[i]
                ]
                
                if bullish_patterns:
                    signals.append({
                        'type': 'BUY',
                        'price': self.data['Close'].iloc[i],
                        'time': self.data.index[i],
                        'reason': f"Bullish pattern detected: {bullish_patterns[0]['name']}"
                    })
                elif bearish_patterns:
                    signals.append({
                        'type': 'SELL',
                        'price': self.data['Close'].iloc[i],
                        'time': self.data.index[i],
                        'reason': f"Bearish pattern detected: {bearish_patterns[0]['name']}"
                    })
            
            return signals
        except Exception as e:
            logger.error(f"Error in pattern recognition strategy: {str(e)}")
            return []

    def enhanced_combined_strategy(self) -> List[Dict]:
        """
        增强版组合策略（综合所有技术指标和策略）
        
        Returns:
            交易信号列表
        """
        try:
            signals = []
            
            # 获取所有策略的信号
            ma_signals = self.ma_cross_strategy()
            rsi_signals = self.rsi_strategy()
            macd_signals = self.macd_strategy()
            bb_signals = self.bollinger_bands_strategy()
            stoch_signals = self.stochastic_strategy()
            trend_signals = self.trend_following_strategy()
            momentum_signals = self.momentum_strategy()
            volatility_signals = self.volatility_breakout_strategy()
            volume_signals = self.volume_price_strategy()
            support_resistance_signals = self.support_resistance_strategy()
            pattern_signals = self.pattern_recognition_strategy()
            
            # 合并所有信号
            all_signals = (
                ma_signals + rsi_signals + macd_signals + bb_signals + stoch_signals +
                trend_signals + momentum_signals + volatility_signals + volume_signals +
                support_resistance_signals + pattern_signals
            )
            
            # 按时间排序
            all_signals.sort(key=lambda x: x['time'])
            
            # 合并相同时间的信号
            merged_signals = []
            current_time = None
            current_signal = None
            
            for signal in all_signals:
                if current_time != signal['time']:
                    if current_signal is not None:
                        merged_signals.append(current_signal)
                    current_time = signal['time']
                    current_signal = signal
                else:
                    # 合并信号
                    if current_signal['type'] != signal['type']:
                        # 如果信号类型不同，取消该时间点的信号
                        current_signal = None
                    else:
                        # 如果信号类型相同，合并原因
                        current_signal['reason'] += f" and {signal['reason']}"
            
            if current_signal is not None:
                merged_signals.append(current_signal)
            
            return merged_signals
        except Exception as e:
            logger.error(f"Error in enhanced combined strategy: {str(e)}")
            return []

    def backtest_strategy(
        self,
        strategy_name: str,
        initial_capital: float = None,
        commission_rate: float = None,
        slippage: float = None
    ) -> Dict:
        """
        回测策略
        
        Args:
            strategy_name: 策略名称
            initial_capital: 初始资金
            commission_rate: 手续费率
            slippage: 滑点
            
        Returns:
            回测结果
        """
        try:
            # 获取策略参数
            initial_capital = initial_capital or TRADING_STRATEGY['initial_capital']
            commission_rate = commission_rate or TRADING_STRATEGY['commission_rate']
            slippage = slippage or TRADING_STRATEGY['slippage']
            
            # 获取策略信号
            if strategy_name == 'MA_CROSS':
                signals = self.ma_cross_strategy()
            elif strategy_name == 'RSI':
                signals = self.rsi_strategy()
            elif strategy_name == 'MACD':
                signals = self.macd_strategy()
            elif strategy_name == 'BOLLINGER_BANDS':
                signals = self.bollinger_bands_strategy()
            elif strategy_name == 'STOCHASTIC':
                signals = self.stochastic_strategy()
            elif strategy_name == 'COMBINED':
                signals = self.enhanced_combined_strategy()
            else:
                raise ValueError(f"Unknown strategy: {strategy_name}")
            
            # 初始化回测结果
            results = {
                'initial_capital': initial_capital,
                'final_capital': initial_capital,
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'trades': []
            }
            
            # 执行回测
            position = 0
            entry_price = 0
            capital = initial_capital
            
            for signal in signals:
                if signal['type'] == 'BUY' and position == 0:
                    # 开仓
                    shares = capital / signal['price']
                    cost = shares * signal['price'] * (1 + commission_rate + slippage)
                    if cost <= capital:
                        position = shares
                        entry_price = signal['price']
                        capital -= cost
                        results['trades'].append({
                            'type': 'BUY',
                            'price': signal['price'],
                            'shares': shares,
                            'cost': cost,
                            'time': signal['time']
                        })
                
                elif signal['type'] == 'SELL' and position > 0:
                    # 平仓
                    revenue = position * signal['price'] * (1 - commission_rate - slippage)
                    capital += revenue
                    profit = revenue - (position * entry_price)
                    results['trades'].append({
                        'type': 'SELL',
                        'price': signal['price'],
                        'shares': position,
                        'revenue': revenue,
                        'profit': profit,
                        'time': signal['time']
                    })
                    
                    if profit > 0:
                        results['winning_trades'] += 1
                    else:
                        results['losing_trades'] += 1
                    
                    position = 0
                    entry_price = 0
            
            # 计算回测指标
            results['total_trades'] = len(results['trades']) // 2
            results['win_rate'] = results['winning_trades'] / results['total_trades'] if results['total_trades'] > 0 else 0
            
            # 计算最大回撤
            equity_curve = [initial_capital]
            for trade in results['trades']:
                if trade['type'] == 'SELL':
                    equity_curve.append(equity_curve[-1] + trade['profit'])
            
            max_drawdown = 0
            peak = equity_curve[0]
            for value in equity_curve:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak
                max_drawdown = max(max_drawdown, drawdown)
            
            results['max_drawdown'] = max_drawdown
            
            # 计算夏普比率
            returns = np.diff(equity_curve) / equity_curve[:-1]
            if len(returns) > 0:
                results['sharpe_ratio'] = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
            
            # 计算最终资金
            results['final_capital'] = capital + (position * self.data['Close'].iloc[-1] if position > 0 else 0)
            
            return results
        except Exception as e:
            logger.error(f"Error in backtest: {str(e)}")
            return {} 