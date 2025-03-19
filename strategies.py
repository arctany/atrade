import pandas as pd
import numpy as np
import talib
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging
from config import TECHNICAL_INDICATORS, TRADING_STRATEGY

logger = logging.getLogger(__name__)

class BaseStrategy:
    """策略基类"""
    def __init__(self, name: str):
        self.name = name
        self.positions: Dict[str, float] = {}
        self.trades: List[Dict] = []
        self.initial_capital = TRADING_STRATEGY['initial_capital']
        self.current_capital = self.initial_capital
        self.position_size = TRADING_STRATEGY['position_size']
        self.stop_loss = TRADING_STRATEGY['stop_loss']
        self.take_profit = TRADING_STRATEGY['take_profit']
        self.max_positions = TRADING_STRATEGY['max_positions']

    def calculate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算交易信号"""
        raise NotImplementedError

    def execute_trade(self, symbol: str, signal: int, price: float) -> Optional[Dict]:
        """执行交易"""
        try:
            if signal == 0:
                return None

            position = self.positions.get(symbol, 0)
            quantity = int(self.current_capital * self.position_size / price)

            if signal > 0 and position <= 0:  # 买入信号
                if len(self.positions) >= self.max_positions:
                    return None
                
                if position < 0:  # 平空仓
                    trade = {
                        'symbol': symbol,
                        'type': 'BUY',
                        'quantity': abs(position),
                        'price': price,
                        'timestamp': datetime.now()
                    }
                    self.trades.append(trade)
                    self.positions[symbol] = 0
                    return trade

                # 开多仓
                trade = {
                    'symbol': symbol,
                    'type': 'BUY',
                    'quantity': quantity,
                    'price': price,
                    'timestamp': datetime.now()
                }
                self.trades.append(trade)
                self.positions[symbol] = quantity
                return trade

            elif signal < 0 and position >= 0:  # 卖出信号
                if position > 0:  # 平多仓
                    trade = {
                        'symbol': symbol,
                        'type': 'SELL',
                        'quantity': position,
                        'price': price,
                        'timestamp': datetime.now()
                    }
                    self.trades.append(trade)
                    self.positions[symbol] = 0
                    return trade

                # 开空仓
                trade = {
                    'symbol': symbol,
                    'type': 'SELL',
                    'quantity': quantity,
                    'price': price,
                    'timestamp': datetime.now()
                }
                self.trades.append(trade)
                self.positions[symbol] = -quantity
                return trade

            return None
        except Exception as e:
            logger.error(f"Error executing trade: {str(e)}")
            return None

    def check_stop_loss_take_profit(self, symbol: str, current_price: float) -> Optional[Dict]:
        """检查止损止盈"""
        try:
            position = self.positions.get(symbol)
            if position is None or position == 0:
                return None

            entry_price = self.trades[-1]['price'] if self.trades else 0
            if entry_price == 0:
                return None

            price_change = (current_price - entry_price) / entry_price

            if position > 0:  # 多仓
                if price_change <= -self.stop_loss:
                    return self.execute_trade(symbol, -1, current_price)
                elif price_change >= self.take_profit:
                    return self.execute_trade(symbol, -1, current_price)
            else:  # 空仓
                if price_change >= self.stop_loss:
                    return self.execute_trade(symbol, 1, current_price)
                elif price_change <= -self.take_profit:
                    return self.execute_trade(symbol, 1, current_price)

            return None
        except Exception as e:
            logger.error(f"Error checking stop loss/take profit: {str(e)}")
            return None

class MovingAverageCrossStrategy(BaseStrategy):
    """移动平均线交叉策略"""
    def __init__(self):
        super().__init__("Moving Average Cross")
        self.short_window = TECHNICAL_INDICATORS['MA']['short']
        self.long_window = TECHNICAL_INDICATORS['MA']['long']

    def calculate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算移动平均线交叉信号"""
        try:
            df = data.copy()
            
            # 计算移动平均线
            df['MA_short'] = talib.MA(df['Close'], timeperiod=self.short_window)
            df['MA_long'] = talib.MA(df['Close'], timeperiod=self.long_window)
            
            # 计算信号
            df['Signal'] = 0
            df.loc[df['MA_short'] > df['MA_long'], 'Signal'] = 1
            df.loc[df['MA_short'] < df['MA_long'], 'Signal'] = -1
            
            return df
        except Exception as e:
            logger.error(f"Error calculating MA cross signals: {str(e)}")
            return data

class RSIMomentumStrategy(BaseStrategy):
    """RSI动量策略"""
    def __init__(self):
        super().__init__("RSI Momentum")
        self.rsi_period = TECHNICAL_INDICATORS['RSI']['period']
        self.overbought = TECHNICAL_INDICATORS['RSI']['overbought']
        self.oversold = TECHNICAL_INDICATORS['RSI']['oversold']

    def calculate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算RSI信号"""
        try:
            df = data.copy()
            
            # 计算RSI
            df['RSI'] = talib.RSI(df['Close'], timeperiod=self.rsi_period)
            
            # 计算信号
            df['Signal'] = 0
            df.loc[df['RSI'] < self.oversold, 'Signal'] = 1
            df.loc[df['RSI'] > self.overbought, 'Signal'] = -1
            
            return df
        except Exception as e:
            logger.error(f"Error calculating RSI signals: {str(e)}")
            return data

class MACDStrategy(BaseStrategy):
    """MACD策略"""
    def __init__(self):
        super().__init__("MACD")
        self.fast_period = TECHNICAL_INDICATORS['MACD']['fast']
        self.slow_period = TECHNICAL_INDICATORS['MACD']['slow']
        self.signal_period = TECHNICAL_INDICATORS['MACD']['signal']

    def calculate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算MACD信号"""
        try:
            df = data.copy()
            
            # 计算MACD
            macd, signal, hist = talib.MACD(
                df['Close'],
                fastperiod=self.fast_period,
                slowperiod=self.slow_period,
                signalperiod=self.signal_period
            )
            
            df['MACD'] = macd
            df['Signal'] = signal
            df['MACD_Hist'] = hist
            
            # 计算交易信号
            df['Trade_Signal'] = 0
            df.loc[df['MACD'] > df['Signal'], 'Trade_Signal'] = 1
            df.loc[df['MACD'] < df['Signal'], 'Trade_Signal'] = -1
            
            return df
        except Exception as e:
            logger.error(f"Error calculating MACD signals: {str(e)}")
            return data

class BollingerBandsStrategy(BaseStrategy):
    """布林带策略"""
    def __init__(self):
        super().__init__("Bollinger Bands")
        self.bb_period = TECHNICAL_INDICATORS['BB']['period']
        self.bb_std = TECHNICAL_INDICATORS['BB']['std_dev']

    def calculate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算布林带信号"""
        try:
            df = data.copy()
            
            # 计算布林带
            upper, middle, lower = talib.BBANDS(
                df['Close'],
                timeperiod=self.bb_period,
                nbdevup=self.bb_std,
                nbdevdn=self.bb_std
            )
            
            df['BB_Upper'] = upper
            df['BB_Middle'] = middle
            df['BB_Lower'] = lower
            
            # 计算信号
            df['Signal'] = 0
            df.loc[df['Close'] < df['BB_Lower'], 'Signal'] = 1
            df.loc[df['Close'] > df['BB_Upper'], 'Signal'] = -1
            
            return df
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands signals: {str(e)}")
            return data

class CombinedStrategy(BaseStrategy):
    """组合策略"""
    def __init__(self):
        super().__init__("Combined")
        self.strategies = [
            MovingAverageCrossStrategy(),
            RSIMomentumStrategy(),
            MACDStrategy(),
            BollingerBandsStrategy()
        ]

    def calculate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算组合信号"""
        try:
            df = data.copy()
            signals = pd.DataFrame(index=df.index)
            
            # 获取各个策略的信号
            for strategy in self.strategies:
                strategy_data = strategy.calculate_signals(df)
                signals[strategy.name] = strategy_data['Signal']
            
            # 计算综合信号
            signals['Combined_Signal'] = signals.mean(axis=1)
            
            # 转换为交易信号
            df['Signal'] = 0
            df.loc[signals['Combined_Signal'] > 0.5, 'Signal'] = 1
            df.loc[signals['Combined_Signal'] < -0.5, 'Signal'] = -1
            
            return df
        except Exception as e:
            logger.error(f"Error calculating combined signals: {str(e)}")
            return data 