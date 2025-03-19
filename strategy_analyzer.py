import pandas as pd
import numpy as np
from scipy import stats
import talib
from datetime import datetime, timedelta

class StrategyAnalyzer:
    def __init__(self):
        self.performance_metrics = {}
        self.trade_history = None
        self.positions = None

    def set_data(self, trade_history, positions):
        """设置交易历史和持仓数据"""
        self.trade_history = trade_history
        self.positions = positions

    def calculate_basic_metrics(self):
        """计算基本性能指标"""
        if self.trade_history is None:
            return None

        df = self.trade_history.copy()
        
        # 计算总收益
        total_return = df['realizedPnl'].sum()
        
        # 计算胜率
        winning_trades = len(df[df['realizedPnl'] > 0])
        total_trades = len(df)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # 计算平均收益
        avg_return = df['realizedPnl'].mean()
        
        # 计算最大回撤
        cumulative_returns = df['realizedPnl'].cumsum()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = cumulative_returns - rolling_max
        max_drawdown = drawdowns.min()
        
        # 计算夏普比率
        returns = df['realizedPnl'].pct_change().dropna()
        sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std() if len(returns) > 0 else 0
        
        self.performance_metrics = {
            'total_return': total_return,
            'win_rate': win_rate,
            'avg_return': avg_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'total_trades': total_trades
        }
        
        return self.performance_metrics

    def analyze_technical_indicators(self, historical_data):
        """分析技术指标"""
        if historical_data is None:
            return None

        df = historical_data.copy()
        
        # 计算移动平均线
        df['MA20'] = talib.MA(df['close'], timeperiod=20)
        df['MA50'] = talib.MA(df['close'], timeperiod=50)
        
        # 计算RSI
        df['RSI'] = talib.RSI(df['close'], timeperiod=14)
        
        # 计算MACD
        macd, macd_signal, macd_hist = talib.MACD(df['close'])
        df['MACD'] = macd
        df['MACD_Signal'] = macd_signal
        df['MACD_Hist'] = macd_hist
        
        # 计算布林带
        upper, middle, lower = talib.BBANDS(df['close'], timeperiod=20)
        df['BB_Upper'] = upper
        df['BB_Middle'] = middle
        df['BB_Lower'] = lower
        
        return df

    def generate_signals(self, technical_data):
        """生成交易信号"""
        if technical_data is None:
            return None

        df = technical_data.copy()
        signals = pd.DataFrame(index=df.index)
        
        # 移动平均线交叉信号
        signals['MA_Cross'] = np.where(df['MA20'] > df['MA50'], 1, -1)
        
        # RSI信号
        signals['RSI_Signal'] = np.where(df['RSI'] > 70, -1,
                                       np.where(df['RSI'] < 30, 1, 0))
        
        # MACD信号
        signals['MACD_Signal'] = np.where(df['MACD'] > df['MACD_Signal'], 1, -1)
        
        # 布林带信号
        signals['BB_Signal'] = np.where(df['close'] > df['BB_Upper'], -1,
                                      np.where(df['close'] < df['BB_Lower'], 1, 0))
        
        # 综合信号
        signals['Combined_Signal'] = (signals['MA_Cross'] + signals['RSI_Signal'] + 
                                    signals['MACD_Signal'] + signals['BB_Signal']) / 4
        
        return signals

    def backtest_strategy(self, historical_data, signals, initial_capital=100000):
        """回测策略"""
        if historical_data is None or signals is None:
            return None

        df = historical_data.copy()
        signals_df = signals.copy()
        
        # 初始化回测结果
        portfolio_value = initial_capital
        position = 0
        trades = []
        
        for i in range(1, len(df)):
            signal = signals_df['Combined_Signal'].iloc[i]
            
            # 根据信号调整仓位
            if signal > 0.5 and position <= 0:  # 买入信号
                shares = portfolio_value / df['close'].iloc[i]
                position = shares
                trades.append({
                    'date': df.index[i],
                    'type': 'BUY',
                    'price': df['close'].iloc[i],
                    'shares': shares
                })
            elif signal < -0.5 and position >= 0:  # 卖出信号
                if position > 0:
                    trades.append({
                        'date': df.index[i],
                        'type': 'SELL',
                        'price': df['close'].iloc[i],
                        'shares': position
                    })
                position = -shares
            
            # 更新组合价值
            portfolio_value = position * df['close'].iloc[i]
        
        return pd.DataFrame(trades)

    def analyze_risk_metrics(self, returns):
        """分析风险指标"""
        if returns is None:
            return None

        # 计算波动率
        volatility = returns.std() * np.sqrt(252)
        
        # 计算VaR (Value at Risk)
        var_95 = np.percentile(returns, 5)
        
        # 计算最大回撤
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = cumulative_returns / rolling_max - 1
        max_drawdown = drawdowns.min()
        
        return {
            'volatility': volatility,
            'var_95': var_95,
            'max_drawdown': max_drawdown
        } 