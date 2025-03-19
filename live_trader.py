import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from strategies import BaseStrategy
from ibkr_handler import IBKRHandler
from database import Database
from utils import calculate_returns, calculate_volatility, calculate_sharpe_ratio, calculate_max_drawdown

logger = logging.getLogger(__name__)

class LiveTrader:
    def __init__(self, strategy: BaseStrategy, ibkr_handler: IBKRHandler, database: Database):
        self.strategy = strategy
        self.ibkr = ibkr_handler
        self.db = database
        self.positions: Dict[str, float] = {}
        self.trades: List[Dict] = []
        self.equity_curve: List[float] = []
        self.dates: List[datetime] = []
        self.commission_rate = 0.001  # 0.1% 交易费用

    def initialize(self):
        """初始化交易系统"""
        try:
            # 连接IBKR
            if not self.ibkr.connect():
                raise Exception("Failed to connect to IBKR")

            # 连接数据库
            if not self.db.connect():
                raise Exception("Failed to connect to database")

            # 获取当前持仓
            positions_df = self.db.get_positions()
            if positions_df is not None and not positions_df.empty:
                for _, row in positions_df.iterrows():
                    self.positions[row['symbol']] = row['quantity']

            # 获取账户信息
            account_summary = self.ibkr.get_account_summary()
            if account_summary is not None:
                self.initial_capital = float(account_summary[account_summary['tag'] == 'NetLiquidation']['value'].iloc[0])
                self.current_capital = self.initial_capital
                self.equity_curve.append(self.initial_capital)
                self.dates.append(datetime.now())

            logger.info("Trading system initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Error initializing trading system: {str(e)}")
            return False

    def update_market_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """更新市场数据"""
        try:
            # 获取历史数据
            historical_data = self.ibkr.get_historical_data(symbol)
            if historical_data is None:
                return None

            # 计算技术指标
            data = self.strategy.calculate_signals(historical_data)
            return data
        except Exception as e:
            logger.error(f"Error updating market data: {str(e)}")
            return None

    def check_signals(self, data: pd.DataFrame) -> List[Dict]:
        """检查交易信号"""
        try:
            signals = []
            if data is None or data.empty:
                return signals

            # 获取最新数据
            latest_data = data.iloc[-1]
            
            # 检查止损止盈
            for symbol in self.positions:
                if self.strategy.check_stop_loss_take_profit(symbol, latest_data['Close']):
                    signals.append({
                        'symbol': symbol,
                        'signal': -1 if self.positions[symbol] > 0 else 1,
                        'price': latest_data['Close'],
                        'type': 'Stop Loss/Take Profit'
                    })

            # 检查新信号
            if 'Signal' in latest_data and latest_data['Signal'] != 0:
                signals.append({
                    'symbol': data.index[-1],
                    'signal': latest_data['Signal'],
                    'price': latest_data['Close'],
                    'type': 'Strategy Signal'
                })

            return signals
        except Exception as e:
            logger.error(f"Error checking signals: {str(e)}")
            return []

    def execute_trades(self, signals: List[Dict]) -> List[Dict]:
        """执行交易"""
        try:
            executed_trades = []
            for signal in signals:
                # 执行交易
                trade = self.strategy.execute_trade(
                    signal['symbol'],
                    signal['signal'],
                    signal['price']
                )
                
                if trade:
                    # 添加交易费用
                    trade['commission'] = trade['price'] * trade['quantity'] * self.commission_rate
                    
                    # 通过IBKR执行交易
                    if self.ibkr.place_order(
                        trade['symbol'],
                        trade['quantity'],
                        trade['type'],
                        'MKT'
                    ):
                        # 保存交易记录
                        self.db.save_trade(trade)
                        executed_trades.append(trade)
                        
                        # 更新持仓
                        self.positions[trade['symbol']] = self.positions.get(trade['symbol'], 0) + trade['quantity']
            
            return executed_trades
        except Exception as e:
            logger.error(f"Error executing trades: {str(e)}")
            return []

    def update_positions(self):
        """更新持仓信息"""
        try:
            for symbol in self.positions:
                if self.positions[symbol] != 0:
                    # 获取当前价格
                    current_price = self.ibkr.get_historical_data(symbol).iloc[-1]['Close']
                    
                    # 计算未实现盈亏
                    position_data = {
                        'symbol': symbol,
                        'quantity': self.positions[symbol],
                        'average_price': self.trades[-1]['price'] if self.trades else 0,
                        'current_price': current_price,
                        'unrealized_pnl': (current_price - self.trades[-1]['price']) * self.positions[symbol] if self.trades else 0,
                        'timestamp': datetime.now()
                    }
                    
                    # 更新数据库
                    self.db.save_position(position_data)
        except Exception as e:
            logger.error(f"Error updating positions: {str(e)}")

    def update_equity_curve(self):
        """更新权益曲线"""
        try:
            # 获取账户总价值
            account_summary = self.ibkr.get_account_summary()
            if account_summary is not None:
                current_equity = float(account_summary[account_summary['tag'] == 'NetLiquidation']['value'].iloc[0])
                self.equity_curve.append(current_equity)
                self.dates.append(datetime.now())
        except Exception as e:
            logger.error(f"Error updating equity curve: {str(e)}")

    def get_performance_metrics(self) -> Dict:
        """获取性能指标"""
        try:
            if not self.equity_curve:
                return {
                    'total_return': 0,
                    'sharpe_ratio': 0,
                    'max_drawdown': 0,
                    'win_rate': 0
                }

            # 计算收益率
            returns = calculate_returns(pd.Series(self.equity_curve))
            total_return = (self.equity_curve[-1] - self.initial_capital) / self.initial_capital

            # 计算风险指标
            sharpe_ratio = calculate_sharpe_ratio(returns)
            max_drawdown = calculate_max_drawdown(pd.Series(self.equity_curve))

            # 获取交易统计
            trades_df = self.db.get_trades()
            if trades_df is not None and not trades_df.empty:
                winning_trades = len(trades_df[trades_df['realized_pnl'] > 0])
                total_trades = len(trades_df)
                win_rate = winning_trades / total_trades if total_trades > 0 else 0
            else:
                win_rate = 0

            return {
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate
            }
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {str(e)}")
            return {}

    def shutdown(self):
        """关闭交易系统"""
        try:
            # 平掉所有持仓
            for symbol in self.positions:
                if self.positions[symbol] != 0:
                    self.ibkr.place_order(
                        symbol,
                        abs(self.positions[symbol]),
                        'SELL' if self.positions[symbol] > 0 else 'BUY',
                        'MKT'
                    )

            # 断开连接
            self.ibkr.disconnect()
            self.db.disconnect()

            logger.info("Trading system shut down successfully")
        except Exception as e:
            logger.error(f"Error shutting down trading system: {str(e)}") 