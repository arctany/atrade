from ib_insync import *
import pandas as pd
from datetime import datetime, timedelta
import logging

class IBKRHandler:
    def __init__(self):
        self.ib = IB()
        self.connected = False
        self.logger = logging.getLogger(__name__)

    def connect(self, host='127.0.0.1', port=7497, client_id=0):
        """连接到IBKR TWS或IB Gateway"""
        try:
            self.ib.connect(host, port, clientId=client_id)
            self.connected = True
            self.logger.info("Successfully connected to IBKR")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to IBKR: {str(e)}")
            return False

    def disconnect(self):
        """断开与IBKR的连接"""
        if self.connected:
            self.ib.disconnect()
            self.connected = False
            self.logger.info("Disconnected from IBKR")

    def get_account_summary(self):
        """获取账户摘要信息"""
        if not self.connected:
            return None
        
        try:
            account = self.ib.managedAccounts()[0]
            summary = self.ib.accountSummary(account)
            return pd.DataFrame(summary)
        except Exception as e:
            self.logger.error(f"Failed to get account summary: {str(e)}")
            return None

    def get_positions(self):
        """获取当前持仓信息"""
        if not self.connected:
            return None
        
        try:
            positions = self.ib.positions()
            return pd.DataFrame(positions)
        except Exception as e:
            self.logger.error(f"Failed to get positions: {str(e)}")
            return None

    def get_trade_history(self, days=30):
        """获取交易历史"""
        if not self.connected:
            return None
        
        try:
            account = self.ib.managedAccounts()[0]
            trades = self.ib.trades()
            df = pd.DataFrame(trades)
            df['date'] = pd.to_datetime(df['date'])
            df = df[df['date'] > datetime.now() - timedelta(days=days)]
            return df
        except Exception as e:
            self.logger.error(f"Failed to get trade history: {str(e)}")
            return None

    def get_historical_data(self, symbol, duration='1 Y', bar_size='1 day'):
        """获取历史数据"""
        if not self.connected:
            return None
        
        try:
            contract = Stock(symbol, 'SMART', 'USD')
            bars = self.ib.reqHistoricalData(
                contract,
                endDateTime='',
                durationStr=duration,
                barSizeSetting=bar_size
            )
            df = pd.DataFrame(bars)
            df['date'] = pd.to_datetime(df['date'])
            return df
        except Exception as e:
            self.logger.error(f"Failed to get historical data: {str(e)}")
            return None

    def place_order(self, symbol, quantity, action, order_type='MKT'):
        """下单"""
        if not self.connected:
            return None
        
        try:
            contract = Stock(symbol, 'SMART', 'USD')
            if order_type == 'MKT':
                order = MarketOrder(action, quantity)
            else:
                order = LimitOrder(action, quantity, price)
            
            trade = self.ib.placeOrder(contract, order)
            return trade
        except Exception as e:
            self.logger.error(f"Failed to place order: {str(e)}")
            return None

    def cancel_order(self, order_id):
        """取消订单"""
        if not self.connected:
            return False
        
        try:
            self.ib.cancelOrder(order_id)
            return True
        except Exception as e:
            self.logger.error(f"Failed to cancel order: {str(e)}")
            return False 