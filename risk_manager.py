import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from datetime import datetime
from config import TRADING_STRATEGY, RISK_MANAGEMENT_CONFIG

logger = logging.getLogger(__name__)

class RiskManager:
    def __init__(self, data: pd.DataFrame):
        """
        初始化风险管理器
        
        Args:
            data: 包含OHLCV数据的DataFrame
        """
        self.data = data.copy()
        self.positions = {}
        self.risk_metrics = {}
        self.risk_limits = RISK_MANAGEMENT_CONFIG
        
    def check_position_risk(
        self,
        strategy_name: str,
        symbol: str,
        size: float,
        price: float
    ) -> Dict:
        """
        检查持仓风险
        
        Args:
            strategy_name: 策略名称
            symbol: 交易品种
            size: 持仓数量
            price: 当前价格
            
        Returns:
            风险检查结果
        """
        try:
            # 计算持仓价值
            position_value = size * price
            
            # 检查持仓规模限制
            if position_value > self.risk_limits['max_position_size']:
                return {
                    'allowed': False,
                    'reason': 'Position size exceeds limit',
                    'limit': self.risk_limits['max_position_size'],
                    'current': position_value
                }
            
            # 检查单个品种持仓限制
            if symbol in self.positions:
                current_value = self.positions[symbol]['size'] * self.positions[symbol]['price']
                if current_value + position_value > self.risk_limits['max_position_size']:
                    return {
                        'allowed': False,
                        'reason': 'Symbol position size exceeds limit',
                        'limit': self.risk_limits['max_position_size'],
                        'current': current_value + position_value
                    }
            
            # 检查总持仓限制
            total_value = sum(
                p['size'] * p['price']
                for p in self.positions.values()
            )
            if total_value + position_value > self.risk_limits['max_total_position']:
                return {
                    'allowed': False,
                    'reason': 'Total position size exceeds limit',
                    'limit': self.risk_limits['max_total_position'],
                    'current': total_value + position_value
                }
            
            # 检查波动率限制
            volatility = self._calculate_volatility(symbol)
            if volatility > self.risk_limits['volatility_limit']:
                return {
                    'allowed': False,
                    'reason': 'Volatility exceeds limit',
                    'limit': self.risk_limits['volatility_limit'],
                    'current': volatility
                }
            
            # 检查Beta限制
            beta = self._calculate_beta(symbol)
            if beta > self.risk_limits['beta_limit']:
                return {
                    'allowed': False,
                    'reason': 'Beta exceeds limit',
                    'limit': self.risk_limits['beta_limit'],
                    'current': beta
                }
            
            return {
                'allowed': True,
                'reason': 'Position risk within limits',
                'position_value': position_value,
                'volatility': volatility,
                'beta': beta
            }
            
        except Exception as e:
            logger.error(f"Error checking position risk: {str(e)}")
            return {
                'allowed': False,
                'reason': f'Error: {str(e)}'
            }
    
    def update_position(
        self,
        strategy_name: str,
        symbol: str,
        size: float,
        price: float,
        action: str
    ):
        """
        更新持仓
        
        Args:
            strategy_name: 策略名称
            symbol: 交易品种
            size: 持仓数量
            price: 当前价格
            action: 交易动作
        """
        try:
            if action == 'BUY':
                self.positions[symbol] = {
                    'strategy': strategy_name,
                    'size': size,
                    'price': price,
                    'entry_time': datetime.now()
                }
            elif action == 'SELL' and symbol in self.positions:
                del self.positions[symbol]
                
        except Exception as e:
            logger.error(f"Error updating position: {str(e)}")
    
    def calculate_portfolio_risk(self) -> Dict:
        """
        计算投资组合风险
        
        Returns:
            投资组合风险指标
        """
        try:
            if not self.positions:
                return {
                    'total_value': 0.0,
                    'volatility': 0.0,
                    'beta': 0.0,
                    'var': 0.0,
                    'sharpe_ratio': 0.0
                }
            
            # 计算总价值
            total_value = sum(
                p['size'] * p['price']
                for p in self.positions.values()
            )
            
            # 计算投资组合波动率
            portfolio_volatility = self._calculate_portfolio_volatility()
            
            # 计算投资组合Beta
            portfolio_beta = self._calculate_portfolio_beta()
            
            # 计算VaR
            var = self._calculate_var()
            
            # 计算夏普比率
            sharpe_ratio = self._calculate_sharpe_ratio()
            
            return {
                'total_value': total_value,
                'volatility': portfolio_volatility,
                'beta': portfolio_beta,
                'var': var,
                'sharpe_ratio': sharpe_ratio,
                'positions': len(self.positions)
            }
            
        except Exception as e:
            logger.error(f"Error calculating portfolio risk: {str(e)}")
            return {}
    
    def _calculate_volatility(self, symbol: str) -> float:
        """计算波动率"""
        try:
            returns = self.data[symbol].pct_change().dropna()
            return returns.std() * np.sqrt(252)
            
        except Exception as e:
            logger.error(f"Error calculating volatility: {str(e)}")
            return 0.0
    
    def _calculate_beta(self, symbol: str) -> float:
        """计算Beta"""
        try:
            # 使用市场指数作为基准
            market_returns = self.data['SPY'].pct_change().dropna()
            symbol_returns = self.data[symbol].pct_change().dropna()
            
            # 计算协方差和方差
            covariance = np.cov(symbol_returns, market_returns)[0][1]
            market_variance = np.var(market_returns)
            
            return covariance / market_variance
            
        except Exception as e:
            logger.error(f"Error calculating beta: {str(e)}")
            return 0.0
    
    def _calculate_portfolio_volatility(self) -> float:
        """计算投资组合波动率"""
        try:
            # 计算权重
            weights = {}
            total_value = sum(p['size'] * p['price'] for p in self.positions.values())
            
            for symbol, position in self.positions.items():
                weights[symbol] = (position['size'] * position['price']) / total_value
            
            # 计算协方差矩阵
            returns = pd.DataFrame()
            for symbol in self.positions.keys():
                returns[symbol] = self.data[symbol].pct_change()
            
            cov_matrix = returns.cov()
            
            # 计算投资组合波动率
            portfolio_volatility = np.sqrt(
                np.dot(
                    np.array(list(weights.values())),
                    np.dot(cov_matrix, np.array(list(weights.values())))
                )
            ) * np.sqrt(252)
            
            return portfolio_volatility
            
        except Exception as e:
            logger.error(f"Error calculating portfolio volatility: {str(e)}")
            return 0.0
    
    def _calculate_portfolio_beta(self) -> float:
        """计算投资组合Beta"""
        try:
            # 计算权重
            weights = {}
            total_value = sum(p['size'] * p['price'] for p in self.positions.values())
            
            for symbol, position in self.positions.items():
                weights[symbol] = (position['size'] * position['price']) / total_value
            
            # 计算加权平均Beta
            portfolio_beta = sum(
                weights[symbol] * self._calculate_beta(symbol)
                for symbol in self.positions.keys()
            )
            
            return portfolio_beta
            
        except Exception as e:
            logger.error(f"Error calculating portfolio beta: {str(e)}")
            return 0.0
    
    def _calculate_var(self, confidence_level: float = 0.95) -> float:
        """计算VaR"""
        try:
            # 计算投资组合收益
            portfolio_returns = pd.Series(0.0, index=self.data.index)
            for symbol, position in self.positions.items():
                portfolio_returns += (
                    position['size'] * position['price'] *
                    self.data[symbol].pct_change()
                )
            
            # 计算VaR
            var = np.percentile(
                portfolio_returns.dropna(),
                (1 - confidence_level) * 100
            )
            
            return abs(var)
            
        except Exception as e:
            logger.error(f"Error calculating VaR: {str(e)}")
            return 0.0
    
    def _calculate_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """计算夏普比率"""
        try:
            # 计算投资组合收益
            portfolio_returns = pd.Series(0.0, index=self.data.index)
            for symbol, position in self.positions.items():
                portfolio_returns += (
                    position['size'] * position['price'] *
                    self.data[symbol].pct_change()
                )
            
            # 计算年化收益率和波动率
            annual_return = portfolio_returns.mean() * 252
            annual_volatility = portfolio_returns.std() * np.sqrt(252)
            
            # 计算夏普比率
            sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility
            
            return sharpe_ratio
            
        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {str(e)}")
            return 0.0
    
    def get_risk_report(self) -> Dict:
        """
        生成风险报告
        
        Returns:
            风险报告
        """
        try:
            portfolio_risk = self.calculate_portfolio_risk()
            
            return {
                'portfolio_risk': portfolio_risk,
                'positions': self.positions,
                'risk_limits': self.risk_limits,
                'generated_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating risk report: {str(e)}")
            return {} 