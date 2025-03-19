import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from typing import Dict, List, Union, Optional
import logging

logger = logging.getLogger(__name__)

def calculate_returns(prices: pd.Series) -> pd.Series:
    """计算收益率"""
    return prices.pct_change()

def calculate_volatility(returns: pd.Series, window: int = 252) -> pd.Series:
    """计算波动率"""
    return returns.rolling(window=window).std() * np.sqrt(252)

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """计算夏普比率"""
    if len(returns) < 2:
        return 0
    excess_returns = returns - risk_free_rate/252
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

def calculate_max_drawdown(prices: pd.Series) -> float:
    """计算最大回撤"""
    rolling_max = prices.expanding().max()
    drawdowns = prices / rolling_max - 1
    return drawdowns.min()

def calculate_position_size(
    capital: float,
    price: float,
    risk_per_trade: float = 0.02,
    stop_loss: float = 0.02
) -> int:
    """计算仓位大小"""
    risk_amount = capital * risk_per_trade
    price_risk = price * stop_loss
    return int(risk_amount / price_risk)

def format_currency(value: float) -> str:
    """格式化货币金额"""
    return f"${value:,.2f}"

def format_percentage(value: float) -> str:
    """格式化百分比"""
    return f"{value:.2f}%"

def get_market_cap_category(market_cap: float) -> str:
    """获取市值分类"""
    if market_cap >= 200e9:
        return "Large Cap"
    elif market_cap >= 10e9:
        return "Mid Cap"
    else:
        return "Small Cap"

def calculate_beta(
    stock_returns: pd.Series,
    market_returns: pd.Series,
    window: int = 252
) -> float:
    """计算贝塔系数"""
    if len(stock_returns) < window or len(market_returns) < window:
        return 0
    
    covariance = stock_returns.rolling(window=window).cov(market_returns)
    market_variance = market_returns.rolling(window=window).var()
    beta = covariance / market_variance
    return beta.mean()

def get_sector_performance(sector: str) -> Optional[Dict]:
    """获取行业表现"""
    try:
        # 使用SPDR行业ETF作为行业代表
        etf_mapping = {
            'Technology': 'XLK',
            'Healthcare': 'XLV',
            'Financial': 'XLF',
            'Consumer': 'XLP',
            'Industrial': 'XLI',
            'Energy': 'XLE',
            'Materials': 'XLB',
            'Utilities': 'XLU',
            'Real Estate': 'XLRE',
            'Communication': 'XLC'
        }
        
        if sector in etf_mapping:
            etf = yf.Ticker(etf_mapping[sector])
            info = etf.info
            return {
                'name': sector,
                'price': info.get('regularMarketPrice', 0),
                'change': info.get('regularMarketChange', 0),
                'change_percent': info.get('regularMarketChangePercent', 0),
                'volume': info.get('regularMarketVolume', 0)
            }
        return None
    except Exception as e:
        logger.error(f"Error getting sector performance: {str(e)}")
        return None

def calculate_trade_metrics(trades: pd.DataFrame) -> Dict:
    """计算交易指标"""
    if len(trades) == 0:
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'avg_profit': 0,
            'avg_loss': 0,
            'profit_factor': 0,
            'max_drawdown': 0
        }
    
    winning_trades = trades[trades['realized_pnl'] > 0]
    losing_trades = trades[trades['realized_pnl'] < 0]
    
    total_trades = len(trades)
    winning_count = len(winning_trades)
    losing_count = len(losing_trades)
    
    win_rate = winning_count / total_trades if total_trades > 0 else 0
    avg_profit = winning_trades['realized_pnl'].mean() if len(winning_trades) > 0 else 0
    avg_loss = losing_trades['realized_pnl'].mean() if len(losing_trades) > 0 else 0
    
    total_profit = winning_trades['realized_pnl'].sum() if len(winning_trades) > 0 else 0
    total_loss = abs(losing_trades['realized_pnl'].sum()) if len(losing_trades) > 0 else 0
    profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
    
    cumulative_returns = trades['realized_pnl'].cumsum()
    rolling_max = cumulative_returns.expanding().max()
    drawdowns = cumulative_returns - rolling_max
    max_drawdown = drawdowns.min()
    
    return {
        'total_trades': total_trades,
        'winning_trades': winning_count,
        'losing_trades': losing_count,
        'win_rate': win_rate,
        'avg_profit': avg_profit,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'max_drawdown': max_drawdown
    }

def calculate_position_metrics(positions: pd.DataFrame) -> Dict:
    """计算持仓指标"""
    if len(positions) == 0:
        return {
            'total_positions': 0,
            'total_value': 0,
            'total_unrealized_pnl': 0,
            'avg_position_size': 0,
            'largest_position': 0,
            'smallest_position': 0
        }
    
    total_value = (positions['quantity'] * positions['current_price']).sum()
    total_unrealized_pnl = positions['unrealized_pnl'].sum()
    position_sizes = positions['quantity'] * positions['current_price']
    
    return {
        'total_positions': len(positions),
        'total_value': total_value,
        'total_unrealized_pnl': total_unrealized_pnl,
        'avg_position_size': position_sizes.mean(),
        'largest_position': position_sizes.max(),
        'smallest_position': position_sizes.min()
    } 