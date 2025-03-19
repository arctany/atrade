import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import yfinance as yf
from indicators import TechnicalIndicators
from config import MARKET_INDICES, INDUSTRY_ETF_MAPPING

logger = logging.getLogger(__name__)

class MarketAnalyzer:
    def __init__(self):
        """初始化市场分析器"""
        self.cache = {}
        self.cache_timeout = timedelta(minutes=5)

    def get_market_data(
        self,
        symbol: str,
        period: str = "1y",
        interval: str = "1d"
    ) -> Optional[pd.DataFrame]:
        """
        获取市场数据
        
        Args:
            symbol: 股票代码
            period: 时间周期
            interval: 时间间隔
            
        Returns:
            包含OHLCV数据的DataFrame
        """
        try:
            # 检查缓存
            cache_key = f"{symbol}_{period}_{interval}"
            if cache_key in self.cache:
                data, timestamp = self.cache[cache_key]
                if datetime.now() - timestamp < self.cache_timeout:
                    return data

            # 获取数据
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                logger.warning(f"No data available for {symbol}")
                return None

            # 更新缓存
            self.cache[cache_key] = (data, datetime.now())
            
            return data
        except Exception as e:
            logger.error(f"Error fetching market data: {str(e)}")
            return None

    def get_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d"
    ) -> Optional[pd.DataFrame]:
        """
        获取历史数据
        
        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            interval: 时间间隔
            
        Returns:
            包含OHLCV数据的DataFrame
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(
                start=start_date,
                end=end_date,
                interval=interval
            )
            
            if data.empty:
                logger.warning(f"No historical data available for {symbol}")
                return None
            
            return data
        except Exception as e:
            logger.error(f"Error fetching historical data: {str(e)}")
            return None

    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        计算技术指标
        
        Args:
            data: 包含OHLCV数据的DataFrame
            
        Returns:
            技术指标字典
        """
        try:
            indicators = TechnicalIndicators(data)
            all_indicators = indicators.calculate_all()
            
            # 获取最新值
            latest_indicators = {}
            for name, series in all_indicators.items():
                if not series.empty:
                    latest_indicators[name] = series.iloc[-1]
            
            return latest_indicators
        except Exception as e:
            logger.error(f"Error calculating indicators: {str(e)}")
            return {}

    def get_market_trend(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        分析市场趋势
        
        Args:
            data: 包含OHLCV数据的DataFrame
            
        Returns:
            趋势指标字典
        """
        try:
            indicators = TechnicalIndicators(data)
            trend_indicators = indicators.calculate_trend_indicators()
            
            # 计算趋势强度
            trend_strength = {
                'SMA_20': trend_indicators['SMA_20'].iloc[-1],
                'SMA_50': trend_indicators['SMA_50'].iloc[-1],
                'SMA_200': trend_indicators['SMA_200'].iloc[-1],
                'ADX': trend_indicators['ADX'].iloc[-1],
                'MACD': trend_indicators['MACD'].iloc[-1],
                'MACD_Signal': trend_indicators['MACD_Signal'].iloc[-1]
            }
            
            return trend_strength
        except Exception as e:
            logger.error(f"Error analyzing market trend: {str(e)}")
            return {}

    def get_market_momentum(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        分析市场动量
        
        Args:
            data: 包含OHLCV数据的DataFrame
            
        Returns:
            动量指标字典
        """
        try:
            indicators = TechnicalIndicators(data)
            momentum_indicators = indicators.calculate_momentum_indicators()
            
            # 计算动量指标
            momentum = {
                'RSI_14': momentum_indicators['RSI_14'].iloc[-1],
                'Stoch_K': momentum_indicators['Stoch_K'].iloc[-1],
                'Stoch_D': momentum_indicators['Stoch_D'].iloc[-1],
                'Williams_R': momentum_indicators['Williams_R'].iloc[-1],
                'ROC': momentum_indicators['ROC'].iloc[-1],
                'MFI': momentum_indicators['MFI'].iloc[-1]
            }
            
            return momentum
        except Exception as e:
            logger.error(f"Error analyzing market momentum: {str(e)}")
            return {}

    def get_market_volatility(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        分析市场波动率
        
        Args:
            data: 包含OHLCV数据的DataFrame
            
        Returns:
            波动率指标字典
        """
        try:
            indicators = TechnicalIndicators(data)
            volatility_indicators = indicators.calculate_volatility_indicators()
            
            # 计算波动率指标
            volatility = {
                'BB_Upper': volatility_indicators['BB_Upper'].iloc[-1],
                'BB_Middle': volatility_indicators['BB_Middle'].iloc[-1],
                'BB_Lower': volatility_indicators['BB_Lower'].iloc[-1],
                'BB_Width': volatility_indicators['BB_Width'].iloc[-1],
                'ATR': volatility_indicators['ATR'].iloc[-1],
                'Ulcer_Index': volatility_indicators['Ulcer_Index'].iloc[-1]
            }
            
            return volatility
        except Exception as e:
            logger.error(f"Error analyzing market volatility: {str(e)}")
            return {}

    def get_market_volume(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        分析市场成交量
        
        Args:
            data: 包含OHLCV数据的DataFrame
            
        Returns:
            成交量指标字典
        """
        try:
            indicators = TechnicalIndicators(data)
            volume_indicators = indicators.calculate_volume_indicators()
            
            # 计算成交量指标
            volume = {
                'OBV': volume_indicators['OBV'].iloc[-1],
                'ADI': volume_indicators['ADI'].iloc[-1],
                'CMF': volume_indicators['CMF'].iloc[-1],
                'VWAP': volume_indicators['VWAP'].iloc[-1]
            }
            
            return volume
        except Exception as e:
            logger.error(f"Error analyzing market volume: {str(e)}")
            return {}

    def get_support_resistance(
        self,
        data: pd.DataFrame,
        window: int = 20,
        threshold: float = 0.02
    ) -> Tuple[List[float], List[float]]:
        """
        计算支撑位和阻力位
        
        Args:
            data: 包含OHLCV数据的DataFrame
            window: 计算窗口大小
            threshold: 价格变化阈值
            
        Returns:
            支撑位和阻力位列表
        """
        try:
            indicators = TechnicalIndicators(data)
            return indicators.get_support_resistance_levels(window, threshold)
        except Exception as e:
            logger.error(f"Error calculating support/resistance levels: {str(e)}")
            return [], []

    def get_price_patterns(self, data: pd.DataFrame) -> List[Dict]:
        """
        识别价格形态
        
        Args:
            data: 包含OHLCV数据的DataFrame
            
        Returns:
            价格形态列表
        """
        try:
            indicators = TechnicalIndicators(data)
            return indicators.get_price_patterns()
        except Exception as e:
            logger.error(f"Error identifying price patterns: {str(e)}")
            return []

    def get_volume_profile(self, data: pd.DataFrame, bins: int = 10) -> Dict[float, float]:
        """
        计算成交量分布
        
        Args:
            data: 包含OHLCV数据的DataFrame
            bins: 价格区间数量
            
        Returns:
            价格区间对应的成交量
        """
        try:
            indicators = TechnicalIndicators(data)
            return indicators.get_volume_profile(bins)
        except Exception as e:
            logger.error(f"Error calculating volume profile: {str(e)}")
            return {}

    def get_industry_analysis(self, industry: str) -> Dict[str, pd.DataFrame]:
        """
        分析行业数据
        
        Args:
            industry: 行业名称
            
        Returns:
            行业分析数据
        """
        try:
            if industry not in INDUSTRY_ETF_MAPPING:
                logger.warning(f"Industry {industry} not found in mapping")
                return {}

            etf_symbol = INDUSTRY_ETF_MAPPING[industry]
            data = self.get_market_data(etf_symbol)
            
            if data is None:
                return {}

            return {
                'price_data': data,
                'indicators': self.calculate_indicators(data),
                'trend': self.get_market_trend(data),
                'momentum': self.get_market_momentum(data),
                'volatility': self.get_market_volatility(data),
                'volume': self.get_market_volume(data)
            }
        except Exception as e:
            logger.error(f"Error analyzing industry: {str(e)}")
            return {} 