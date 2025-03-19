import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from ta.trend import (
    SMAIndicator, EMAIndicator, MACD, IchimokuIndicator,
    ADXIndicator, CCIIndicator
)
from ta.momentum import (
    RSIIndicator, StochasticOscillator, WilliamsRIndicator,
    ROCIndicator, MFIIndicator
)
from ta.volatility import (
    BollingerBands, AverageTrueRange, DonchianChannel,
    UlcerIndex
)
from ta.volume import (
    OnBalanceVolumeIndicator, AccDistIndexIndicator,
    ChaikinMoneyFlowIndicator, VolumeWeightedAveragePrice
)

logger = logging.getLogger(__name__)

class TechnicalIndicators:
    def __init__(self, data: pd.DataFrame):
        """
        初始化技术指标计算器
        
        Args:
            data: 包含OHLCV数据的DataFrame
        """
        self.data = data.copy()
        self._validate_data()
        
    def _validate_data(self):
        """验证输入数据的完整性"""
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

    def calculate_all(self) -> Dict[str, pd.Series]:
        """
        计算所有技术指标
        
        Returns:
            包含所有技术指标的字典
        """
        try:
            indicators = {}
            
            # 趋势指标
            indicators.update(self.calculate_trend_indicators())
            
            # 动量指标
            indicators.update(self.calculate_momentum_indicators())
            
            # 波动率指标
            indicators.update(self.calculate_volatility_indicators())
            
            # 成交量指标
            indicators.update(self.calculate_volume_indicators())
            
            return indicators
        except Exception as e:
            logger.error(f"Error calculating indicators: {str(e)}")
            raise

    def calculate_trend_indicators(self) -> Dict[str, pd.Series]:
        """计算趋势指标"""
        indicators = {}
        
        # 移动平均线
        for period in [5, 10, 20, 50, 200]:
            sma = SMAIndicator(close=self.data['Close'], window=period)
            ema = EMAIndicator(close=self.data['Close'], window=period)
            indicators[f'SMA_{period}'] = sma.sma_indicator()
            indicators[f'EMA_{period}'] = ema.ema_indicator()
        
        # MACD
        macd = MACD(close=self.data['Close'])
        indicators['MACD'] = macd.macd()
        indicators['MACD_Signal'] = macd.macd_signal()
        indicators['MACD_Hist'] = macd.macd_diff()
        
        # 一目均衡图
        ichimoku = IchimokuIndicator(
            high=self.data['High'],
            low=self.data['Low']
        )
        indicators['Ichimoku_Span_A'] = ichimoku.ichimoku_a()
        indicators['Ichimoku_Span_B'] = ichimoku.ichimoku_b()
        indicators['Ichimoku_Conversion'] = ichimoku.ichimoku_conversion_line()
        indicators['Ichimoku_Base'] = ichimoku.ichimoku_base_line()
        
        # ADX
        adx = ADXIndicator(
            high=self.data['High'],
            low=self.data['Low'],
            close=self.data['Close']
        )
        indicators['ADX'] = adx.adx()
        indicators['ADX_Pos'] = adx.adx_pos()
        indicators['ADX_Neg'] = adx.adx_neg()
        
        # CCI
        cci = CCIIndicator(
            high=self.data['High'],
            low=self.data['Low'],
            close=self.data['Close']
        )
        indicators['CCI'] = cci.cci()
        
        return indicators

    def calculate_momentum_indicators(self) -> Dict[str, pd.Series]:
        """计算动量指标"""
        indicators = {}
        
        # RSI
        for period in [6, 12, 24]:
            rsi = RSIIndicator(close=self.data['Close'], window=period)
            indicators[f'RSI_{period}'] = rsi.rsi()
        
        # 随机指标
        stoch = StochasticOscillator(
            high=self.data['High'],
            low=self.data['Low'],
            close=self.data['Close']
        )
        indicators['Stoch_K'] = stoch.stoch()
        indicators['Stoch_D'] = stoch.stoch_signal()
        
        # Williams %R
        williams = WilliamsRIndicator(
            high=self.data['High'],
            low=self.data['Low'],
            close=self.data['Close']
        )
        indicators['Williams_R'] = williams.williams_r()
        
        # ROC
        roc = ROCIndicator(close=self.data['Close'])
        indicators['ROC'] = roc.roc()
        
        # MFI
        mfi = MFIIndicator(
            high=self.data['High'],
            low=self.data['Low'],
            close=self.data['Close'],
            volume=self.data['Volume']
        )
        indicators['MFI'] = mfi.money_flow_index()
        
        return indicators

    def calculate_volatility_indicators(self) -> Dict[str, pd.Series]:
        """计算波动率指标"""
        indicators = {}
        
        # 布林带
        bb = BollingerBands(close=self.data['Close'])
        indicators['BB_Upper'] = bb.bollinger_hband()
        indicators['BB_Middle'] = bb.bollinger_mavg()
        indicators['BB_Lower'] = bb.bollinger_lband()
        indicators['BB_Width'] = bb.bollinger_pband()
        
        # ATR
        atr = AverageTrueRange(
            high=self.data['High'],
            low=self.data['Low'],
            close=self.data['Close']
        )
        indicators['ATR'] = atr.average_true_range()
        
        # 唐奇安通道
        dc = DonchianChannel(
            high=self.data['High'],
            low=self.data['Low']
        )
        indicators['DC_Upper'] = dc.donchian_channel_hband()
        indicators['DC_Middle'] = dc.donchian_channel_mband()
        indicators['DC_Lower'] = dc.donchian_channel_lband()
        
        # 溃疡指数
        ui = UlcerIndex(close=self.data['Close'])
        indicators['Ulcer_Index'] = ui.ulcer_index()
        
        return indicators

    def calculate_volume_indicators(self) -> Dict[str, pd.Series]:
        """计算成交量指标"""
        indicators = {}
        
        # OBV
        obv = OnBalanceVolumeIndicator(
            close=self.data['Close'],
            volume=self.data['Volume']
        )
        indicators['OBV'] = obv.on_balance_volume()
        
        # ADI
        adi = AccDistIndexIndicator(
            high=self.data['High'],
            low=self.data['Low'],
            close=self.data['Close'],
            volume=self.data['Volume']
        )
        indicators['ADI'] = adi.acc_dist_index()
        
        # CMF
        cmf = ChaikinMoneyFlowIndicator(
            high=self.data['High'],
            low=self.data['Low'],
            close=self.data['Close'],
            volume=self.data['Volume']
        )
        indicators['CMF'] = cmf.chaikin_money_flow()
        
        # VWAP
        vwap = VolumeWeightedAveragePrice(
            high=self.data['High'],
            low=self.data['Low'],
            close=self.data['Close'],
            volume=self.data['Volume']
        )
        indicators['VWAP'] = vwap.volume_weighted_average_price()
        
        return indicators

    def get_support_resistance_levels(
        self,
        window: int = 20,
        threshold: float = 0.02
    ) -> Tuple[List[float], List[float]]:
        """
        计算支撑位和阻力位
        
        Args:
            window: 计算窗口大小
            threshold: 价格变化阈值
            
        Returns:
            支撑位和阻力位列表
        """
        try:
            highs = self.data['High'].rolling(window=window).max()
            lows = self.data['Low'].rolling(window=window).min()
            
            resistance_levels = []
            support_levels = []
            
            for i in range(window, len(self.data)):
                if highs[i] > highs[i-1] * (1 + threshold):
                    resistance_levels.append(highs[i])
                if lows[i] < lows[i-1] * (1 - threshold):
                    support_levels.append(lows[i])
            
            return support_levels, resistance_levels
        except Exception as e:
            logger.error(f"Error calculating support/resistance levels: {str(e)}")
            return [], []

    def get_price_patterns(self) -> List[Dict]:
        """
        识别价格形态
        
        Returns:
            价格形态列表
        """
        patterns = []
        
        # 双底形态
        for i in range(2, len(self.data)-2):
            if (self.data['Low'][i] < self.data['Low'][i-1] and
                self.data['Low'][i] < self.data['Low'][i+1] and
                self.data['Low'][i+2] < self.data['Low'][i+1] and
                self.data['Low'][i+2] < self.data['Low'][i+3]):
                patterns.append({
                    'type': 'Double Bottom',
                    'index': i,
                    'price': self.data['Low'][i]
                })
        
        # 双顶形态
        for i in range(2, len(self.data)-2):
            if (self.data['High'][i] > self.data['High'][i-1] and
                self.data['High'][i] > self.data['High'][i+1] and
                self.data['High'][i+2] > self.data['High'][i+1] and
                self.data['High'][i+2] > self.data['High'][i+3]):
                patterns.append({
                    'type': 'Double Top',
                    'index': i,
                    'price': self.data['High'][i]
                })
        
        return patterns

    def get_volume_profile(self, bins: int = 10) -> Dict[float, float]:
        """
        计算成交量分布
        
        Args:
            bins: 价格区间数量
            
        Returns:
            价格区间对应的成交量
        """
        try:
            price_bins = pd.qcut(self.data['Close'], q=bins, labels=False)
            volume_profile = self.data.groupby(price_bins)['Volume'].sum()
            return dict(zip(volume_profile.index, volume_profile.values))
        except Exception as e:
            logger.error(f"Error calculating volume profile: {str(e)}")
            return {} 