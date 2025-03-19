import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
import openai
import anthropic
import os
import json
import asyncio
import aiohttp
from config import AI_CONFIG

logger = logging.getLogger(__name__)

class AIStrategy:
    def __init__(
        self,
        data: pd.DataFrame,
        api_key: str,
        model_type: str = 'gpt-4',
        max_tokens: int = 1000,
        temperature: float = 0.7
    ):
        """
        初始化AI策略
        
        Args:
            data: 市场数据
            api_key: API密钥
            model_type: 模型类型 ('gpt-4', 'claude-2', 'custom')
            max_tokens: 最大token数
            temperature: 温度参数
        """
        self.data = data
        self.api_key = api_key
        self.model_type = model_type
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.signals = {}
        self.positions = {}
        self.performance_metrics = {}
        
        # 初始化API客户端
        if model_type == 'gpt-4':
            openai.api_key = api_key
        elif model_type == 'claude-2':
            self.client = anthropic.Client(api_key=api_key)
        elif model_type == 'custom':
            self.api_url = AI_CONFIG['custom_api_url']
            self.headers = {
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            }
    
    def generate_market_analysis(
        self,
        timeframe: str = '1d',
        symbols: List[str] = None
    ) -> Dict:
        """
        生成市场分析
        
        Args:
            timeframe: 时间周期
            symbols: 交易品种列表
            
        Returns:
            市场分析结果
        """
        try:
            # 准备市场数据
            market_data = self._prepare_market_data(timeframe, symbols)
            
            # 构建提示词
            prompt = self._build_analysis_prompt(market_data)
            
            # 获取AI分析
            analysis = self._get_ai_analysis(prompt)
            
            # 解析分析结果
            result = self._parse_analysis(analysis)
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating market analysis: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def generate_trading_signals(
        self,
        analysis: Dict,
        risk_level: str = 'medium'
    ) -> Dict:
        """
        生成交易信号
        
        Args:
            analysis: 市场分析结果
            risk_level: 风险等级 ('low', 'medium', 'high')
            
        Returns:
            交易信号
        """
        try:
            # 构建信号生成提示词
            prompt = self._build_signal_prompt(analysis, risk_level)
            
            # 获取AI信号
            signals = self._get_ai_signals(prompt)
            
            # 解析信号
            result = self._parse_signals(signals)
            
            # 更新信号历史
            self.signals[datetime.now()] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating trading signals: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def optimize_portfolio(
        self,
        signals: Dict,
        constraints: Optional[Dict] = None
    ) -> Dict:
        """
        优化投资组合
        
        Args:
            signals: 交易信号
            constraints: 约束条件
            
        Returns:
            优化结果
        """
        try:
            # 构建优化提示词
            prompt = self._build_optimization_prompt(signals, constraints)
            
            # 获取AI优化建议
            optimization = self._get_ai_optimization(prompt)
            
            # 解析优化结果
            result = self._parse_optimization(optimization)
            
            return result
            
        except Exception as e:
            logger.error(f"Error optimizing portfolio: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def generate_risk_analysis(
        self,
        positions: Dict,
        market_conditions: Dict
    ) -> Dict:
        """
        生成风险分析
        
        Args:
            positions: 当前持仓
            market_conditions: 市场状况
            
        Returns:
            风险分析结果
        """
        try:
            # 构建风险分析提示词
            prompt = self._build_risk_prompt(positions, market_conditions)
            
            # 获取AI风险分析
            risk_analysis = self._get_ai_risk_analysis(prompt)
            
            # 解析风险分析
            result = self._parse_risk_analysis(risk_analysis)
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating risk analysis: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def generate_performance_report(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> Dict:
        """
        生成绩效报告
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            绩效报告
        """
        try:
            # 准备绩效数据
            performance_data = self._prepare_performance_data(start_date, end_date)
            
            # 构建报告提示词
            prompt = self._build_report_prompt(performance_data)
            
            # 获取AI报告
            report = self._get_ai_report(prompt)
            
            # 解析报告
            result = self._parse_report(report)
            
            # 更新绩效指标
            self.performance_metrics[datetime.now()] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating performance report: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def _prepare_market_data(
        self,
        timeframe: str,
        symbols: List[str]
    ) -> Dict:
        """准备市场数据"""
        try:
            if symbols is None:
                symbols = self.data['symbol'].unique().tolist()
            
            market_data = {}
            for symbol in symbols:
                symbol_data = self.data[self.data['symbol'] == symbol].tail(100)
                market_data[symbol] = {
                    'price': symbol_data['close'].iloc[-1],
                    'volume': symbol_data['volume'].iloc[-1],
                    'change': (symbol_data['close'].iloc[-1] / symbol_data['close'].iloc[-2] - 1),
                    'high': symbol_data['high'].max(),
                    'low': symbol_data['low'].min(),
                    'volatility': symbol_data['close'].pct_change().std()
                }
            
            return market_data
            
        except Exception as e:
            logger.error(f"Error preparing market data: {str(e)}")
            raise
    
    def _build_analysis_prompt(self, market_data: Dict) -> str:
        """构建分析提示词"""
        prompt = f"""
        请分析以下市场数据并生成详细的市场分析报告：
        
        市场数据：
        {json.dumps(market_data, indent=2)}
        
        请提供：
        1. 市场趋势分析
        2. 关键支撑和阻力位
        3. 技术指标分析
        4. 市场情绪分析
        5. 潜在风险因素
        6. 交易机会
        """
        return prompt
    
    def _build_signal_prompt(self, analysis: Dict, risk_level: str) -> str:
        """构建信号提示词"""
        prompt = f"""
        基于以下市场分析和风险等级，生成具体的交易信号：
        
        市场分析：
        {json.dumps(analysis, indent=2)}
        
        风险等级：{risk_level}
        
        请提供：
        1. 交易方向（做多/做空/观望）
        2. 入场价格区间
        3. 止损位
        4. 目标位
        5. 仓位建议
        6. 交易理由
        """
        return prompt
    
    def _build_optimization_prompt(self, signals: Dict, constraints: Dict) -> str:
        """构建优化提示词"""
        prompt = f"""
        基于以下交易信号和约束条件，优化投资组合配置：
        
        交易信号：
        {json.dumps(signals, indent=2)}
        
        约束条件：
        {json.dumps(constraints, indent=2)}
        
        请提供：
        1. 最优权重配置
        2. 预期收益
        3. 风险指标
        4. 再平衡建议
        5. 风险控制措施
        """
        return prompt
    
    def _build_risk_prompt(self, positions: Dict, market_conditions: Dict) -> str:
        """构建风险分析提示词"""
        prompt = f"""
        基于以下持仓和市场状况，进行风险分析：
        
        当前持仓：
        {json.dumps(positions, indent=2)}
        
        市场状况：
        {json.dumps(market_conditions, indent=2)}
        
        请提供：
        1. 风险敞口分析
        2. 压力测试结果
        3. 风险预警指标
        4. 风险控制建议
        5. 应急预案
        """
        return prompt
    
    def _build_report_prompt(self, performance_data: Dict) -> str:
        """构建报告提示词"""
        prompt = f"""
        基于以下绩效数据，生成详细的绩效报告：
        
        绩效数据：
        {json.dumps(performance_data, indent=2)}
        
        请提供：
        1. 收益分析
        2. 风险分析
        3. 策略评估
        4. 改进建议
        5. 未来展望
        """
        return prompt
    
    async def _get_ai_analysis(self, prompt: str) -> str:
        """获取AI分析"""
        try:
            if self.model_type == 'gpt-4':
                response = await openai.ChatCompletion.acreate(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature
                )
                return response.choices[0].message.content
                
            elif self.model_type == 'claude-2':
                response = await self.client.messages.create(
                    model="claude-2",
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text
                
            elif self.model_type == 'custom':
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        self.api_url,
                        headers=self.headers,
                        json={"prompt": prompt}
                    ) as response:
                        result = await response.json()
                        return result['response']
            
        except Exception as e:
            logger.error(f"Error getting AI analysis: {str(e)}")
            raise
    
    async def _get_ai_signals(self, prompt: str) -> str:
        """获取AI交易信号"""
        return await self._get_ai_analysis(prompt)
    
    async def _get_ai_optimization(self, prompt: str) -> str:
        """获取AI优化建议"""
        return await self._get_ai_analysis(prompt)
    
    async def _get_ai_risk_analysis(self, prompt: str) -> str:
        """获取AI风险分析"""
        return await self._get_ai_analysis(prompt)
    
    async def _get_ai_report(self, prompt: str) -> str:
        """获取AI报告"""
        return await self._get_ai_analysis(prompt)
    
    def _parse_analysis(self, analysis: str) -> Dict:
        """解析分析结果"""
        try:
            # 这里需要根据实际的AI输出格式进行解析
            # 示例实现
            sections = analysis.split('\n\n')
            result = {
                'trend_analysis': sections[0],
                'support_resistance': sections[1],
                'technical_indicators': sections[2],
                'market_sentiment': sections[3],
                'risk_factors': sections[4],
                'trading_opportunities': sections[5]
            }
            return result
            
        except Exception as e:
            logger.error(f"Error parsing analysis: {str(e)}")
            raise
    
    def _parse_signals(self, signals: str) -> Dict:
        """解析交易信号"""
        try:
            # 这里需要根据实际的AI输出格式进行解析
            # 示例实现
            sections = signals.split('\n\n')
            result = {
                'direction': sections[0],
                'entry_price': sections[1],
                'stop_loss': sections[2],
                'target': sections[3],
                'position_size': sections[4],
                'reasoning': sections[5]
            }
            return result
            
        except Exception as e:
            logger.error(f"Error parsing signals: {str(e)}")
            raise
    
    def _parse_optimization(self, optimization: str) -> Dict:
        """解析优化结果"""
        try:
            # 这里需要根据实际的AI输出格式进行解析
            # 示例实现
            sections = optimization.split('\n\n')
            result = {
                'weights': sections[0],
                'expected_return': sections[1],
                'risk_metrics': sections[2],
                'rebalancing': sections[3],
                'risk_control': sections[4]
            }
            return result
            
        except Exception as e:
            logger.error(f"Error parsing optimization: {str(e)}")
            raise
    
    def _parse_risk_analysis(self, risk_analysis: str) -> Dict:
        """解析风险分析"""
        try:
            # 这里需要根据实际的AI输出格式进行解析
            # 示例实现
            sections = risk_analysis.split('\n\n')
            result = {
                'exposure_analysis': sections[0],
                'stress_test': sections[1],
                'risk_indicators': sections[2],
                'control_measures': sections[3],
                'emergency_plan': sections[4]
            }
            return result
            
        except Exception as e:
            logger.error(f"Error parsing risk analysis: {str(e)}")
            raise
    
    def _parse_report(self, report: str) -> Dict:
        """解析绩效报告"""
        try:
            # 这里需要根据实际的AI输出格式进行解析
            # 示例实现
            sections = report.split('\n\n')
            result = {
                'return_analysis': sections[0],
                'risk_analysis': sections[1],
                'strategy_evaluation': sections[2],
                'improvement_suggestions': sections[3],
                'future_outlook': sections[4]
            }
            return result
            
        except Exception as e:
            logger.error(f"Error parsing report: {str(e)}")
            raise
    
    def _prepare_performance_data(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> Dict:
        """准备绩效数据"""
        try:
            # 筛选时间范围内的数据
            mask = (self.data.index >= start_date) & (self.data.index <= end_date)
            period_data = self.data[mask]
            
            # 计算绩效指标
            returns = period_data['close'].pct_change()
            cumulative_returns = (1 + returns).cumprod()
            
            performance_data = {
                'total_return': cumulative_returns.iloc[-1] - 1,
                'annual_return': (1 + returns.mean()) ** 252 - 1,
                'annual_volatility': returns.std() * np.sqrt(252),
                'sharpe_ratio': (returns.mean() * 252) / (returns.std() * np.sqrt(252)),
                'max_drawdown': (cumulative_returns / cumulative_returns.cummax() - 1).min(),
                'win_rate': (returns > 0).mean(),
                'profit_factor': abs(returns[returns > 0].sum() / returns[returns < 0].sum())
            }
            
            return performance_data
            
        except Exception as e:
            logger.error(f"Error preparing performance data: {str(e)}")
            raise
    
    def get_signal_history(self) -> pd.DataFrame:
        """获取信号历史"""
        return pd.DataFrame(self.signals).T
    
    def get_performance_history(self) -> pd.DataFrame:
        """获取绩效历史"""
        return pd.DataFrame(self.performance_metrics).T 