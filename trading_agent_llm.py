import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import os
import json
import requests
from config import TRADING_STRATEGY, RISK_MANAGEMENT_CONFIG, MONITOR_CONFIG
from risk_manager import RiskManager
from backtest import Backtest
from trading_agent import TradingAgent

class TradingAgentLLM(TradingAgent):
    """增强版交易助手代理，集成大模型和数据增强功能"""
    
    def __init__(self, data: pd.DataFrame, positions: Dict, capital: float):
        """
        初始化增强版交易助手代理
        
        Args:
            data: 市场数据
            positions: 当前持仓
            capital: 当前资金
        """
        super().__init__(data, positions, capital)
        
        # 初始化大模型配置
        self.llm_config = {
            'api_key': os.getenv('LLM_API_KEY'),
            'model': os.getenv('LLM_MODEL', 'gpt-4'),
            'temperature': float(os.getenv('LLM_TEMPERATURE', '0.7')),
            'max_tokens': int(os.getenv('LLM_MAX_TOKENS', '1000'))
        }
        
        # 初始化数据源配置
        self.data_sources = {
            'market_data': os.getenv('MARKET_DATA_API'),
            'news_api': os.getenv('NEWS_API_KEY'),
            'sentiment_api': os.getenv('SENTIMENT_API_KEY')
        }
        
        # 缓存配置
        self.cache_config = {
            'market_data_cache': {},
            'news_cache': {},
            'sentiment_cache:': {}
        }
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
    
    def process_message(self, message: str) -> Dict:
        """
        处理用户消息，集成大模型分析
        
        Args:
            message: 用户消息
            
        Returns:
            Dict: 代理响应
        """
        try:
            # 记录用户消息
            self.conversation_history.append({
                'role': 'user',
                'content': message,
                'timestamp': datetime.now().isoformat()
            })
            
            # 分析消息意图
            intent = self._analyze_intent(message)
            
            # 获取增强数据
            enhanced_data = self._get_enhanced_data(intent)
            
            # 根据意图生成响应
            response = self._generate_enhanced_response(intent, message, enhanced_data)
            
            # 记录代理响应
            self.conversation_history.append({
                'role': 'assistant',
                'content': response['message'],
                'timestamp': datetime.now().isoformat()
            })
            
            return response
            
        except Exception as e:
            self.logger.error(f"处理消息时发生错误: {str(e)}")
            return {
                'message': "抱歉，处理您的消息时出现错误。请稍后重试。",
                'type': 'error'
            }
    
    def _get_enhanced_data(self, intent: str) -> Dict:
        """
        获取增强数据
        
        Args:
            intent: 意图类型
            
        Returns:
            Dict: 增强数据
        """
        try:
            enhanced_data = {}
            
            # 获取市场数据
            if intent in ['market_analysis', 'strategy_suggestion']:
                enhanced_data['market_data'] = self._get_market_data()
            
            # 获取新闻数据
            if intent in ['market_analysis', 'risk_analysis']:
                enhanced_data['news'] = self._get_news_data()
            
            # 获取情绪数据
            if intent in ['market_analysis', 'strategy_suggestion']:
                enhanced_data['sentiment'] = self._get_sentiment_data()
            
            return enhanced_data
            
        except Exception as e:
            self.logger.error(f"获取增强数据时发生错误: {str(e)}")
            return {}
    
    def _get_market_data(self) -> Dict:
        """获取市场数据"""
        try:
            # 检查缓存
            cache_key = datetime.now().strftime('%Y%m%d')
            if cache_key in self.cache_config['market_data_cache']:
                return self.cache_config['market_data_cache'][cache_key]
            
            # 获取实时市场数据
            response = requests.get(
                self.data_sources['market_data'],
                params={
                    'symbol': self.data.index[-1],
                    'interval': '1d'
                }
            )
            
            market_data = response.json()
            
            # 更新缓存
            self.cache_config['market_data_cache'][cache_key] = market_data
            
            return market_data
            
        except Exception as e:
            self.logger.error(f"获取市场数据时发生错误: {str(e)}")
            return {}
    
    def _get_news_data(self) -> List[Dict]:
        """获取新闻数据"""
        try:
            # 检查缓存
            cache_key = datetime.now().strftime('%Y%m%d')
            if cache_key in self.cache_config['news_cache']:
                return self.cache_config['news_cache'][cache_key]
            
            # 获取相关新闻
            response = requests.get(
                self.data_sources['news_api'],
                params={
                    'q': self.data.index[-1],
                    'from': (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
                }
            )
            
            news_data = response.json()
            
            # 更新缓存
            self.cache_config['news_cache'][cache_key] = news_data
            
            return news_data
            
        except Exception as e:
            self.logger.error(f"获取新闻数据时发生错误: {str(e)}")
            return []
    
    def _get_sentiment_data(self) -> Dict:
        """获取情绪数据"""
        try:
            # 检查缓存
            cache_key = datetime.now().strftime('%Y%m%d')
            if cache_key in self.cache_config['sentiment_cache']:
                return self.cache_config['sentiment_cache'][cache_key]
            
            # 获取市场情绪数据
            response = requests.get(
                self.data_sources['sentiment_api'],
                params={
                    'symbol': self.data.index[-1]
                }
            )
            
            sentiment_data = response.json()
            
            # 更新缓存
            self.cache_config['sentiment_cache'][cache_key] = sentiment_data
            
            return sentiment_data
            
        except Exception as e:
            self.logger.error(f"获取情绪数据时发生错误: {str(e)}")
            return {}
    
    def _generate_enhanced_response(self, intent: str, message: str, enhanced_data: Dict) -> Dict:
        """
        生成增强版响应
        
        Args:
            intent: 意图类型
            message: 用户消息
            enhanced_data: 增强数据
            
        Returns:
            Dict: 响应内容
        """
        try:
            # 构建提示词
            prompt = self._build_prompt(intent, message, enhanced_data)
            
            # 调用大模型
            llm_response = self._call_llm(prompt)
            
            # 解析大模型响应
            response = self._parse_llm_response(llm_response, intent)
            
            return response
            
        except Exception as e:
            self.logger.error(f"生成增强响应时发生错误: {str(e)}")
            return {
                'message': "抱歉，生成响应时出现错误。请稍后重试。",
                'type': 'error'
            }
    
    def _build_prompt(self, intent: str, message: str, enhanced_data: Dict) -> str:
        """
        构建提示词
        
        Args:
            intent: 意图类型
            message: 用户消息
            enhanced_data: 增强数据
            
        Returns:
            str: 提示词
        """
        try:
            # 基础提示词
            prompt = f"你是一个专业的交易助手。用户的问题是：{message}\n\n"
            
            # 添加市场数据
            if 'market_data' in enhanced_data:
                prompt += f"当前市场数据：\n{json.dumps(enhanced_data['market_data'], indent=2)}\n\n"
            
            # 添加新闻数据
            if 'news' in enhanced_data:
                prompt += f"相关新闻：\n{json.dumps(enhanced_data['news'], indent=2)}\n\n"
            
            # 添加情绪数据
            if 'sentiment' in enhanced_data:
                prompt += f"市场情绪：\n{json.dumps(enhanced_data['sentiment'], indent=2)}\n\n"
            
            # 添加意图特定提示
            if intent == 'market_analysis':
                prompt += "请基于以上数据，提供详细的市场分析。"
            elif intent == 'strategy_suggestion':
                prompt += "请基于以上数据，提供具体的交易策略建议。"
            elif intent == 'risk_analysis':
                prompt += "请基于以上数据，提供详细的风险分析。"
            else:
                prompt += "请基于以上数据，回答用户的问题。"
            
            return prompt
            
        except Exception as e:
            self.logger.error(f"构建提示词时发生错误: {str(e)}")
            return message
    
    def _call_llm(self, prompt: str) -> str:
        """
        调用大模型
        
        Args:
            prompt: 提示词
            
        Returns:
            str: 大模型响应
        """
        try:
            # 调用OpenAI API
            response = requests.post(
                'https://api.openai.com/v1/chat/completions',
                headers={
                    'Authorization': f"Bearer {self.llm_config['api_key']}",
                    'Content-Type': 'application/json'
                },
                json={
                    'model': self.llm_config['model'],
                    'messages': [{'role': 'user', 'content': prompt}],
                    'temperature': self.llm_config['temperature'],
                    'max_tokens': self.llm_config['max_tokens']
                }
            )
            
            return response.json()['choices'][0]['message']['content']
            
        except Exception as e:
            self.logger.error(f"调用大模型时发生错误: {str(e)}")
            return ""
    
    def _parse_llm_response(self, llm_response: str, intent: str) -> Dict:
        """
        解析大模型响应
        
        Args:
            llm_response: 大模型响应
            intent: 意图类型
            
        Returns:
            Dict: 解析后的响应
        """
        try:
            # 基础响应
            response = {
                'message': llm_response,
                'type': intent
            }
            
            # 添加数据
            if intent == 'market_analysis':
                response['data'] = {
                    'analysis': llm_response,
                    'timestamp': datetime.now().isoformat()
                }
            elif intent == 'strategy_suggestion':
                response['data'] = {
                    'suggestions': llm_response.split('\n'),
                    'timestamp': datetime.now().isoformat()
                }
            elif intent == 'risk_analysis':
                response['data'] = {
                    'analysis': llm_response,
                    'timestamp': datetime.now().isoformat()
                }
            
            return response
            
        except Exception as e:
            self.logger.error(f"解析大模型响应时发生错误: {str(e)}")
            return {
                'message': llm_response,
                'type': intent
            }
    
    def get_enhanced_data(self) -> Dict:
        """
        获取所有增强数据
        
        Returns:
            Dict: 增强数据
        """
        return {
            'market_data': self._get_market_data(),
            'news': self._get_news_data(),
            'sentiment': self._get_sentiment_data()
        } 