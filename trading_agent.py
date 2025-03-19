import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import os
from config import TRADING_STRATEGY, RISK_MANAGEMENT_CONFIG, MONITOR_CONFIG
from risk_manager import RiskManager
from backtest import Backtest

class TradingAgent:
    """交易助手代理，用于实时对话和策略建议"""
    
    def __init__(self, data: pd.DataFrame, positions: Dict, capital: float):
        """
        初始化交易助手代理
        
        Args:
            data: 市场数据
            positions: 当前持仓
            capital: 当前资金
        """
        self.data = data.copy()
        self.positions = positions
        self.capital = capital
        self.risk_manager = RiskManager(data)
        self.backtest = Backtest(data)
        self.conversation_history: List[Dict] = []  # 对话历史
        self.strategy_suggestions: List[Dict] = []  # 策略建议历史
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        
    def process_message(self, message: str) -> Dict:
        """
        处理用户消息
        
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
            
            # 根据意图生成响应
            response = self._generate_response(intent, message)
            
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
    
    def _analyze_intent(self, message: str) -> str:
        """
        分析消息意图
        
        Args:
            message: 用户消息
            
        Returns:
            str: 意图类型
        """
        try:
            # 定义意图关键词
            intents = {
                'market_analysis': ['市场', '行情', '趋势', '走势', '分析'],
                'position_analysis': ['持仓', '仓位', '头寸', '分析'],
                'risk_analysis': ['风险', '回撤', '波动', '分析'],
                'strategy_suggestion': ['策略', '建议', '推荐', '操作'],
                'performance_analysis': ['表现', '收益', '绩效', '分析'],
                'general_question': ['什么', '如何', '为什么', '是否']
            }
            
            # 分析消息中的关键词
            for intent, keywords in intents.items():
                if any(keyword in message for keyword in keywords):
                    return intent
            
            return 'general_question'
            
        except Exception as e:
            self.logger.error(f"分析意图时发生错误: {str(e)}")
            return 'general_question'
    
    def _generate_response(self, intent: str, message: str) -> Dict:
        """
        生成响应
        
        Args:
            intent: 意图类型
            message: 用户消息
            
        Returns:
            Dict: 响应内容
        """
        try:
            if intent == 'market_analysis':
                return self._generate_market_analysis()
            elif intent == 'position_analysis':
                return self._generate_position_analysis()
            elif intent == 'risk_analysis':
                return self._generate_risk_analysis()
            elif intent == 'strategy_suggestion':
                return self._generate_strategy_suggestion()
            elif intent == 'performance_analysis':
                return self._generate_performance_analysis()
            else:
                return self._generate_general_response(message)
                
        except Exception as e:
            self.logger.error(f"生成响应时发生错误: {str(e)}")
            return {
                'message': "抱歉，生成响应时出现错误。请稍后重试。",
                'type': 'error'
            }
    
    def _generate_market_analysis(self) -> Dict:
        """生成市场分析"""
        try:
            # 计算技术指标
            ma20 = self.data['close'].rolling(window=20).mean()
            ma50 = self.data['close'].rolling(window=50).mean()
            rsi = self._calculate_rsi()
            
            # 分析市场趋势
            current_price = self.data['close'].iloc[-1]
            trend = "上升" if current_price > ma20.iloc[-1] > ma50.iloc[-1] else \
                   "下降" if current_price < ma20.iloc[-1] < ma50.iloc[-1] else "震荡"
            
            # 分析RSI
            rsi_status = "超买" if rsi > 70 else "超卖" if rsi < 30 else "中性"
            
            # 生成分析报告
            analysis = {
                'message': f"当前市场分析：\n"
                          f"1. 市场趋势：{trend}\n"
                          f"2. RSI状态：{rsi_status} (RSI: {rsi:.2f})\n"
                          f"3. 20日均线：{ma20.iloc[-1]:.2f}\n"
                          f"4. 50日均线：{ma50.iloc[-1]:.2f}\n"
                          f"5. 当前价格：{current_price:.2f}",
                'type': 'market_analysis',
                'data': {
                    'trend': trend,
                    'rsi': rsi,
                    'ma20': ma20.iloc[-1],
                    'ma50': ma50.iloc[-1],
                    'current_price': current_price
                }
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"生成市场分析时发生错误: {str(e)}")
            return {
                'message': "抱歉，生成市场分析时出现错误。",
                'type': 'error'
            }
    
    def _generate_position_analysis(self) -> Dict:
        """生成持仓分析"""
        try:
            if not self.positions:
                return {
                    'message': "当前没有持仓。",
                    'type': 'position_analysis'
                }
            
            # 计算持仓信息
            position_analysis = []
            total_value = 0
            
            for symbol, position in self.positions.items():
                current_price = self.data['close'].iloc[-1]
                position_value = position['quantity'] * current_price
                total_value += position_value
                
                # 计算盈亏
                profit = (current_price - position['entry_price']) * position['quantity']
                profit_pct = profit / (position['entry_price'] * position['quantity'])
                
                position_analysis.append({
                    'symbol': symbol,
                    'quantity': position['quantity'],
                    'entry_price': position['entry_price'],
                    'current_price': current_price,
                    'value': position_value,
                    'profit': profit,
                    'profit_pct': profit_pct
                })
            
            # 生成分析报告
            message = "当前持仓分析：\n"
            for pos in position_analysis:
                message += f"\n{pos['symbol']}:\n"
                message += f"持仓数量: {pos['quantity']:.2f}\n"
                message += f"持仓价值: ${pos['value']:.2f}\n"
                message += f"持仓盈亏: ${pos['profit']:.2f} ({pos['profit_pct']:.2%})\n"
            
            message += f"\n总持仓价值: ${total_value:.2f}"
            
            return {
                'message': message,
                'type': 'position_analysis',
                'data': {
                    'positions': position_analysis,
                    'total_value': total_value
                }
            }
            
        except Exception as e:
            self.logger.error(f"生成持仓分析时发生错误: {str(e)}")
            return {
                'message': "抱歉，生成持仓分析时出现错误。",
                'type': 'error'
            }
    
    def _generate_risk_analysis(self) -> Dict:
        """生成风险分析"""
        try:
            # 获取风险指标
            risk_metrics = self.risk_manager.calculate_portfolio_risk()
            
            # 生成风险报告
            message = "当前风险分析：\n"
            message += f"1. 投资组合波动率: {risk_metrics['volatility']:.2%}\n"
            message += f"2. 投资组合Beta: {risk_metrics['beta']:.2f}\n"
            message += f"3. VaR (95%): {risk_metrics['var']:.2%}\n"
            message += f"4. 夏普比率: {risk_metrics['sharpe_ratio']:.2f}\n"
            message += f"5. 持仓数量: {risk_metrics['positions']}\n"
            
            # 添加风险建议
            if risk_metrics['volatility'] > RISK_MANAGEMENT_CONFIG['max_volatility']:
                message += "\n风险提示：当前波动率较高，建议降低仓位或增加对冲。"
            if risk_metrics['beta'] > RISK_MANAGEMENT_CONFIG['max_systematic_risk']:
                message += "\n风险提示：当前Beta值较高，建议增加防御性资产。"
            
            return {
                'message': message,
                'type': 'risk_analysis',
                'data': risk_metrics
            }
            
        except Exception as e:
            self.logger.error(f"生成风险分析时发生错误: {str(e)}")
            return {
                'message': "抱歉，生成风险分析时出现错误。",
                'type': 'error'
            }
    
    def _generate_strategy_suggestion(self) -> Dict:
        """生成策略建议"""
        try:
            # 获取市场分析
            market_analysis = self._generate_market_analysis()
            
            # 获取风险分析
            risk_analysis = self._generate_risk_analysis()
            
            # 获取持仓分析
            position_analysis = self._generate_position_analysis()
            
            # 生成策略建议
            suggestions = []
            
            # 基于市场趋势的建议
            if market_analysis['data']['trend'] == "上升":
                suggestions.append("考虑增加多头仓位")
            elif market_analysis['data']['trend'] == "下降":
                suggestions.append("考虑增加空头仓位或减仓")
            
            # 基于RSI的建议
            if market_analysis['data']['rsi'] > 70:
                suggestions.append("RSI超买，考虑减仓或等待回调")
            elif market_analysis['data']['rsi'] < 30:
                suggestions.append("RSI超卖，考虑逢低买入")
            
            # 基于风险的建议
            if risk_analysis['data']['volatility'] > RISK_MANAGEMENT_CONFIG['max_volatility']:
                suggestions.append("波动率较高，建议降低仓位或增加对冲")
            if risk_analysis['data']['beta'] > RISK_MANAGEMENT_CONFIG['max_systematic_risk']:
                suggestions.append("系统性风险较高，建议增加防御性资产")
            
            # 生成建议报告
            message = "策略建议：\n"
            for i, suggestion in enumerate(suggestions, 1):
                message += f"{i}. {suggestion}\n"
            
            # 记录策略建议
            self.strategy_suggestions.append({
                'timestamp': datetime.now().isoformat(),
                'suggestions': suggestions,
                'market_analysis': market_analysis['data'],
                'risk_analysis': risk_analysis['data'],
                'position_analysis': position_analysis['data']
            })
            
            return {
                'message': message,
                'type': 'strategy_suggestion',
                'data': {
                    'suggestions': suggestions,
                    'market_analysis': market_analysis['data'],
                    'risk_analysis': risk_analysis['data'],
                    'position_analysis': position_analysis['data']
                }
            }
            
        except Exception as e:
            self.logger.error(f"生成策略建议时发生错误: {str(e)}")
            return {
                'message': "抱歉，生成策略建议时出现错误。",
                'type': 'error'
            }
    
    def _generate_performance_analysis(self) -> Dict:
        """生成绩效分析"""
        try:
            # 计算绩效指标
            returns = self.data['close'].pct_change().dropna()
            total_return = (self.data['close'].iloc[-1] / self.data['close'].iloc[0] - 1)
            annual_return = (1 + total_return) ** (252 / len(returns)) - 1
            volatility = returns.std() * np.sqrt(252)
            sharpe_ratio = (annual_return - RISK_MANAGEMENT_CONFIG['risk_free_rate']) / volatility
            
            # 计算最大回撤
            cumulative_returns = (1 + returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdowns = cumulative_returns / rolling_max - 1
            max_drawdown = drawdowns.min()
            
            # 生成分析报告
            message = "绩效分析：\n"
            message += f"1. 总收益率: {total_return:.2%}\n"
            message += f"2. 年化收益率: {annual_return:.2%}\n"
            message += f"3. 年化波动率: {volatility:.2%}\n"
            message += f"4. 夏普比率: {sharpe_ratio:.2f}\n"
            message += f"5. 最大回撤: {max_drawdown:.2%}\n"
            
            return {
                'message': message,
                'type': 'performance_analysis',
                'data': {
                    'total_return': total_return,
                    'annual_return': annual_return,
                    'volatility': volatility,
                    'sharpe_ratio': sharpe_ratio,
                    'max_drawdown': max_drawdown
                }
            }
            
        except Exception as e:
            self.logger.error(f"生成绩效分析时发生错误: {str(e)}")
            return {
                'message': "抱歉，生成绩效分析时出现错误。",
                'type': 'error'
            }
    
    def _generate_general_response(self, message: str) -> Dict:
        """
        生成一般性响应
        
        Args:
            message: 用户消息
            
        Returns:
            Dict: 响应内容
        """
        try:
            # 简单的问答匹配
            if "你好" in message or "在吗" in message:
                return {
                    'message': "你好！我是你的交易助手。我可以帮你分析市场、评估风险、提供策略建议等。有什么我可以帮你的吗？",
                    'type': 'greeting'
                }
            elif "谢谢" in message:
                return {
                    'message': "不客气！如果还有其他问题，随时问我。",
                    'type': 'thanks'
                }
            elif "再见" in message or "拜拜" in message:
                return {
                    'message': "再见！祝您交易顺利！",
                    'type': 'goodbye'
                }
            else:
                return {
                    'message': "抱歉，我可能没有完全理解您的问题。您可以尝试询问市场分析、持仓分析、风险分析或策略建议等具体问题。",
                    'type': 'unknown'
                }
                
        except Exception as e:
            self.logger.error(f"生成一般性响应时发生错误: {str(e)}")
            return {
                'message': "抱歉，处理您的消息时出现错误。请稍后重试。",
                'type': 'error'
            }
    
    def _calculate_rsi(self, period: int = 14) -> float:
        """
        计算RSI
        
        Args:
            period: RSI周期
            
        Returns:
            float: RSI值
        """
        try:
            delta = self.data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs.iloc[-1]))
            
        except Exception as e:
            self.logger.error(f"计算RSI时发生错误: {str(e)}")
            return 50
    
    def get_conversation_history(self) -> List[Dict]:
        """
        获取对话历史
        
        Returns:
            List[Dict]: 对话历史记录
        """
        return self.conversation_history
    
    def get_strategy_suggestions(self) -> List[Dict]:
        """
        获取策略建议历史
        
        Returns:
            List[Dict]: 策略建议历史记录
        """
        return self.strategy_suggestions 