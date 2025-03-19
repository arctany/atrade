import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
from auth import Auth
from database import Database
from ibkr_handler import IBKRHandler
from market_analyzer import MarketAnalyzer
from strategy_analyzer import StrategyAnalyzer
from risk_manager import RiskManager
from monitor import Monitor
from config import TECHNICAL_INDICATORS, MARKET_INDICES, INDUSTRY_ETF_MAPPING

logger = logging.getLogger(__name__)

class UI:
    def __init__(self):
        self.db = Database()
        self.auth = Auth(self.db)
        self.ibkr = IBKRHandler()
        self.market_analyzer = MarketAnalyzer()
        self.strategy_analyzer = StrategyAnalyzer()
        self.risk_manager = RiskManager()
        self.monitor = Monitor()
        self.setup_page()

    def setup_page(self):
        """设置页面配置"""
        st.set_page_config(
            page_title="Trading Platform",
            page_icon="📈",
            layout="wide",
            initial_sidebar_state="expanded"
        )

    def run(self):
        """运行用户界面"""
        try:
            # 检查用户登录状态
            if 'user' not in st.session_state:
                self.show_login_page()
            else:
                self.show_main_page()
        except Exception as e:
            logger.error(f"Error running UI: {str(e)}")
            st.error("An error occurred. Please try again later.")

    def show_login_page(self):
        """显示登录页面"""
        st.title("Trading Platform Login")
        
        # 创建登录表单
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit_button = st.form_submit_button("Login")
            
            if submit_button:
                result = self.auth.login(username, password)
                if result:
                    st.session_state.user = result['user']
                    st.session_state.token = result['token']
                    st.success("Login successful!")
                    st.experimental_rerun()
                else:
                    st.error("Invalid username or password")

        # 注册链接
        if st.button("Register New Account"):
            st.session_state.show_register = True
            st.experimental_rerun()

        # 显示注册表单
        if st.session_state.get('show_register', False):
            self.show_register_form()

    def show_register_form(self):
        """显示注册表单"""
        st.title("Register New Account")
        
        with st.form("register_form"):
            new_username = st.text_input("Username")
            new_password = st.text_input("Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            email = st.text_input("Email")
            submit_button = st.form_submit_button("Register")
            
            if submit_button:
                if new_password != confirm_password:
                    st.error("Passwords do not match")
                else:
                    if self.auth.register_user(new_username, new_password, email):
                        st.success("Registration successful! Please login.")
                        st.session_state.show_register = False
                        st.experimental_rerun()
                    else:
                        st.error("Registration failed. Username may already exist.")

    def show_main_page(self):
        """显示主页面"""
        # 侧边栏
        with st.sidebar:
            st.title(f"Welcome, {st.session_state.user['username']}")
            if st.button("Logout"):
                del st.session_state.user
                del st.session_state.token
                st.experimental_rerun()

            # 导航菜单
            page = st.radio(
                "Navigation",
                ["Dashboard", "Market Analysis", "Trading", "Strategy", "Risk Management", "Alerts"]
            )

        # 主页面内容
        if page == "Dashboard":
            self.show_dashboard()
        elif page == "Market Analysis":
            self.show_market_analysis()
        elif page == "Trading":
            self.show_trading()
        elif page == "Strategy":
            self.show_strategy()
        elif page == "Risk Management":
            self.show_risk_management()
        elif page == "Alerts":
            self.show_alerts()

    def show_dashboard(self):
        """显示仪表板"""
        st.title("Trading Dashboard")
        
        # 创建三列布局
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # 账户概览
            st.subheader("Account Overview")
            account_summary = self.ibkr.get_account_summary()
            if account_summary is not None:
                st.metric("Total Value", f"${account_summary['NetLiquidation']:,.2f}")
                st.metric("Available Funds", f"${account_summary['AvailableFunds']:,.2f}")
                st.metric("Buying Power", f"${account_summary['BuyingPower']:,.2f}")
        
        with col2:
            # 持仓概览
            st.subheader("Positions Overview")
            positions = self.db.get_positions(st.session_state.user['id'])
            if positions:
                total_value = sum(pos['quantity'] * pos['current_price'] for pos in positions)
                st.metric("Total Positions Value", f"${total_value:,.2f}")
                st.metric("Number of Positions", len(positions))
            else:
                st.info("No open positions")
        
        with col3:
            # 今日盈亏
            st.subheader("Today's P&L")
            today = datetime.now().date()
            trades = self.db.get_trades(st.session_state.user['id'])
            if trades:
                today_trades = [t for t in trades if t['timestamp'].date() == today]
                today_pnl = sum(t['realized_pnl'] for t in today_trades)
                st.metric("Today's P&L", f"${today_pnl:,.2f}")
            else:
                st.info("No trades today")

        # 图表区域
        st.subheader("Performance Chart")
        self.plot_performance_chart()

    def show_market_analysis(self):
        """显示市场分析页面"""
        st.title("Market Analysis")
        
        # 市场选择
        market = st.selectbox(
            "Select Market",
            list(MARKET_INDICES.keys())
        )
        
        # 时间范围选择
        time_range = st.selectbox(
            "Time Range",
            ["1D", "1W", "1M", "3M", "6M", "1Y", "5Y"]
        )
        
        # 获取市场数据
        data = self.market_analyzer.get_market_data(
            MARKET_INDICES[market],
            time_range
        )
        
        if data is not None:
            # 绘制价格图表
            fig = go.Figure(data=[go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close']
            )])
            st.plotly_chart(fig, use_container_width=True)
            
            # 显示技术指标
            st.subheader("Technical Indicators")
            indicators = self.market_analyzer.calculate_indicators(data)
            for name, value in indicators.items():
                st.metric(name, f"{value:.2f}")

    def show_trading(self):
        """显示交易页面"""
        st.title("Trading")
        
        # 交易表单
        with st.form("trade_form"):
            symbol = st.text_input("Symbol")
            quantity = st.number_input("Quantity", min_value=1)
            order_type = st.selectbox("Order Type", ["Market", "Limit"])
            if order_type == "Limit":
                limit_price = st.number_input("Limit Price", min_value=0.0)
            
            submit_button = st.form_submit_button("Place Order")
            
            if submit_button:
                # 执行交易
                order = {
                    'symbol': symbol,
                    'quantity': quantity,
                    'type': order_type,
                    'price': limit_price if order_type == "Limit" else None
                }
                if self.ibkr.place_order(order):
                    st.success("Order placed successfully!")
                else:
                    st.error("Failed to place order")

        # 显示当前持仓
        st.subheader("Current Positions")
        positions = self.db.get_positions(st.session_state.user['id'])
        if positions:
            positions_df = pd.DataFrame(positions)
            st.dataframe(positions_df)
        else:
            st.info("No open positions")

    def show_strategy(self):
        """显示策略页面"""
        st.title("Trading Strategy")
        
        # 策略选择
        strategy = st.selectbox(
            "Select Strategy",
            ["Moving Average Cross", "RSI Momentum", "MACD", "Bollinger Bands"]
        )
        
        # 策略参数
        st.subheader("Strategy Parameters")
        if strategy == "Moving Average Cross":
            short_window = st.number_input("Short Window", value=TECHNICAL_INDICATORS['MA']['short'])
            long_window = st.number_input("Long Window", value=TECHNICAL_INDICATORS['MA']['long'])
        elif strategy == "RSI Momentum":
            period = st.number_input("RSI Period", value=TECHNICAL_INDICATORS['RSI']['period'])
            overbought = st.number_input("Overbought Level", value=TECHNICAL_INDICATORS['RSI']['overbought'])
            oversold = st.number_input("Oversold Level", value=TECHNICAL_INDICATORS['RSI']['oversold'])
        elif strategy == "MACD":
            fast = st.number_input("Fast Period", value=TECHNICAL_INDICATORS['MACD']['fast'])
            slow = st.number_input("Slow Period", value=TECHNICAL_INDICATORS['MACD']['slow'])
            signal = st.number_input("Signal Period", value=TECHNICAL_INDICATORS['MACD']['signal'])
        elif strategy == "Bollinger Bands":
            period = st.number_input("BB Period", value=TECHNICAL_INDICATORS['BB']['period'])
            std_dev = st.number_input("Standard Deviation", value=TECHNICAL_INDICATORS['BB']['std_dev'])

        # 回测按钮
        if st.button("Run Backtest"):
            self.run_backtest(strategy)

    def show_risk_management(self):
        """显示风险管理页面"""
        st.title("Risk Management")
        
        # 风险指标
        st.subheader("Portfolio Risk Metrics")
        risk_report = self.risk_manager.get_risk_report()
        
        if risk_report:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Portfolio Volatility", f"{risk_report['risk_metrics']['portfolio_volatility']:.2%}")
                st.metric("Portfolio Beta", f"{risk_report['risk_metrics']['portfolio_beta']:.2f}")
            
            with col2:
                st.metric("Max Drawdown", f"{risk_report['risk_metrics']['max_drawdown']:.2%}")
                st.metric("Total Value", f"${risk_report['risk_metrics']['total_value']:,.2f}")
            
            with col3:
                st.metric("Position Count", len(risk_report['positions']))
                if risk_report['violations']:
                    st.warning(f"{len(risk_report['violations'])} Risk Violations")
        
        # 风险限制设置
        st.subheader("Risk Limits")
        with st.form("risk_limits"):
            max_position_size = st.number_input(
                "Max Position Size (%)",
                value=float(self.risk_manager.max_position_size * 100)
            )
            max_drawdown = st.number_input(
                "Max Drawdown (%)",
                value=float(self.risk_manager.max_drawdown_limit * 100)
            )
            volatility_limit = st.number_input(
                "Volatility Limit (%)",
                value=float(self.risk_manager.volatility_limit * 100)
            )
            beta_limit = st.number_input(
                "Beta Limit",
                value=float(self.risk_manager.beta_limit)
            )
            
            if st.form_submit_button("Update Limits"):
                self.risk_manager.max_position_size = max_position_size / 100
                self.risk_manager.max_drawdown_limit = max_drawdown / 100
                self.risk_manager.volatility_limit = volatility_limit / 100
                self.risk_manager.beta_limit = beta_limit
                st.success("Risk limits updated!")

    def show_alerts(self):
        """显示告警页面"""
        st.title("System Alerts")
        
        # 告警摘要
        alert_summary = self.monitor.get_alert_summary()
        if alert_summary:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Alerts", alert_summary['total_alerts'])
                st.metric("High Severity", alert_summary['alert_severities'].get('HIGH', 0))
            
            with col2:
                st.metric("Medium Severity", alert_summary['alert_severities'].get('MEDIUM', 0))
                st.metric("Low Severity", alert_summary['alert_severities'].get('LOW', 0))
            
            with col3:
                st.metric("Price Alerts", alert_summary['alert_types'].get('PRICE_ALERT', 0))
                st.metric("Volume Alerts", alert_summary['alert_types'].get('VOLUME_ALERT', 0))
        
        # 最近告警
        st.subheader("Recent Alerts")
        if alert_summary and alert_summary['recent_alerts']:
            for alert in alert_summary['recent_alerts']:
                with st.expander(f"{alert['type']} - {alert['timestamp']}"):
                    st.write(f"Message: {alert['message']}")
                    st.write(f"Severity: {alert['severity']}")
                    if 'symbol' in alert:
                        st.write(f"Symbol: {alert['symbol']}")

    def plot_performance_chart(self):
        """绘制性能图表"""
        try:
            # 获取交易记录
            trades = self.db.get_trades(st.session_state.user['id'])
            if not trades:
                st.info("No trading history available")
                return

            # 创建DataFrame
            df = pd.DataFrame(trades)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')

            # 计算累计收益
            df['cumulative_pnl'] = df['realized_pnl'].cumsum()

            # 创建图表
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['cumulative_pnl'],
                mode='lines',
                name='Cumulative P&L'
            ))

            # 更新布局
            fig.update_layout(
                title='Cumulative P&L Over Time',
                xaxis_title='Date',
                yaxis_title='Cumulative P&L ($)',
                hovermode='x unified'
            )

            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            logger.error(f"Error plotting performance chart: {str(e)}")
            st.error("Failed to generate performance chart")

    def run_backtest(self, strategy: str):
        """运行回测"""
        try:
            # 获取回测参数
            symbol = st.text_input("Symbol for Backtest")
            start_date = st.date_input("Start Date")
            end_date = st.date_input("End Date")
            
            if st.button("Run"):
                # 获取历史数据
                data = self.market_analyzer.get_historical_data(
                    symbol,
                    start_date,
                    end_date
                )
                
                if data is not None:
                    # 运行回测
                    results = self.strategy_analyzer.run_backtest(
                        strategy,
                        data
                    )
                    
                    # 显示结果
                    st.subheader("Backtest Results")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total Return", f"{results['total_return']:.2%}")
                        st.metric("Win Rate", f"{results['win_rate']:.2%}")
                    
                    with col2:
                        st.metric("Sharpe Ratio", f"{results['sharpe_ratio']:.2f}")
                        st.metric("Max Drawdown", f"{results['max_drawdown']:.2%}")
                    
                    with col3:
                        st.metric("Total Trades", results['total_trades'])
                        st.metric("Profit Factor", f"{results['profit_factor']:.2f}")
                    
                    # 绘制权益曲线
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=results['equity_curve'].index,
                        y=results['equity_curve'].values,
                        mode='lines',
                        name='Equity Curve'
                    ))
                    st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            logger.error(f"Error running backtest: {str(e)}")
            st.error("Failed to run backtest") 