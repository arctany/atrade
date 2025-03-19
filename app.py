import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import yfinance as yf
import numpy as np
from ib_insync import *
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 设置页面配置
st.set_page_config(
    page_title="股票交易分析平台",
    page_icon="📈",
    layout="wide"
)

# 初始化IB连接
def init_ib_connection():
    ib = IB()
    try:
        ib.connect('127.0.0.1', 7497, clientId=0)
        return ib
    except Exception as e:
        st.error(f"连接IB失败: {str(e)}")
        return None

# 获取股票数据
def get_stock_data(symbol, period="1y"):
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(period=period)
        return df
    except Exception as e:
        st.error(f"获取股票数据失败: {str(e)}")
        return None

# 计算技术指标
def calculate_technical_indicators(df):
    # 计算移动平均线
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    
    # 计算RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    return df

# 主页面
def main():
    st.title("📈 股票交易分析平台")
    
    # 侧边栏
    st.sidebar.title("功能选择")
    page = st.sidebar.radio(
        "选择功能",
        ["市场分析", "交易记录", "策略分析", "系统设置"]
    )
    
    if page == "市场分析":
        st.header("市场分析")
        
        # 股票代码输入
        symbol = st.text_input("输入股票代码", "AAPL")
        
        # 时间范围选择
        period = st.selectbox(
            "选择时间范围",
            ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"]
        )
        
        if st.button("获取数据"):
            df = get_stock_data(symbol, period)
            if df is not None:
                df = calculate_technical_indicators(df)
                
                # 显示K线图
                fig = go.Figure(data=[go.Candlestick(x=df.index,
                                                    open=df['Open'],
                                                    high=df['High'],
                                                    low=df['Low'],
                                                    close=df['Close'])])
                fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], name='MA20'))
                fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], name='MA50'))
                st.plotly_chart(fig, use_container_width=True)
                
                # 显示技术指标
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("RSI指标")
                    fig_rsi = go.Figure(data=[go.Scatter(x=df.index, y=df['RSI'])])
                    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
                    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
                    st.plotly_chart(fig_rsi, use_container_width=True)
                
                with col2:
                    st.subheader("成交量")
                    fig_volume = go.Figure(data=[go.Bar(x=df.index, y=df['Volume'])])
                    st.plotly_chart(fig_volume, use_container_width=True)
    
    elif page == "交易记录":
        st.header("交易记录")
        ib = init_ib_connection()
        if ib is not None:
            # 这里添加交易记录获取和显示逻辑
            st.info("交易记录功能开发中...")
    
    elif page == "策略分析":
        st.header("策略分析")
        # 这里添加策略分析逻辑
        st.info("策略分析功能开发中...")
    
    elif page == "系统设置":
        st.header("系统设置")
        # 这里添加系统设置逻辑
        st.info("系统设置功能开发中...")

if __name__ == "__main__":
    main() 