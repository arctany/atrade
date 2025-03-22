"""Web server for the atrade application."""
import os
from typing import Dict, Any, List, Optional
import logging
import json
import uvicorn
from fastapi import FastAPI, HTTPException, Form, UploadFile, File, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from atrade.core.agent import TradingAgent

# Setup logging
logger = logging.getLogger(__name__)

# Create the FastAPI application
app = FastAPI(title="ATrade Trading System", version="0.1.0")

# Create a global trading agent instance
trading_agent = None

# 添加模型类
class BacktestConfig(BaseModel):
    start_date: str
    end_date: str
    symbol: str
    strategy: str
    initial_capital: float = 10000.0

class BacktestResult(BaseModel):
    performance: Dict[str, float]
    trades: List[Dict[str, Any]]
    equity_curve: List[Dict[str, float]]

# 添加优化模型
class OptimizationConfig(BaseModel):
    symbol: str
    strategy: str
    start_date: str
    end_date: str
    parameters: Dict[str, Dict[str, Any]]

class OptimizationResult(BaseModel):
    best_parameters: Dict[str, Any]
    performance_metrics: Dict[str, float]
    parameter_importance: Dict[str, float]
    tested_combinations: List[Dict[str, Any]]

# 模拟回测数据
def run_mock_backtest(config: BacktestConfig) -> BacktestResult:
    """运行模拟回测"""
    logger.info(f"Running backtest for {config.symbol} from {config.start_date} to {config.end_date}")
    
    # 模拟的回测结果
    performance = {
        "total_return": 15.7,
        "annualized_return": 12.3,
        "sharpe_ratio": 1.45,
        "max_drawdown": 8.2,
        "win_rate": 62.5,
        "profit_factor": 1.8
    }
    
    # 模拟的交易记录
    trades = [
        {"date": "2025-01-05", "action": "BUY", "symbol": config.symbol, "price": 150.25, "quantity": 10, "pnl": 0},
        {"date": "2025-01-15", "action": "SELL", "symbol": config.symbol, "price": 160.50, "quantity": 10, "pnl": 102.5},
        {"date": "2025-01-20", "action": "BUY", "symbol": config.symbol, "price": 155.75, "quantity": 12, "pnl": 0},
        {"date": "2025-02-01", "action": "SELL", "symbol": config.symbol, "price": 172.30, "quantity": 12, "pnl": 198.6},
        {"date": "2025-02-10", "action": "BUY", "symbol": config.symbol, "price": 165.40, "quantity": 15, "pnl": 0},
        {"date": "2025-02-25", "action": "SELL", "symbol": config.symbol, "price": 178.90, "quantity": 15, "pnl": 202.5},
    ]
    
    # 模拟的权益曲线
    equity_curve = []
    equity = config.initial_capital
    for i in range(60):
        date = f"2025-{(i//30)+1:02d}-{(i%30)+1:02d}"
        # 添加一些随机波动
        change = (i * 0.5) + ((i % 5) - 2) * 0.8
        equity += equity * (change / 100)
        equity_curve.append({"date": date, "equity": round(equity, 2)})
    
    return BacktestResult(
        performance=performance,
        trades=trades,
        equity_curve=equity_curve
    )

# 模拟优化过程
def run_mock_optimization(config: OptimizationConfig) -> OptimizationResult:
    """运行模拟优化"""
    logger.info(f"Running optimization for {config.symbol} with strategy {config.strategy}")
    
    # 模拟的最佳参数
    best_params = {}
    if config.strategy == "moving_average_crossover":
        best_params = {"short_window": 15, "long_window": 50}
    elif config.strategy == "rsi":
        best_params = {"rsi_period": 14, "oversold": 30, "overbought": 70}
    elif config.strategy == "bollinger_bands":
        best_params = {"window": 20, "num_std_dev": 2.2}
    else:
        best_params = {"fast_period": 12, "slow_period": 26, "signal_period": 9}
    
    # 模拟的性能指标
    performance = {
        "total_return": 18.2,
        "sharpe_ratio": 1.65,
        "max_drawdown": 7.3,
        "win_rate": 67.8,
        "profit_factor": 2.1
    }
    
    # 模拟的参数重要性
    param_importance = {}
    for key in best_params:
        param_importance[key] = round(0.3 + 0.5 * (hash(key) % 100) / 100, 2)
    
    # 模拟的测试组合
    tested_combinations = []
    for i in range(10):
        combo = {"iteration": i+1}
        for key in best_params:
            # 添加一些随机变化
            combo[key] = best_params[key] + (i % 5 - 2)
        combo["total_return"] = round(performance["total_return"] - (i*1.2), 2)
        combo["sharpe_ratio"] = round(performance["sharpe_ratio"] - (i*0.1), 2)
        tested_combinations.append(combo)
    
    return OptimizationResult(
        best_parameters=best_params,
        performance_metrics=performance,
        parameter_importance=param_importance,
        tested_combinations=tested_combinations
    )

@app.get("/", response_class=HTMLResponse)
async def root():
    """Return the HTML home page"""
    logger.info("Serving home page")
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>ATrade Trading System</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                line-height: 1.6;
            }
            h1 {
                color: #2c3e50;
                border-bottom: 1px solid #eee;
                padding-bottom: 10px;
            }
            .card {
                background-color: #f9f9f9;
                border-radius: 5px;
                padding: 20px;
                margin-bottom: 20px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .status {
                display: inline-block;
                padding: 5px 10px;
                border-radius: 3px;
                color: white;
                font-weight: bold;
            }
            .running {
                background-color: #27ae60;
            }
            .stopped {
                background-color: #e74c3c;
            }
            button {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 10px 15px;
                border-radius: 3px;
                cursor: pointer;
                font-size: 14px;
                margin-right: 10px;
            }
            button:hover {
                background-color: #2980b9;
            }
            #dashboard {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
            }
            .nav {
                display: flex;
                background-color: #34495e;
                padding: 10px;
                border-radius: 5px;
                margin-bottom: 20px;
            }
            .nav a {
                color: white;
                text-decoration: none;
                padding: 10px 15px;
                border-radius: 3px;
                margin-right: 5px;
            }
            .nav a:hover, .nav a.active {
                background-color: #2c3e50;
            }
        </style>
    </head>
    <body>
        <h1>ATrade Trading System</h1>
        
        <div class="nav">
            <a href="/" class="active">Dashboard</a>
            <a href="/backtest">Backtest</a>
            <a href="/optimize">Optimization</a>
            <a href="/settings">Settings</a>
        </div>
        
        <div class="card">
            <h2>Trading Agent Status</h2>
            <p>Status: <span class="status running">Running</span></p>
            <button onclick="fetch('/api/agent/stop').then(() => alert('Agent stopping command sent'))">Stop Agent</button>
            <button onclick="fetch('/api/agent/start').then(() => alert('Agent starting command sent'))">Start Agent</button>
        </div>
        
        <div id="dashboard">
            <div class="card">
                <h2>Account Summary</h2>
                <p>Initial Capital: $100,000.00</p>
                <p>Current Value: $105,324.78</p>
                <p>Total Profit/Loss: +$5,324.78 (+5.32%)</p>
            </div>
            
            <div class="card">
                <h2>Open Positions</h2>
                <p>AAPL: 100 shares @ $180.25</p>
                <p>MSFT: 50 shares @ $322.18</p>
                <p>GOOGL: 25 shares @ $148.67</p>
            </div>
        </div>
        
        <div class="card">
            <h2>Recent Trades</h2>
            <p>2025-03-22 10:30:45 - BUY 100 AAPL @ $180.25</p>
            <p>2025-03-22 10:15:22 - BUY 50 MSFT @ $322.18</p>
            <p>2025-03-22 09:45:10 - BUY 25 GOOGL @ $148.67</p>
            <p>2025-03-22 09:30:05 - SELL 75 AMZN @ $185.32</p>
        </div>
        
        <script>
            // Simple script to update status periodically
            function updateStatus() {
                fetch('/api/agent/status')
                    .then(response => response.json())
                    .then(data => {
                        const statusElement = document.querySelector('.status');
                        if (data.running) {
                            statusElement.textContent = 'Running';
                            statusElement.className = 'status running';
                        } else {
                            statusElement.textContent = 'Stopped';
                            statusElement.className = 'status stopped';
                        }
                    })
                    .catch(error => console.error('Error fetching status:', error));
            }
            
            // Update status every 5 seconds
            setInterval(updateStatus, 5000);
        </script>
    </body>
    </html>
    """
    return html_content

@app.get("/backtest", response_class=HTMLResponse)
async def backtest_page():
    """Return the HTML backtest page"""
    logger.info("Serving backtest page")
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>ATrade - Backtest</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 1000px;
                margin: 0 auto;
                padding: 20px;
                line-height: 1.6;
            }
            h1 {
                color: #2c3e50;
                border-bottom: 1px solid #eee;
                padding-bottom: 10px;
            }
            .card {
                background-color: #f9f9f9;
                border-radius: 5px;
                padding: 20px;
                margin-bottom: 20px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            label {
                display: block;
                margin-bottom: 5px;
                font-weight: bold;
            }
            input, select {
                width: 100%;
                padding: 8px;
                margin-bottom: 15px;
                border: 1px solid #ddd;
                border-radius: 4px;
                box-sizing: border-box;
            }
            button {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 10px 15px;
                border-radius: 3px;
                cursor: pointer;
                font-size: 14px;
            }
            button:hover {
                background-color: #2980b9;
            }
            .results {
                display: none;
                margin-top: 20px;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 20px;
            }
            th, td {
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }
            th {
                background-color: #f2f2f2;
            }
            tr:nth-child(even) {
                background-color: #f9f9f9;
            }
            .nav {
                display: flex;
                background-color: #34495e;
                padding: 10px;
                border-radius: 5px;
                margin-bottom: 20px;
            }
            .nav a {
                color: white;
                text-decoration: none;
                padding: 10px 15px;
                border-radius: 3px;
                margin-right: 5px;
            }
            .nav a:hover, .nav a.active {
                background-color: #2c3e50;
            }
            .tab-container {
                margin-top: 20px;
            }
            .tab-buttons {
                display: flex;
                border-bottom: 1px solid #ddd;
                margin-bottom: 15px;
            }
            .tab-button {
                padding: 10px 15px;
                background: none;
                border: none;
                color: #333;
                cursor: pointer;
                margin-right: 5px;
            }
            .tab-button.active {
                border-bottom: 2px solid #3498db;
                font-weight: bold;
            }
            .tab-content {
                display: none;
            }
            .tab-content.active {
                display: block;
            }
            #equityChart {
                width: 100%;
                height: 300px;
                margin-top: 20px;
            }
            .metrics {
                display: grid;
                grid-template-columns: 1fr 1fr 1fr;
                gap: 10px;
                margin-bottom: 20px;
            }
            .metric {
                padding: 15px;
                border-radius: 5px;
                background-color: #ecf0f1;
                text-align: center;
            }
            .metric h3 {
                margin: 0;
                color: #7f8c8d;
                font-size: 14px;
            }
            .metric p {
                margin: 5px 0 0;
                font-size: 24px;
                font-weight: bold;
                color: #2c3e50;
            }
            .positive {
                color: #27ae60 !important;
            }
            .negative {
                color: #e74c3c !important;
            }
        </style>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    </head>
    <body>
        <h1>Backtest Trading Strategies</h1>
        
        <div class="nav">
            <a href="/">Dashboard</a>
            <a href="/backtest" class="active">Backtest</a>
            <a href="/optimize">Optimization</a>
            <a href="/settings">Settings</a>
        </div>
        
        <div class="card">
            <h2>Backtest Configuration</h2>
            <form id="backtestForm">
                <div>
                    <label for="symbol">Symbol:</label>
                    <input type="text" id="symbol" name="symbol" value="AAPL" required>
                </div>
                
                <div>
                    <label for="startDate">Start Date:</label>
                    <input type="date" id="startDate" name="start_date" value="2025-01-01" required>
                </div>
                
                <div>
                    <label for="endDate">End Date:</label>
                    <input type="date" id="endDate" name="end_date" value="2025-03-01" required>
                </div>
                
                <div>
                    <label for="strategy">Strategy:</label>
                    <select id="strategy" name="strategy" required>
                        <option value="moving_average_crossover">Moving Average Crossover</option>
                        <option value="rsi">RSI Strategy</option>
                        <option value="bollinger_bands">Bollinger Bands</option>
                        <option value="macd">MACD Strategy</option>
                    </select>
                </div>
                
                <div>
                    <label for="initialCapital">Initial Capital:</label>
                    <input type="number" id="initialCapital" name="initial_capital" value="10000" min="1000" step="1000" required>
                </div>
                
                <button type="submit">Run Backtest</button>
            </form>
        </div>
        
        <div class="results" id="backtestResults">
            <div class="card">
                <h2>Backtest Results</h2>
                
                <div class="metrics">
                    <div class="metric">
                        <h3>Total Return</h3>
                        <p id="totalReturn" class="positive">+15.7%</p>
                    </div>
                    <div class="metric">
                        <h3>Sharpe Ratio</h3>
                        <p id="sharpeRatio">1.45</p>
                    </div>
                    <div class="metric">
                        <h3>Max Drawdown</h3>
                        <p id="maxDrawdown" class="negative">-8.2%</p>
                    </div>
                    <div class="metric">
                        <h3>Win Rate</h3>
                        <p id="winRate">62.5%</p>
                    </div>
                    <div class="metric">
                        <h3>Profit Factor</h3>
                        <p id="profitFactor">1.8</p>
                    </div>
                    <div class="metric">
                        <h3>Annualized Return</h3>
                        <p id="annualizedReturn" class="positive">+12.3%</p>
                    </div>
                </div>
                
                <div class="tab-container">
                    <div class="tab-buttons">
                        <button class="tab-button active" onclick="showTab('equityTab')">Equity Curve</button>
                        <button class="tab-button" onclick="showTab('tradesTab')">Trades</button>
                    </div>
                    
                    <div id="equityTab" class="tab-content active">
                        <canvas id="equityChart"></canvas>
                    </div>
                    
                    <div id="tradesTab" class="tab-content">
                        <table id="tradesTable">
                            <thead>
                                <tr>
                                    <th>Date</th>
                                    <th>Action</th>
                                    <th>Symbol</th>
                                    <th>Price</th>
                                    <th>Quantity</th>
                                    <th>P&L</th>
                                </tr>
                            </thead>
                            <tbody>
                                <!-- Trades will be populated here -->
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            // Function to show tab content
            function showTab(tabId) {
                // Hide all tab content
                document.querySelectorAll('.tab-content').forEach(tab => {
                    tab.classList.remove('active');
                });
                
                // Remove active class from all tab buttons
                document.querySelectorAll('.tab-button').forEach(button => {
                    button.classList.remove('active');
                });
                
                // Show selected tab content
                document.getElementById(tabId).classList.add('active');
                
                // Add active class to clicked button
                event.currentTarget.classList.add('active');
            }
            
            // Equity chart instance
            let equityChart;
            
            // Handle form submission
            document.getElementById('backtestForm').addEventListener('submit', function(e) {
                e.preventDefault();
                
                // Get form data
                const formData = new FormData(this);
                const params = {
                    symbol: formData.get('symbol'),
                    start_date: formData.get('start_date'),
                    end_date: formData.get('end_date'),
                    strategy: formData.get('strategy'),
                    initial_capital: parseFloat(formData.get('initial_capital'))
                };
                
                // Call API to run backtest
                fetch('/api/backtest', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(params)
                })
                .then(response => response.json())
                .then(data => {
                    // Show results section
                    document.getElementById('backtestResults').style.display = 'block';
                    
                    // Update performance metrics
                    document.getElementById('totalReturn').textContent = `${data.performance.total_return}%`;
                    document.getElementById('sharpeRatio').textContent = data.performance.sharpe_ratio;
                    document.getElementById('maxDrawdown').textContent = `-${data.performance.max_drawdown}%`;
                    document.getElementById('winRate').textContent = `${data.performance.win_rate}%`;
                    document.getElementById('profitFactor').textContent = data.performance.profit_factor;
                    document.getElementById('annualizedReturn').textContent = `${data.performance.annualized_return}%`;
                    
                    // Populate trades table
                    const tbody = document.querySelector('#tradesTable tbody');
                    tbody.innerHTML = '';
                    
                    data.trades.forEach(trade => {
                        const row = document.createElement('tr');
                        row.innerHTML = `
                            <td>${trade.date}</td>
                            <td>${trade.action}</td>
                            <td>${trade.symbol}</td>
                            <td>$${trade.price.toFixed(2)}</td>
                            <td>${trade.quantity}</td>
                            <td>${trade.pnl > 0 ? '+$' + trade.pnl.toFixed(2) : trade.pnl === 0 ? '-' : '-$' + Math.abs(trade.pnl).toFixed(2)}</td>
                        `;
                        tbody.appendChild(row);
                    });
                    
                    // Create equity curve chart
                    const dates = data.equity_curve.map(point => point.date);
                    const equities = data.equity_curve.map(point => point.equity);
                    
                    if (equityChart) {
                        equityChart.destroy();
                    }
                    
                    const ctx = document.getElementById('equityChart').getContext('2d');
                    equityChart = new Chart(ctx, {
                        type: 'line',
                        data: {
                            labels: dates,
                            datasets: [{
                                label: 'Equity Curve',
                                data: equities,
                                backgroundColor: 'rgba(52, 152, 219, 0.2)',
                                borderColor: 'rgba(52, 152, 219, 1)',
                                borderWidth: 2,
                                pointRadius: 1,
                                tension: 0.1
                            }]
                        },
                        options: {
                            responsive: true,
                            scales: {
                                y: {
                                    beginAtZero: false
                                }
                            }
                        }
                    });
                })
                .catch(error => {
                    console.error('Error running backtest:', error);
                    alert('Error running backtest. Please try again.');
                });
            });
        </script>
    </body>
    </html>
    """
    return html_content
    <body>
        <h1>Optimize Trading Strategies</h1>
        
        <div class="nav">
            <a href="/">Dashboard</a>
            <a href="/backtest">Backtest</a>
            <a href="/optimize" class="active">Optimization</a>
            <a href="/settings">Settings</a>
        </div>
        
        <div class="card">
            <h2>Optimization Configuration</h2>
            <form id="optimizeForm">
                <div>
                    <label for="symbol">Symbol:</label>
                    <input type="text" id="symbol" name="symbol" value="AAPL" required>
                </div>
                
                <div>
                    <label for="startDate">Start Date:</label>
                    <input type="date" id="startDate" name="start_date" value="2025-01-01" required>
                </div>
                
                <div>
                    <label for="endDate">End Date:</label>
                    <input type="date" id="endDate" name="end_date" value="2025-03-01" required>
                </div>
                
                <div>
                    <label for="strategy">Strategy:</label>
                    <select id="strategy" name="strategy" onchange="updateParameterForm()" required>
                        <option value="moving_average_crossover">Moving Average Crossover</option>
                        <option value="rsi">RSI Strategy</option>
                        <option value="bollinger_bands">Bollinger Bands</option>
                        <option value="macd">MACD Strategy</option>
                    </select>
                </div>
                
                <div id="parameterContainer">
                    <!-- Parameter inputs will be added here based on strategy selection -->
                </div>
                
                <button type="submit">Run Optimization</button>
            </form>
        </div>
        
        <div class="results" id="optimizationResults">
            <div class="card">
                <h2>Optimization Results</h2>
                
                <div>
                    <h3>Best Parameters:</h3>
                    <div id="bestParameters"></div>
                </div>
                
                <div class="metrics">
                    <div class="metric">
                        <h3>Total Return</h3>
                        <p id="totalReturn" class="positive">+18.2%</p>
                    </div>
                    <div class="metric">
                        <h3>Sharpe Ratio</h3>
                        <p id="sharpeRatio">1.65</p>
                    </div>
                    <div class="metric">
                        <h3>Max Drawdown</h3>
                        <p id="maxDrawdown" class="negative">-7.3%</p>
                    </div>
                </div>
                
                <div class="tab-container">
                    <div class="tab-buttons">
                        <button class="tab-button active" onclick="showTab('importanceTab')">Parameter Importance</button>
                        <button class="tab-button" onclick="showTab('combinationsTab')">Tested Combinations</button>
                    </div>
                    
                    <div id="importanceTab" class="tab-content active">
                        <canvas id="parameterImportanceChart"></canvas>
                    </div>
                    
                    <div id="combinationsTab" class="tab-content">
                        <table id="combinationsTable">
                            <thead>
                                <tr>
                                    <th>Iteration</th>
                                    <th colspan="3">Parameters</th>
                                    <th>Total Return</th>
                                    <th>Sharpe Ratio</th>
                                </tr>
                            </thead>
                            <tbody>
                                <!-- Combinations will be populated here -->
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            // Function to show tab content
            function showTab(tabId) {
                // Hide all tab content
                document.querySelectorAll('.tab-content').forEach(tab => {
                    tab.classList.remove('active');
                });
                
                // Remove active class from all tab buttons
                document.querySelectorAll('.tab-button').forEach(button => {
                    button.classList.remove('active');
                });
                
                // Show selected tab content
                document.getElementById(tabId).classList.add('active');
                
                // Add active class to clicked button
                event.currentTarget.classList.add('active');
            }
            
            // Parameter configurations by strategy
            const strategyParameters = {
                moving_average_crossover: {
                    short_window: {
                        min: 5,
                        max: 50,
                        step: 5,
                        default: 10
                    },
                    long_window: {
                        min: 20,
                        max: 200,
                        step: 10,
                        default: 50
                    }
                },
                rsi: {
                    rsi_period: {
                        min: 2,
                        max: 30,
                        step: 1,
                        default: 14
                    },
                    oversold: {
                        min: 10,
                        max: 40,
                        step: 5,
                        default: 30
                    },
                    overbought: {
                        min: 60,
                        max: 90,
                        step: 5,
                        default: 70
                    }
                },
                bollinger_bands: {
                    window: {
                        min: 5,
                        max: 50,
                        step: 5,
                        default: 20
                    },
                    num_std_dev: {
                        min: 1,
                        max: 4,
                        step: 0.5,
                        default: 2
                    }
                },
                macd: {
                    fast_period: {
                        min: 5,
                        max: 20,
                        step: 1,
                        default: 12
                    },
                    slow_period: {
                        min: 15,
                        max: 50,
                        step: 1,
                        default: 26
                    },
                    signal_period: {
                        min: 5,
                        max: 20,
                        step: 1,
                        default: 9
                    }
                }
            };
            
            // Parameter importance chart instance
            let importanceChart;
            
            // Function to update parameter form based on strategy selection
            function updateParameterForm() {
                const strategy = document.getElementById('strategy').value;
                const container = document.getElementById('parameterContainer');
                container.innerHTML = '';
                
                // Get parameters for selected strategy
                const parameters = strategyParameters[strategy];
                
                // Create parameter inputs
                const paramGroup = document.createElement('div');
                paramGroup.className = 'parameter-group';
                paramGroup.innerHTML = '<h3>Parameter Ranges</h3>';
                
                for (const [param, config] of Object.entries(parameters)) {
                    const paramDiv = document.createElement('div');
                    paramDiv.className = 'parameter-input';
                    
                    // Parameter label
                    const label = document.createElement('label');
                    label.style.width = '150px';
                    label.textContent = param.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase());
                    
                    // Create inputs for min, max, step
                    const minInput = document.createElement('input');
                    minInput.type = 'number';
                    minInput.name = `parameters.${param}.min`;
                    minInput.value = config.min;
                    minInput.placeholder = 'Min';
                    minInput.required = true;
                    
                    const maxInput = document.createElement('input');
                    maxInput.type = 'number';
                    maxInput.name = `parameters.${param}.max`;
                    maxInput.value = config.max;
                    maxInput.placeholder = 'Max';
                    maxInput.required = true;
                    
                    const stepInput = document.createElement('input');
                    stepInput.type = 'number';
                    stepInput.name = `parameters.${param}.step`;
                    stepInput.value = config.step;
                    stepInput.placeholder = 'Step';
                    stepInput.required = true;
                    stepInput.step = param.includes('std') ? '0.1' : '1';
                    
                    // Add everything to the parameter div
                    paramDiv.appendChild(label);
                    paramDiv.appendChild(minInput);
                    paramDiv.appendChild(maxInput);
                    paramDiv.appendChild(stepInput);
                    
                    // Add to parameter group
                    paramGroup.appendChild(paramDiv);
                }
                
                container.appendChild(paramGroup);
            }
            
            // Initialize parameter form
            document.addEventListener('DOMContentLoaded', updateParameterForm);
            
            // Function to gather form data in the correct format
            function gatherFormData() {
                const form = document.getElementById('optimizeForm');
                const formData = new FormData(form);
                
                // Basic configuration
                const config = {
                    symbol: formData.get('symbol'),
                    start_date: formData.get('start_date'),
                    end_date: formData.get('end_date'),
                    strategy: formData.get('strategy'),
                    parameters: {}
                };
                
                // Process parameters
                for (const [key, value] of formData.entries()) {
                    if (key.startsWith('parameters.')) {
                        const [_, param, field] = key.split('.');
                        
                        if (!config.parameters[param]) {
                            config.parameters[param] = {};
                        }
                        
                        config.parameters[param][field] = param.includes('std') ? parseFloat(value) : parseInt(value);
                    }
                }
                
                return config;
            }
            
            // Handle form submission
            document.getElementById('optimizeForm').addEventListener('submit', function(e) {
                e.preventDefault();
                
                // Get form data
                const params = gatherFormData();
                
                // Call API to run optimization
                fetch('/api/optimize', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(params)
                })
                .then(response => response.json())
                .then(data => {
                    // Show results section
                    document.getElementById('optimizationResults').style.display = 'block';
                    
                    // Update performance metrics
                    document.getElementById('totalReturn').textContent = `${data.performance_metrics.total_return}%`;
                    document.getElementById('sharpeRatio').textContent = data.performance_metrics.sharpe_ratio;
                    document.getElementById('maxDrawdown').textContent = `-${data.performance_metrics.max_drawdown}%`;
                    
                    // Display best parameters
                    const bestParamsDiv = document.getElementById('bestParameters');
                    bestParamsDiv.innerHTML = '';
                    
                    for (const [param, value] of Object.entries(data.best_parameters)) {
                        const paramSpan = document.createElement('div');
                        paramSpan.innerHTML = `<strong>${param.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}:</strong> ${value}`;
                        paramSpan.style.marginBottom = '5px';
                        bestParamsDiv.appendChild(paramSpan);
                    }
                    
                    // Populate combinations table
                    const tbody = document.querySelector('#combinationsTable tbody');
                    tbody.innerHTML = '';
                    
                    const paramKeys = Object.keys(data.best_parameters);
                    
                    // Update table header for dynamic parameters
                    const headerRow = document.querySelector('#combinationsTable thead tr');
                    headerRow.innerHTML = '<th>Iteration</th>';
                    paramKeys.forEach(param => {
                        headerRow.innerHTML += `<th>${param.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}</th>`;
                    });
                    headerRow.innerHTML += '<th>Total Return</th><th>Sharpe Ratio</th>';
                    
                    data.tested_combinations.forEach(combo => {
                        const row = document.createElement('tr');
                        let rowHtml = `<td>${combo.iteration}</td>`;
                        
                        paramKeys.forEach(param => {
                            rowHtml += `<td>${combo[param]}</td>`;
                        });
                        
                        rowHtml += `
                            <td>${combo.total_return}%</td>
                            <td>${combo.sharpe_ratio}</td>
                        `;
                        row.innerHTML = rowHtml;
                        tbody.appendChild(row);
                    });
                    
                    // Create parameter importance chart
                    if (importanceChart) {
                        importanceChart.destroy();
                    }
                    
                    const importanceLabels = Object.keys(data.parameter_importance).map(
                        param => param.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())
                    );
                    const importanceValues = Object.values(data.parameter_importance);
                    
                    const ctx = document.getElementById('parameterImportanceChart').getContext('2d');
                    importanceChart = new Chart(ctx, {
                        type: 'bar',
                        data: {
                            labels: importanceLabels,
                            datasets: [{
                                label: 'Parameter Importance',
                                data: importanceValues,
                                backgroundColor: 'rgba(52, 152, 219, 0.7)',
                                borderColor: 'rgba(52, 152, 219, 1)',
                                borderWidth: 1
                            }]
                        },
                        options: {
                            indexAxis: 'y',
                            responsive: true,
                            scales: {
                                x: {
                                    beginAtZero: true,
                                    max: 1
                                }
                            }
                        }
                    });
                })
                .catch(error => {
                    console.error('Error running optimization:', error);
                    alert('Error running optimization. Please try again.');
                });
            });
        </script>
    </body>
    </html>
    """
    return html_content

@app.get("/settings", response_class=HTMLResponse)
async def settings_page():
    """Return the HTML settings page"""
    logger.info("Serving settings page")
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>ATrade - Settings</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                line-height: 1.6;
            }
            h1 {
                color: #2c3e50;
                border-bottom: 1px solid #eee;
                padding-bottom: 10px;
            }
            .card {
                background-color: #f9f9f9;
                border-radius: 5px;
                padding: 20px;
                margin-bottom: 20px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            label {
                display: block;
                margin-bottom: 5px;
                font-weight: bold;
            }
            input, select, textarea {
                width: 100%;
                padding: 8px;
                margin-bottom: 15px;
                border: 1px solid #ddd;
                border-radius: 4px;
                box-sizing: border-box;
            }
            button {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 10px 15px;
                border-radius: 3px;
                cursor: pointer;
                font-size: 14px;
                margin-right: 10px;
            }
            button:hover {
                background-color: #2980b9;
            }
            .nav {
                display: flex;
                background-color: #34495e;
                padding: 10px;
                border-radius: 5px;
                margin-bottom: 20px;
            }
            .nav a {
                color: white;
                text-decoration: none;
                padding: 10px 15px;
                border-radius: 3px;
                margin-right: 5px;
            }
            .nav a:hover, .nav a.active {
                background-color: #2c3e50;
            }
            .settings-group {
                margin-bottom: 20px;
            }
            .checkbox-group {
                display: flex;
                align-items: center;
            }
            .checkbox-group input {
                width: auto;
                margin-right: 10px;
            }
            .checkbox-group label {
                margin-bottom: 0;
            }
            textarea {
                height: 150px;
                font-family: monospace;
            }
            .success-message {
                background-color: #27ae60;
                color: white;
                padding: 10px;
                border-radius: 3px;
                margin-bottom: 15px;
                display: none;
            }
            .error-message {
                background-color: #e74c3c;
                color: white;
                padding: 10px;
                border-radius: 3px;
                margin-bottom: 15px;
                display: none;
            }
        </style>
    </head>
    <body>
        <h1>System Settings</h1>
        
        <div class="nav">
            <a href="/">Dashboard</a>
            <a href="/backtest">Backtest</a>
            <a href="/optimize">Optimization</a>
            <a href="/settings" class="active">Settings</a>
        </div>
        
        <div id="successMessage" class="success-message">
            Settings saved successfully!
        </div>
        
        <div id="errorMessage" class="error-message">
            Error saving settings. Please try again.
        </div>
        
        <div class="card">
            <h2>Trading Settings</h2>
            <form id="tradingSettingsForm">
                <div class="settings-group">
                    <h3>General</h3>
                    <div>
                        <label for="tradingMode">Trading Mode:</label>
                        <select id="tradingMode" name="trading_mode">
                            <option value="paper">Paper Trading</option>
                            <option value="live">Live Trading</option>
                            <option value="backtest">Backtest</option>
                        </select>
                    </div>
                    
                    <div>
                        <label for="initialCapital">Initial Capital:</label>
                        <input type="number" id="initialCapital" name="initial_capital" value="10000" min="1000" step="1000">
                    </div>
                    
                    <div class="checkbox-group">
                        <input type="checkbox" id="autoStart" name="auto_start" checked>
                        <label for="autoStart">Auto-start trading on system startup</label>
                    </div>
                </div>
                
                <div class="settings-group">
                    <h3>Risk Management</h3>
                    <div>
                        <label for="maxPositionSize">Max Position Size (% of capital):</label>
                        <input type="number" id="maxPositionSize" name="max_position_size" value="10" min="1" max="100">
                    </div>
                    
                    <div>
                        <label for="maxLoss">Max Daily Loss (% of capital):</label>
                        <input type="number" id="maxLoss" name="max_loss" value="5" min="1" max="100">
                    </div>
                    
                    <div>
                        <label for="stopLoss">Default Stop Loss (%):</label>
                        <input type="number" id="stopLoss" name="stop_loss" value="2" min="0.1" max="50" step="0.1">
                    </div>
                    
                    <div>
                        <label for="takeProfit">Default Take Profit (%):</label>
                        <input type="number" id="takeProfit" name="take_profit" value="5" min="0.1" max="100" step="0.1">
                    </div>
                </div>
                
                <button type="submit">Save Trading Settings</button>
            </form>
        </div>
        
        <div class="card">
            <h2>API Connections</h2>
            <form id="apiSettingsForm">
                <div class="settings-group">
                    <h3>Exchange API</h3>
                    <div>
                        <label for="exchangeType">Exchange:</label>
                        <select id="exchangeType" name="exchange_type">
                            <option value="binance">Binance</option>
                            <option value="coinbase">Coinbase</option>
                            <option value="interactive_brokers">Interactive Brokers</option>
                            <option value="alpaca">Alpaca</option>
                        </select>
                    </div>
                    
                    <div>
                        <label for="apiKey">API Key:</label>
                        <input type="text" id="apiKey" name="api_key" placeholder="Enter your API key">
                    </div>
                    
                    <div>
                        <label for="apiSecret">API Secret:</label>
                        <input type="password" id="apiSecret" name="api_secret" placeholder="Enter your API secret">
                    </div>
                    
                    <div class="checkbox-group">
                        <input type="checkbox" id="testMode" name="test_mode" checked>
                        <label for="testMode">Use test/sandbox environment</label>
                    </div>
                </div>
                
                <div class="settings-group">
                    <h3>Data Provider</h3>
                    <div>
                        <label for="dataProvider">Provider:</label>
                        <select id="dataProvider" name="data_provider">
                            <option value="yahoo">Yahoo Finance</option>
                            <option value="alpha_vantage">Alpha Vantage</option>
                            <option value="polygon">Polygon.io</option>
                        </select>
                    </div>
                    
                    <div>
                        <label for="dataApiKey">API Key:</label>
                        <input type="text" id="dataApiKey" name="data_api_key" placeholder="Enter data provider API key">
                    </div>
                </div>
                
                <button type="submit">Save API Settings</button>
            </form>
        </div>
        
        <div class="card">
            <h2>System Configuration</h2>
            <form id="systemSettingsForm">
                <div>
                    <label for="logLevel">Log Level:</label>
                    <select id="logLevel" name="log_level">
                        <option value="DEBUG">Debug</option>
                        <option value="INFO" selected>Info</option>
                        <option value="WARNING">Warning</option>
                        <option value="ERROR">Error</option>
                    </select>
                </div>
                
                <div>
                    <label for="configYaml">Configuration YAML:</label>
                    <textarea id="configYaml" name="config_yaml" spellcheck="false">trading:
  mode: paper
  initial_capital: 10000
  auto_start: true

risk:
  max_position_size: 10
  max_daily_loss: 5
  default_stop_loss: 2
  default_take_profit: 5

api:
  exchange:
    type: binance
    test_mode: true
    
  data:
    provider: yahoo
    
logging:
  level: INFO
  file: logs/atrade.log</textarea>
                </div>
                
                <button type="submit">Save System Settings</button>
                <button type="button" class="danger" onclick="resetSettings()">Reset All Settings</button>
            </form>
        </div>
        
        <script>
            // Show success message
            function showSuccess() {
                const successMsg = document.getElementById('successMessage');
                successMsg.style.display = 'block';
                
                // Hide after 3 seconds
                setTimeout(() => {
                    successMsg.style.display = 'none';
                }, 3000);
            }
            
            // Show error message
            function showError(message) {
                const errorMsg = document.getElementById('errorMessage');
                errorMsg.textContent = message || 'Error saving settings. Please try again.';
                errorMsg.style.display = 'block';
                
                // Hide after 5 seconds
                setTimeout(() => {
                    errorMsg.style.display = 'none';
                }, 5000);
            }
            
            // Handle trading settings form submission
            document.getElementById('tradingSettingsForm').addEventListener('submit', function(e) {
                e.preventDefault();
                
                // Get form data
                const formData = new FormData(this);
                const settings = {};
                
                formData.forEach((value, key) => {
                    // Convert to appropriate types
                    if (key === 'auto_start' || key === 'test_mode') {
                        settings[key] = true; // Checkbox is checked
                    } else if (key.includes('capital') || key.includes('size') || key.includes('loss') || key.includes('profit')) {
                        settings[key] = parseFloat(value);
                    } else {
                        settings[key] = value;
                    }
                });
                
                // Simulate API call
                setTimeout(() => {
                    console.log('Trading settings saved:', settings);
                    showSuccess();
                }, 500);
            });
            
            // Handle API settings form submission
            document.getElementById('apiSettingsForm').addEventListener('submit', function(e) {
                e.preventDefault();
                
                // Get form data
                const formData = new FormData(this);
                const settings = {};
                
                formData.forEach((value, key) => {
                    if (key === 'test_mode') {
                        settings[key] = true; // Checkbox is checked
                    } else {
                        settings[key] = value;
                    }
                });
                
                // Simulate API call
                setTimeout(() => {
                    console.log('API settings saved:', settings);
                    showSuccess();
                }, 500);
            });
            
            // Handle system settings form submission
            document.getElementById('systemSettingsForm').addEventListener('submit', function(e) {
                e.preventDefault();
                
                // Get form data
                const formData = new FormData(this);
                const settings = {};
                
                formData.forEach((value, key) => {
                    settings[key] = value;
                });
                
                // Validate YAML
                try {
                    // In a real app, you would validate the YAML here
                    
                    // Simulate API call
                    setTimeout(() => {
                        console.log('System settings saved:', settings);
                        showSuccess();
                    }, 500);
                } catch (err) {
                    showError('Invalid YAML configuration');
                }
            });
            
            // Reset all settings
            function resetSettings() {
                if (confirm('Are you sure you want to reset all settings to default values? This cannot be undone.')) {
                    // Reset form values
                    document.getElementById('tradingSettingsForm').reset();
                    document.getElementById('apiSettingsForm').reset();
                    document.getElementById('systemSettingsForm').reset();
                    
                    // Restore default YAML
                    document.getElementById('configYaml').value = `trading:
  mode: paper
  initial_capital: 10000
  auto_start: true

risk:
  max_position_size: 10
  max_daily_loss: 5
  default_stop_loss: 2
  default_take_profit: 5

api:
  exchange:
    type: binance
    test_mode: true
    
  data:
    provider: yahoo
    
logging:
  level: INFO
  file: logs/atrade.log`;
                    
                    showSuccess();
                }
            }
        </script>
    </body>
    </html>
    """
    return html_content

@app.get("/api/agent/status")
async def agent_status():
    """Return the status of the trading agent"""
    logger.debug("Checking agent status")
    global trading_agent
    if trading_agent is None:
        return {"running": False, "message": "Agent not initialized"}
    return {"running": True, "message": "Agent is running"}

@app.get("/api/agent/start")
async def start_agent():
    """Start the trading agent"""
    logger.info("Starting trading agent")
    global trading_agent
    if trading_agent is None:
        trading_agent = TradingAgent(config={})
        trading_agent.start()
        return {"status": "success", "message": "Agent started"}
    else:
        return {"status": "warning", "message": "Agent already running"}

@app.get("/api/agent/stop")
async def stop_agent():
    """Stop the trading agent"""
    logger.info("Stopping trading agent")
    global trading_agent
    if trading_agent is not None:
        trading_agent.stop()
        trading_agent = None
        return {"status": "success", "message": "Agent stopped"}
    else:
        return {"status": "warning", "message": "No agent running"}

@app.post("/api/backtest")
async def backtest_api(config: BacktestConfig):
    """Run a backtest with the given configuration"""
    logger.info(f"Received backtest request: {config}")
    
    # 运行模拟回测
    result = run_mock_backtest(config)
    
    return result

@app.post("/api/optimize")
async def optimize_api(config: OptimizationConfig):
    """Run optimization with the given configuration"""
    logger.info(f"Received optimization request: {config}")
    
    # 运行模拟优化
    result = run_mock_optimization(config)
    
    return result

def start_server(host: str = "0.0.0.0", port: int = 8000):
    """Start the FastAPI server"""
    logger.info(f"Starting API server on http://{host}:{port}")
    
    # Configure logging for uvicorn
    log_config = uvicorn.config.LOGGING_CONFIG
    log_config["formatters"]["access"]["fmt"] = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_config["formatters"]["default"]["fmt"] = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Start the server
    uvicorn.run(
        "atrade.api.server:app", 
        host=host, 
        port=port, 
        log_config=log_config,
        reload=False
    )

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    start_server() 