# 智能交易助手系统

一个基于大模型的智能交易助手系统，提供实时市场分析、风险评估、策略回测和交易建议等功能。

## 功能特点

- 实时市场数据分析
- 智能风险评估
- 策略回测
- 实时监控和告警
- 大模型驱动的对话系统
- RESTful API接口

## 安装说明

1. 克隆仓库：
```bash
git clone https://github.com/yourusername/trading-assistant.git
cd trading-assistant
```

2. 创建虚拟环境：
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. 安装依赖：
```bash
pip install -r requirements.txt
```

4. 配置环境变量：
```bash
cp .env.example .env
# 编辑 .env 文件，填入必要的配置信息
```

5. 初始化数据库：
```bash
# 确保PostgreSQL已安装并运行
# 创建数据库
createdb trading_system
```

## 运行系统

1. 启动服务器：
```bash
python main.py
```

2. 访问API文档：
```
http://localhost:8000/docs
```

3. 健康检查：
```
http://localhost:8000/api/health
```

## API接口

### 对话接口
- `POST /api/chat`: 处理用户消息并返回交易建议

### 市场数据接口
- `POST /api/update_market_data`: 更新市场数据
- `GET /api/market_data`: 获取市场数据

### 风险分析接口
- `GET /api/risk_analysis`: 获取风险分析指标
- `GET /api/risk_metrics`: 获取风险指标历史数据

### 性能指标接口
- `GET /api/performance_metrics`: 获取性能指标
- `GET /api/backtest_results`: 获取回测结果

### 系统接口
- `GET /api/health`: 系统健康检查

## 配置说明

### 环境变量
- `DB_*`: 数据库配置
- `API_*`: API服务配置
- `TRADING_*`: 交易参数配置
- `RISK_*`: 风险控制参数
- `STRATEGY_*`: 策略配置
- `MONITOR_*`: 监控配置
- `REPORT_*`: 报告配置
- `LLM_*`: 大模型配置
- `CACHE_*`: 缓存配置

### 风险控制参数
- `MAX_DRAWDOWN`: 最大回撤限制
- `VAR_LIMIT`: VaR限制
- `VOLATILITY_LIMIT`: 波动率限制
- `BETA_LIMIT`: Beta限制
- `CORRELATION_LIMIT`: 相关性限制

## 使用示例

### 对话示例
```python
import requests

response = requests.post(
    "http://localhost:8000/api/chat",
    json={"message": "请分析当前市场趋势并给出交易建议"}
)
print(response.json())
```

### 更新市场数据示例
```python
import requests

market_data = {
    "symbol": "AAPL",
    "timestamp": "2023-01-01T00:00:00",
    "open": 150.0,
    "high": 155.0,
    "low": 149.0,
    "close": 153.0,
    "volume": 1000000
}

response = requests.post(
    "http://localhost:8000/api/update_market_data",
    json=market_data
)
print(response.json())
```

## 注意事项

1. 确保已正确配置所有必要的API密钥
2. 定期更新市场数据以保持系统实时性
3. 监控系统日志以排查潜在问题
4. 根据实际需求调整风险控制参数

## 贡献指南

1. Fork 项目
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建 Pull Request

## 许可证

MIT License 