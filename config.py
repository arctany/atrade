import os
from dotenv import load_dotenv
from typing import Dict, List

# 加载环境变量
load_dotenv()

# JWT配置
JWT_CONFIG = {
    'secret_key': os.getenv('JWT_SECRET_KEY', 'your-secret-key'),
    'algorithm': 'HS256',
    'expire_minutes': int(os.getenv('JWT_EXPIRE_MINUTES', '60'))
}

# IBKR配置
IBKR_CONFIG = {
    'host': os.getenv('IBKR_HOST', '127.0.0.1'),
    'port': int(os.getenv('IBKR_PORT', '7496')),
    'client_id': int(os.getenv('IBKR_CLIENT_ID', '0'))
}

# 市场数据配置
MARKET_DATA_CONFIG = {
    'default_period': '1y',
    'available_periods': ['1mo', '3mo', '6mo', '1y', '2y', '5y', 'max'],
    'default_interval': '1d',
    'available_intervals': ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo']
}

# 技术指标配置
TECHNICAL_INDICATORS = {
    'MA': {
        'short': 20,
        'long': 50,
        'longest': 200
    },
    'RSI': {
        'period': 14,
        'overbought': 70,
        'oversold': 30
    },
    'MACD': {
        'fast': 12,
        'slow': 26,
        'signal': 9
    },
    'BB': {
        'period': 20,
        'std_dev': 2
    },
    'ATR': {
        'period': 14
    },
    'Stochastic': {
        'k_period': 14,
        'd_period': 3,
        'smooth_period': 3
    }
}

# 交易策略配置
TRADING_STRATEGY = {
    'initial_capital': float(os.getenv('INITIAL_CAPITAL', '100000.0')),
    'position_size': float(os.getenv('POSITION_SIZE', '0.1')),  # 每次交易使用资金比例
    'stop_loss': float(os.getenv('STOP_LOSS', '0.02')),    # 止损比例
    'take_profit': float(os.getenv('TAKE_PROFIT', '0.05')),  # 止盈比例
    'max_positions': int(os.getenv('MAX_POSITIONS', '5')),    # 最大持仓数量
    'commission_rate': float(os.getenv('COMMISSION_RATE', '0.001')),  # 手续费率
    'slippage': float(os.getenv('SLIPPAGE', '0.001')),  # 滑点
    'strategies': {
        'MA_CROSS': {
            'enabled': True,
            'weight': 1.0
        },
        'RSI': {
            'enabled': True,
            'weight': 1.0
        },
        'MACD': {
            'enabled': True,
            'weight': 1.0
        },
        'BOLLINGER_BANDS': {
            'enabled': True,
            'weight': 1.0
        },
        'STOCHASTIC': {
            'enabled': True,
            'weight': 1.0
        },
        'TREND_FOLLOWING': {
            'enabled': True,
            'weight': 1.5,
            'atr_threshold': 0.1
        },
        'MOMENTUM': {
            'enabled': True,
            'weight': 1.2,
            'momentum_window': 10,
            'rsi_threshold': {
                'oversold': 30,
                'overbought': 70
            },
            'stoch_threshold': {
                'oversold': 20,
                'overbought': 80
            }
        },
        'VOLATILITY_BREAKOUT': {
            'enabled': True,
            'weight': 1.3,
            'atr_change_threshold': 0.1,
            'bb_period': 20,
            'bb_std': 2
        },
        'VOLUME_PRICE': {
            'enabled': True,
            'weight': 1.1,
            'volume_ma_period': 20,
            'volume_std_multiplier': 2,
            'price_ma_period': 20
        },
        'SUPPORT_RESISTANCE': {
            'enabled': True,
            'weight': 1.4,
            'level_threshold': 0.01,
            'min_touches': 3
        },
        'PATTERN_RECOGNITION': {
            'enabled': True,
            'weight': 1.2,
            'patterns': {
                'double_top': True,
                'double_bottom': True,
                'head_and_shoulders': True,
                'inverse_head_and_shoulders': True,
                'triangle': True,
                'wedge': True,
                'flag': True,
                'pennant': True
            }
        }
    }
}

# 风险管理配置
RISK_MANAGEMENT_CONFIG = {
    # 风险限制
    'max_var': 0.02,  # 最大VaR
    'max_volatility': 0.3,  # 最大波动率
    'min_liquidity': 0.5,  # 最小流动性得分
    'max_concentration': 0.2,  # 最大集中度
    'max_systematic_risk': 1.5,  # 最大系统性风险
    'max_total_risk': 0.7,  # 最大总风险
    
    # 风险参数
    'risk_free_rate': 0.02,  # 无风险利率
    'confidence_level': 0.95,  # 置信水平
    'var_window': 252,  # VaR计算窗口
    'volatility_window': 20,  # 波动率计算窗口
    
    # 风险权重
    'risk_weights': {
        'position_risk': 0.3,
        'market_risk': 0.2,
        'liquidity_risk': 0.2,
        'concentration_risk': 0.15,
        'systematic_risk': 0.15
    },
    
    # 风险等级阈值
    'risk_levels': {
        'low': 0.3,
        'medium': 0.6,
        'high': 0.8,
        'critical': 0.9
    },
    
    # 风险监控设置
    'monitoring': {
        'check_interval': 300,  # 检查间隔（秒）
        'alert_threshold': 0.7,  # 告警阈值
        'max_alerts': 100,  # 最大告警数量
        'alert_retention_days': 30  # 告警保留天数
    },
    
    # 风险报告设置
    'report': {
        'output_dir': 'risk_reports',
        'chart_format': 'html',
        'include_charts': True,
        'include_tables': True,
        'retention_days': 30  # 报告保留天数
    }
}

# 监控配置
MONITOR_CONFIG = {
    # 告警阈值
    'alert_thresholds': {
        'total_return': {
            'min': -0.1,  # 最小总收益率
            'max': 0.5    # 最大总收益率
        },
        'sharpe_ratio': {
            'min': 0.5,   # 最小夏普比率
            'max': 5.0    # 最大夏普比率
        },
        'max_drawdown': {
            'min': 0.0,   # 最小回撤
            'max': 0.2    # 最大回撤
        },
        'win_rate': {
            'min': 0.4,   # 最小胜率
            'max': 0.8    # 最大胜率
        },
        'profit_factor': {
            'min': 1.2,   # 最小盈亏比
            'max': 5.0    # 最大盈亏比
        }
    },
    
    # 数据更新超时设置
    'data_update_timeout': 300,  # 数据更新超时时间（秒）
    'error_retry_interval': 60,  # 错误重试间隔（秒）
    
    # 通知设置
    'notifications': {
        'email': {
            'enabled': True,
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587,
            'sender': 'your-email@gmail.com',
            'recipients': ['recipient1@example.com', 'recipient2@example.com']
        },
        'webhook': {
            'enabled': True,
            'url': 'https://your-webhook-url.com/endpoint',
            'headers': {
                'Content-Type': 'application/json'
            }
        },
        'telegram': {
            'enabled': True,
            'bot_token': 'your-bot-token',
            'chat_id': 'your-chat-id'
        }
    },
    
    # 报告设置
    'report': {
        'output_dir': 'monitor_reports',
        'chart_format': 'html',
        'include_charts': True,
        'include_tables': True,
        'max_alerts': 100,  # 最大告警记录数
        'retention_days': 30  # 报告保留天数
    },
    
    # 性能指标设置
    'performance_metrics': {
        'risk_free_rate': 0.02,  # 无风险利率
        'trading_days': 252,      # 年化交易天数
        'min_data_points': 20,    # 最小数据点数
        'confidence_level': 0.95  # 置信水平
    },
    
    # 系统监控设置
    'system_monitoring': {
        'cpu_threshold': 80,      # CPU使用率阈值（%）
        'memory_threshold': 80,   # 内存使用率阈值（%）
        'disk_threshold': 80,     # 磁盘使用率阈值（%）
        'check_interval': 300     # 检查间隔（秒）
    }
}

# 行业ETF映射
INDUSTRY_ETF_MAPPING = {
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

# 主要市场指数
MARKET_INDICES = {
    'S&P 500': '^GSPC',
    'Dow Jones': '^DJI',
    'NASDAQ': '^IXIC',
    'Russell 2000': '^RUT',
    'VIX': '^VIX'
}

# 日志配置
LOG_CONFIG = {
    'level': os.getenv('LOG_LEVEL', 'INFO'),
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': os.getenv('LOG_FILE', 'trading_system.log')
}

# API配置
API_CONFIG = {
    'host': os.getenv('API_HOST', '0.0.0.0'),
    'port': int(os.getenv('API_PORT', '8000')),
    'debug': os.getenv('API_DEBUG', 'False').lower() == 'true'
}

# 数据库配置
DATABASE_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5432'),
    'database': os.getenv('DB_NAME', 'trading_system'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', 'postgres')
}

# 邮件配置
EMAIL_CONFIG = {
    'smtp_server': os.getenv('SMTP_SERVER', 'smtp.gmail.com'),
    'smtp_port': int(os.getenv('SMTP_PORT', '587')),
    'username': os.getenv('EMAIL_USERNAME', ''),
    'password': os.getenv('EMAIL_PASSWORD', ''),
    'sender': os.getenv('EMAIL_SENDER', ''),
    'recipient': os.getenv('EMAIL_RECIPIENT', '')
}

# 回测配置
BACKTEST_CONFIG = {
    'initial_capital': float(os.getenv('INITIAL_CAPITAL', '100000.0')),
    'commission_rate': float(os.getenv('COMMISSION_RATE', '0.001')),
    'slippage': float(os.getenv('SLIPPAGE', '0.001'))
}

# 图表配置
CHART_CONFIG = {
    'default_period': '1y',
    'default_interval': '1d',
    'chart_height': 600,
    'chart_width': 800,
    'theme': 'plotly_white'
}

# 缓存配置
CACHE_CONFIG = {
    'timeout': int(os.getenv('CACHE_TIMEOUT', '300')),  # 缓存超时时间（秒）
    'max_size': int(os.getenv('CACHE_MAX_SIZE', '1000'))  # 最大缓存条目数
}

# 交易配置
TRADING_CONFIG = {
    'initial_capital': float(os.getenv('INITIAL_CAPITAL', '100000.0')),
    'commission_rate': float(os.getenv('COMMISSION_RATE', '0.0003')),
    'slippage': float(os.getenv('SLIPPAGE', '0.0001')),
    'max_positions': int(os.getenv('MAX_POSITIONS', '10')),
    'position_size_limit': float(os.getenv('POSITION_SIZE_LIMIT', '0.1'))  # 单个持仓最大比例
}

# 风险控制配置
RISK_CONFIG = {
    'max_drawdown': float(os.getenv('MAX_DRAWDOWN', '0.2')),  # 最大回撤限制
    'var_limit': float(os.getenv('VAR_LIMIT', '0.02')),  # VaR限制
    'volatility_limit': float(os.getenv('VOLATILITY_LIMIT', '0.3')),  # 波动率限制
    'beta_limit': float(os.getenv('BETA_LIMIT', '1.5')),  # Beta限制
    'correlation_limit': float(os.getenv('CORRELATION_LIMIT', '0.7'))  # 相关性限制
}

# 策略配置
STRATEGY_CONFIG = {
    'default_strategy': os.getenv('DEFAULT_STRATEGY', 'mean_reversion'),
    'backtest_start_date': os.getenv('BACKTEST_START_DATE', '2023-01-01'),
    'backtest_end_date': os.getenv('BACKTEST_END_DATE', '2023-12-31'),
    'rebalance_frequency': os.getenv('REBALANCE_FREQUENCY', 'daily')
}

# 报告配置
REPORT_CONFIG = {
    'output_dir': os.getenv('REPORT_OUTPUT_DIR', 'reports'),
    'chart_format': os.getenv('REPORT_CHART_FORMAT', 'html'),
    'include_charts': os.getenv('REPORT_INCLUDE_CHARTS', 'True').lower() == 'true',
    'include_tables': os.getenv('REPORT_INCLUDE_TABLES', 'True').lower() == 'true',
    'retention_days': int(os.getenv('REPORT_RETENTION_DAYS', '90'))  # 报告保留天数
} 