import logging
import os
from datetime import datetime
from config import LOG_CONFIG

def setup_logger(name):
    """设置日志记录器"""
    # 创建日志目录
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 创建日志文件名
    log_file = os.path.join(log_dir, f'{name}_{datetime.now().strftime("%Y%m%d")}.log')

    # 创建日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(LOG_CONFIG['level'])

    # 创建文件处理器
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(LOG_CONFIG['level'])

    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(LOG_CONFIG['level'])

    # 创建格式化器
    formatter = logging.Formatter(LOG_CONFIG['format'])
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 添加处理器到日志记录器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

def get_logger(name):
    """获取日志记录器"""
    return logging.getLogger(name)

# 创建默认日志记录器
default_logger = setup_logger('trading_platform') 