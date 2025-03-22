# Use Python 3.8 slim image
FROM python:3.8-slim

WORKDIR /app

# 复制项目文件
COPY . .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 创建数据和日志目录
RUN mkdir -p /app/data /app/logs

# 暴露端口
EXPOSE 8000

# 设置入口命令
CMD ["python", "-m", "atrade.cli", "web", "--host", "0.0.0.0", "--port", "8000"] 