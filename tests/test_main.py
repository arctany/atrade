import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_health_check():
    """测试健康检查接口"""
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert "timestamp" in response.json()
    assert "components" in response.json()

def test_chat_endpoint():
    """测试聊天接口"""
    response = client.post(
        "/api/chat",
        json={"message": "Hello"}
    )
    assert response.status_code == 200
    assert "response" in response.json()

def test_risk_analysis():
    """测试风险分析接口"""
    response = client.get("/api/risk_analysis")
    assert response.status_code == 200

def test_performance_metrics():
    """测试性能指标接口"""
    response = client.get("/api/performance_metrics")
    assert response.status_code == 200

def test_backtest_results():
    """测试回测结果接口"""
    response = client.get("/api/backtest_results")
    assert response.status_code == 200 