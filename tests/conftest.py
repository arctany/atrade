import os
import pytest
from typing import Generator
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from redis import Redis

from atrade.db.base import Base
from atrade.api.main import app
from atrade.core.config import Settings

# Test settings
TEST_SETTINGS = Settings(
    DATABASE_URL="postgresql://postgres:postgres@localhost:5432/trading_system_test",
    REDIS_URL="redis://localhost:6379/1",
    DEBUG=True,
    TESTING=True
)

@pytest.fixture(scope="session")
def settings() -> Settings:
    return TEST_SETTINGS

@pytest.fixture(scope="session")
def engine():
    engine = create_engine(TEST_SETTINGS.DATABASE_URL)
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)

@pytest.fixture(scope="function")
def db_session(engine) -> Generator[Session, None, None]:
    connection = engine.connect()
    transaction = connection.begin()
    session = sessionmaker(bind=connection)()
    
    yield session
    
    session.close()
    transaction.rollback()
    connection.close()

@pytest.fixture(scope="function")
def redis_client(settings) -> Generator[Redis, None, None]:
    client = Redis.from_url(settings.REDIS_URL)
    yield client
    client.flushdb()

@pytest.fixture(scope="function")
def client(db_session, redis_client) -> Generator[TestClient, None, None]:
    app.dependency_overrides = {
        "get_db": lambda: db_session,
        "get_redis": lambda: redis_client,
        "get_settings": lambda: TEST_SETTINGS
    }
    
    with TestClient(app) as test_client:
        yield test_client
    
    app.dependency_overrides.clear()

@pytest.fixture(scope="function")
def test_user():
    return {
        "username": "testuser",
        "email": "test@example.com",
        "password": "testpassword123"
    }

@pytest.fixture(scope="function")
def test_token(client, test_user):
    response = client.post("/api/auth/token", data={
        "username": test_user["username"],
        "password": test_user["password"]
    })
    return response.json()["access_token"]

@pytest.fixture(scope="function")
def authorized_client(client, test_token) -> Generator[TestClient, None, None]:
    client.headers = {
        **client.headers,
        "Authorization": f"Bearer {test_token}"
    }
    yield client
    client.headers = {} 