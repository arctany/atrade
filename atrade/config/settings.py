import os
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
from pydantic import BaseModel

class DatabaseSettings(BaseModel):
    host: str = "localhost"
    port: int = 5432
    name: str = "trading_system"
    user: str = "postgres"
    password: str = "postgres"

class APISettings(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False

class TradingSettings(BaseModel):
    initial_capital: float = 100000.0
    max_position_size: float = 0.1
    stop_loss_pct: float = 0.02
    take_profit_pct: float = 0.05

class RiskSettings(BaseModel):
    max_drawdown: float = 0.1
    max_leverage: float = 2.0
    max_correlation: float = 0.7

class LoggingSettings(BaseModel):
    level: str = "INFO"
    file: str = "logs/trading_system.log"

class RedisSettings(BaseModel):
    host: str = "localhost"
    port: int = 6379
    db: int = 0

class SecuritySettings(BaseModel):
    jwt_secret_key: str = "your-secret-key-here"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30

class ExternalAPISettings(BaseModel):
    alpha_vantage_api_key: str = "your-api-key-here"
    polygon_api_key: str = "your-api-key-here"

class Settings(BaseModel):
    database: DatabaseSettings = DatabaseSettings()
    api: APISettings = APISettings()
    trading: TradingSettings = TradingSettings()
    risk: RiskSettings = RiskSettings()
    logging: LoggingSettings = LoggingSettings()
    redis: RedisSettings = RedisSettings()
    security: SecuritySettings = SecuritySettings()
    external_api: ExternalAPISettings = ExternalAPISettings()

def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load configuration from file and environment variables"""
    settings_dict: Dict[str, Any] = {}
    
    # Load from config file if provided
    if config_path and config_path.exists():
        with open(config_path, 'r') as f:
            settings_dict = yaml.safe_load(f) or {}
    
    # Override with environment variables
    env_prefix = "ATRADE_"
    for key, value in os.environ.items():
        if key.startswith(env_prefix):
            # Convert ATRADE_DATABASE_HOST to database.host
            config_key = key[len(env_prefix):].lower().replace('_', '.')
            settings_dict[config_key] = value
    
    # Return settings as dictionary
    return settings_dict

def get_default_config_path() -> Path:
    """Get the default configuration file path"""
    return Path(__file__).parent / "config.yaml" 