# Atrade Project Structure

This document describes the internal structure of the Atrade project.

## Directory Structure

```
atrade/
├── api/              # API endpoints and web interface
│   ├── __init__.py
│   ├── routes/       # API routes
│   └── schemas/      # Pydantic models for API
├── core/             # Core trading functionality
│   ├── __init__.py
│   ├── agent.py      # Trading agent implementation
│   ├── broker.py     # Broker interface
│   └── engine.py     # Trading engine
├── strategies/       # Trading strategies
│   ├── __init__.py
│   ├── base.py      # Base strategy class
│   └── models/      # Strategy implementations
├── risk/            # Risk management
│   ├── __init__.py
│   ├── manager.py   # Risk manager
│   └── models/      # Risk models
├── analysis/        # Market analysis
│   ├── __init__.py
│   ├── technical.py # Technical analysis
│   └── fundamental.py # Fundamental analysis
├── utils/           # Utility functions
│   ├── __init__.py
│   ├── indicators.py # Technical indicators
│   └── helpers.py   # Helper functions
├── config/          # Configuration
│   ├── __init__.py
│   └── settings.py  # Settings management
├── data/            # Data management
│   ├── __init__.py
│   ├── fetcher.py   # Data fetching
│   └── storage.py   # Data storage
└── tests/           # Tests
    ├── __init__.py
    ├── unit/        # Unit tests
    ├── integration/ # Integration tests
    └── performance/ # Performance tests
```

## Module Descriptions

### API
- `api/`: Contains all API-related code
- `routes/`: API endpoint definitions
- `schemas/`: Data validation and serialization

### Core
- `core/`: Core trading functionality
- `agent.py`: Main trading agent implementation
- `broker.py`: Broker interface for executing trades
- `engine.py`: Trading engine for managing orders

### Strategies
- `strategies/`: Trading strategy implementations
- `base.py`: Base strategy class with common functionality
- `models/`: Individual strategy implementations

### Risk
- `risk/`: Risk management functionality
- `manager.py`: Risk management system
- `models/`: Risk calculation models

### Analysis
- `analysis/`: Market analysis tools
- `technical.py`: Technical analysis functions
- `fundamental.py`: Fundamental analysis functions

### Utils
- `utils/`: Utility functions and helpers
- `indicators.py`: Technical indicators
- `helpers.py`: General helper functions

### Config
- `config/`: Configuration management
- `settings.py`: Application settings

### Data
- `data/`: Data management
- `fetcher.py`: Market data fetching
- `storage.py`: Data storage and retrieval

### Tests
- `tests/`: All test files
- `unit/`: Unit tests for individual components
- `integration/`: Integration tests
- `performance/`: Performance tests

## Development Guidelines

1. All new code should be placed in the appropriate module
2. Each module should have its own `__init__.py` file
3. Tests should be placed in the corresponding test directory
4. Configuration should be managed through the config module
5. Data access should go through the data module
6. API endpoints should be defined in the api module 