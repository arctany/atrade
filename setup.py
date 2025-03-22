from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="atrade",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="An intelligent trading system based on large language models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/atrade",
    packages=find_packages(),
    package_data={
        "atrade": ["config/*.yaml", "data/*.csv"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "python-dotenv>=0.19.0",
        "requests>=2.26.0",
        "sqlalchemy>=1.4.0",
        "psycopg2-binary>=2.9.0",
        "fastapi>=0.68.0",
        "uvicorn>=0.15.0",
        "pydantic>=1.8.0",
        "scikit-learn>=0.24.0",
        "optuna>=2.10.0",
        "xgboost>=1.5.0",
        "lightgbm>=3.3.0",
        "plotly>=5.3.0",
        "dash>=2.0.0",
        "dash-bootstrap-components>=1.0.0",
        "ib_insync>=0.9.70",
        "ta-lib>=0.4.24",
        "aiohttp>=3.8.0",
        "asyncio>=3.4.3",
        "python-jose>=3.3.0",
        "passlib>=1.7.4",
        "python-multipart>=0.0.5",
        "python-telegram-bot>=20.0",
        "streamlit>=1.0.0",
        "yfinance>=0.1.63",
        "ta>=0.7.0",
        "python-dateutil>=2.8.2",
        "pytz>=2021.1",
        "statsmodels>=0.13.0",
        "scipy>=1.7.0",
        "tensorflow>=2.6.0",
        "keras>=2.6.0",
        "bcrypt>=3.2.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.5",
            "pytest-cov>=2.12.1",
            "black>=21.7b0",
            "flake8>=3.9.2",
            "mypy>=0.910",
            "isort>=5.9.3",
            "pre-commit>=2.15.0",
        ],
        "docs": [
            "sphinx>=4.2.0",
            "sphinx-rtd-theme>=0.5.2",
        ],
    },
    entry_points={
        "console_scripts": [
            "atrade=atrade.cli:main",
        ],
    },
) 