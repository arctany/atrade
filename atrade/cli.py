#!/usr/bin/env python
"""Command-line interface for the trading system."""
import logging
import click
from pathlib import Path

from atrade.core.agent import TradingAgent
from atrade.config.settings import load_config, get_default_config_path

@click.group()
def cli():
    """ATrade: Algorithmic Trading System."""
    pass

@cli.command()
@click.option(
    '--config-file', '-c',
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help='Path to configuration file.'
)
@click.option(
    '--log-level', '-l',
    type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']),
    default='INFO',
    help='Set the logging level.'
)
def start(config_file, log_level):
    """Start the trading agent."""
    setup_logging(log_level)
    
    if not config_file:
        config_file = get_default_config_path()
    
    config = load_config(config_file)
    agent = TradingAgent(config)
    agent.start()

@cli.command()
@click.option(
    '--config-file', '-c',
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help='Path to configuration file.'
)
@click.option(
    '--log-level', '-l',
    type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']),
    default='INFO',
    help='Set the logging level.'
)
@click.option(
    '--host', '-h',
    default='0.0.0.0',
    help='Host to bind the web server to.'
)
@click.option(
    '--port', '-p',
    default=8000,
    type=int,
    help='Port to bind the web server to.'
)
def web(config_file, log_level, host, port):
    """Start the web interface."""
    setup_logging(log_level)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting web interface on {host}:{port}")
    
    if not config_file:
        config_file = get_default_config_path()
        logger.info(f"Using default config file: {config_file}")
    else:
        logger.info(f"Using config file: {config_file}")
    
    try:
        config = load_config(config_file)
        from atrade.api.server import start_server
        logger.info("Web server starting...")
        start_server(host=host, port=port)
    except Exception as e:
        logger.error(f"Failed to start web server: {str(e)}")
        raise click.Abort()

@cli.command()
@click.option(
    '--config-file', '-c',
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help='Path to configuration file.'
)
@click.option(
    '--log-level', '-l',
    type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']),
    default='INFO',
    help='Set the logging level.'
)
def backtest(config_file, log_level):
    """Run backtesting of trading strategies."""
    setup_logging(log_level)
    
    if not config_file:
        config_file = get_default_config_path()
    
    config = load_config(config_file)
    click.echo("Backtesting not implemented yet.")

@cli.command()
@click.option(
    '--config-file', '-c',
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help='Path to configuration file.'
)
@click.option(
    '--log-level', '-l',
    type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']),
    default='INFO',
    help='Set the logging level.'
)
def optimize(config_file, log_level):
    """Optimize trading strategy parameters."""
    setup_logging(log_level)
    
    if not config_file:
        config_file = get_default_config_path()
    
    config = load_config(config_file)
    click.echo("Optimization not implemented yet.")

def setup_logging(log_level):
    """Set up logging configuration."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )

if __name__ == '__main__':
    cli() 