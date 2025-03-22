import click
import logging
from pathlib import Path
from typing import Optional

from atrade.core.agent import TradingAgent
from atrade.config.settings import load_config
from atrade.utils.logger import setup_logger

@click.group()
def cli():
    """Atrade - Intelligent Trading System"""
    pass

@cli.command()
@click.option('--config', '-c', type=click.Path(exists=True),
              help='Path to configuration file')
@click.option('--log-level', '-l', default='INFO',
              type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']),
              help='Logging level')
def start(config: Optional[str], log_level: str):
    """Start the trading system"""
    # Setup logging
    setup_logger(log_level)
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        config_path = Path(config) if config else None
        settings = load_config(config_path)
        
        # Initialize trading agent
        agent = TradingAgent(settings)
        
        # Start trading
        agent.start()
        
    except Exception as e:
        logger.error(f"Failed to start trading system: {str(e)}")
        raise click.Abort()

@cli.command()
@click.option('--config', '-c', type=click.Path(exists=True),
              help='Path to configuration file')
@click.option('--output', '-o', type=click.Path(),
              help='Output directory for backtest results')
def backtest(config: Optional[str], output: Optional[str]):
    """Run backtest simulation"""
    # Setup logging
    setup_logger('INFO')
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        config_path = Path(config) if config else None
        settings = load_config(config_path)
        
        # Initialize backtest
        from atrade.core.backtest import Backtest
        backtest = Backtest(settings)
        
        # Run backtest
        results = backtest.run()
        
        # Save results
        if output:
            output_path = Path(output)
            backtest.save_results(results, output_path)
            logger.info(f"Backtest results saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Backtest failed: {str(e)}")
        raise click.Abort()

@cli.command()
@click.option('--config', '-c', type=click.Path(exists=True),
              help='Path to configuration file')
def optimize(config: Optional[str]):
    """Optimize strategy parameters"""
    # Setup logging
    setup_logger('INFO')
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        config_path = Path(config) if config else None
        settings = load_config(config_path)
        
        # Initialize optimizer
        from atrade.core.optimizer import StrategyOptimizer
        optimizer = StrategyOptimizer(settings)
        
        # Run optimization
        results = optimizer.optimize()
        
        # Save results
        optimizer.save_results(results)
        logger.info("Strategy optimization completed")
        
    except Exception as e:
        logger.error(f"Strategy optimization failed: {str(e)}")
        raise click.Abort()

if __name__ == '__main__':
    cli() 