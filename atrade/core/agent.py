"""
Trading Agent Module
"""
import logging
import time
from typing import Dict, Any


class TradingAgent:
    """
    Trading Agent responsible for executing trading strategies
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the trading agent
        
        Args:
            config: Configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.running = False
        self.logger.info("Trading agent initialized")
        
    def start(self):
        """
        Start the trading agent
        """
        self.running = True
        self.logger.info("Trading agent started")
        
        try:
            # Simulate running
            while self.running:
                self.logger.info("Trading agent running... Press Ctrl+C to stop")
                time.sleep(10)
        except KeyboardInterrupt:
            self.stop()
            
    def stop(self):
        """
        Stop the trading agent
        """
        self.running = False
        self.logger.info("Trading agent stopped") 