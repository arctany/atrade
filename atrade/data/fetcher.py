import yfinance as yf

class MarketDataFetcher:
    def __init__(self):
        pass

    def get_historical_data(self, symbol: str, period: str = "1mo"):
        """Fetch historical market data for a given symbol and period using Yahoo Finance."""
        stock = yf.Ticker(symbol)
        hist = stock.history(period=period)
        return hist

# Example usage
if __name__ == "__main__":
    fetcher = MarketDataFetcher()
    data = fetcher.get_historical_data("AAPL")
    print(data) 