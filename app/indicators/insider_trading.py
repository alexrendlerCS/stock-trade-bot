import pandas as pd
import requests
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional
from ..core.config import settings

logger = logging.getLogger(__name__)

class InsiderTracker:
    """Track and analyze insider trading activity"""
    
    def __init__(self):
        self.sec_endpoint = "https://www.sec.gov/cgi-bin/own-disp"
        self.cache_duration = timedelta(hours=12)  # Cache insider data for 12 hours
        self.cache = {}  # {symbol: (timestamp, data)}
        
    def get_insider_data(self, symbol: str) -> Optional[Dict]:
        """
        Get recent insider trading data for a symbol
        Returns metrics about insider activity
        """
        try:
            # Check cache first
            if symbol in self.cache:
                timestamp, data = self.cache[symbol]
                if datetime.now() - timestamp < self.cache_duration:
                    return data
            
            # Fetch new data from SEC
            params = {
                'action': 'getissuer',
                'CIK': self._get_cik(symbol),
                'type': 'getissuer',
                'dateb': '',
                'owner': 'include',
                'start': '0',
                'count': '40'  # Last 40 transactions
            }
            
            # Make SEC request
            response = requests.get(self.sec_endpoint, params=params)
            if response.status_code != 200:
                logger.error(f"SEC API error for {symbol}: {response.status_code}")
                return None
            
            # Parse transactions
            transactions = self._parse_transactions(response.text)
            
            # Calculate metrics
            metrics = self._calculate_metrics(transactions)
            
            # Cache the results
            self.cache[symbol] = (datetime.now(), metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error fetching insider data for {symbol}: {str(e)}")
            return None
    
    def _get_cik(self, symbol: str) -> str:
        """Get CIK number for a symbol"""
        # This would normally use a CIK database or API
        # For now, return a placeholder that will be replaced
        return "0000320193"  # Example CIK (Apple)
    
    def _parse_transactions(self, html_content: str) -> List[Dict]:
        """Parse SEC form 4 transactions from HTML"""
        transactions = []
        
        try:
            # This would normally use BeautifulSoup or similar to parse HTML
            # For now, return placeholder data
            transactions = [
                {
                    'date': datetime.now() - timedelta(days=1),
                    'insider_name': 'John Doe',
                    'title': 'Director',
                    'transaction_type': 'P',  # Purchase
                    'shares': 1000,
                    'price': 150.00
                }
            ]
        except Exception as e:
            logger.error(f"Error parsing transactions: {str(e)}")
        
        return transactions
    
    def _calculate_metrics(self, transactions: List[Dict]) -> Dict:
        """Calculate insider trading metrics"""
        try:
            # Initialize metrics
            total_buy_volume = 0
            total_sell_volume = 0
            buy_transactions = 0
            sell_transactions = 0
            director_buys = 0
            officer_buys = 0
            
            # Last 30 days
            recent_date = datetime.now() - timedelta(days=30)
            
            for tx in transactions:
                if tx['date'] < recent_date:
                    continue
                    
                volume = tx['shares'] * tx['price']
                
                if tx['transaction_type'] in ['P', 'B']:  # Purchase/Buy
                    total_buy_volume += volume
                    buy_transactions += 1
                    if 'Director' in tx['title']:
                        director_buys += 1
                    elif 'Officer' in tx['title']:
                        officer_buys += 1
                elif tx['transaction_type'] in ['S']:  # Sell
                    total_sell_volume += volume
                    sell_transactions += 1
            
            # Calculate buy/sell ratio
            total_volume = total_buy_volume + total_sell_volume
            if total_volume > 0:
                buy_ratio = total_buy_volume / total_volume
            else:
                buy_ratio = 0.5  # Neutral if no activity
            
            # Calculate significance score (0 to 1)
            significance = min(1.0, total_volume / 10000000)  # Cap at $10M
            
            # Calculate confidence adjustment (-0.1 to +0.1)
            if buy_ratio > 0.7 and significance > 0.3:
                confidence_adj = 0.1  # Strong buying
            elif buy_ratio < 0.3 and significance > 0.3:
                confidence_adj = -0.1  # Strong selling
            elif buy_ratio > 0.6:
                confidence_adj = 0.05  # Moderate buying
            elif buy_ratio < 0.4:
                confidence_adj = -0.05  # Moderate selling
            else:
                confidence_adj = 0.0  # Neutral
            
            return {
                'buy_volume': total_buy_volume,
                'sell_volume': total_sell_volume,
                'buy_ratio': buy_ratio,
                'significance': significance,
                'confidence_adjustment': confidence_adj,
                'buy_transactions': buy_transactions,
                'sell_transactions': sell_transactions,
                'director_buys': director_buys,
                'officer_buys': officer_buys
            }
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            return {
                'buy_ratio': 0.5,
                'significance': 0.0,
                'confidence_adjustment': 0.0
            }
    
    def get_confidence_adjustment(self, symbol: str, direction: str) -> float:
        """
        Get confidence adjustment based on insider trading
        Returns adjustment between -0.1 and +0.1
        """
        try:
            metrics = self.get_insider_data(symbol)
            if not metrics:
                return 0.0
            
            adj = metrics['confidence_adjustment']
            
            # Align adjustment with trade direction
            if direction == 'LONG':
                return adj  # Positive for buying, negative for selling
            else:  # SHORT
                return -adj  # Reverse for short positions
                
        except Exception as e:
            logger.error(f"Error getting confidence adjustment: {str(e)}")
            return 0.0 