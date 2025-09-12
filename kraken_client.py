"""Kraken Pro API client for trading operations."""

import time
import hmac
import hashlib
import base64
import requests
from typing import Dict, List, Optional, Any
from urllib.parse import urlencode
import ccxt
from loguru import logger
from config import config

class KrakenClient:
    """Client for interacting with Kraken Pro API."""
    
    def __init__(self):
        self.api_key = config.kraken_api_key
        self.secret_key = config.kraken_secret_key
        self.sandbox = config.kraken_sandbox
        
        # Initialize CCXT exchange
        self.exchange = ccxt.kraken({
            'apiKey': self.api_key,
            'secret': self.secret_key,
            'sandbox': self.sandbox,
            'enableRateLimit': True,
        })
        
        self.base_url = "https://api.kraken.com" if not self.sandbox else "https://api-sandbox.kraken.com"
        
    def _generate_signature(self, urlpath: str, data: Dict[str, Any], nonce: str) -> str:
        """Generate API signature for authenticated requests."""
        postdata = urlencode(data)
        encoded = (nonce + postdata).encode()
        message = urlpath.encode() + hashlib.sha256(encoded).digest()
        
        mac = hmac.new(base64.b64decode(self.secret_key), message, hashlib.sha512)
        sigdigest = base64.b64encode(mac.digest())
        return sigdigest.decode()
    
    def _make_request(self, endpoint: str, data: Optional[Dict] = None, private: bool = False) -> Dict:
        """Make HTTP request to Kraken API."""
        url = f"{self.base_url}{endpoint}"
        
        if private:
            nonce = str(int(1000 * time.time()))
            if data is None:
                data = {}
            data['nonce'] = nonce
            
            headers = {
                'API-Key': self.api_key,
                'API-Sign': self._generate_signature(endpoint, data, nonce)
            }
            
            response = requests.post(url, data=data, headers=headers)
        else:
            response = requests.get(url, params=data or {})
        
        response.raise_for_status()
        return response.json()
    
    def get_account_balance(self) -> Dict[str, float]:
        """Get account balance."""
        try:
            result = self._make_request('/0/private/Balance', private=True)
            return result.get('result', {})
        except Exception as e:
            logger.error(f"Error getting account balance: {e}")
            return {}
    
    def get_ticker(self, pair: str) -> Dict[str, Any]:
        """Get ticker information for a trading pair."""
        try:
            result = self._make_request('/0/public/Ticker', {'pair': pair})
            return result.get('result', {})
        except Exception as e:
            logger.error(f"Error getting ticker for {pair}: {e}")
            return {}
    
    def get_ohlc_data(self, pair: str, interval: int = 1, since: Optional[int] = None) -> List[Dict]:
        """Get OHLC (candlestick) data."""
        try:
            params = {'pair': pair, 'interval': interval}
            if since:
                params['since'] = since
                
            result = self._make_request('/0/public/OHLC', params)
            data = result.get('result', {}).get(pair, [])
            
            # Convert to structured format
            ohlc_data = []
            for item in data:
                ohlc_data.append({
                    'timestamp': int(item[0]),
                    'open': float(item[1]),
                    'high': float(item[2]),
                    'low': float(item[3]),
                    'close': float(item[4]),
                    'volume': float(item[6])
                })
            
            return ohlc_data
        except Exception as e:
            logger.error(f"Error getting OHLC data for {pair}: {e}")
            return []
    
    def place_market_order(self, pair: str, side: str, amount: float) -> Optional[Dict]:
        """Place a market order."""
        try:
            data = {
                'pair': pair,
                'type': side.lower(),
                'ordertype': 'market',
                'volume': str(amount)
            }
            
            result = self._make_request('/0/private/AddOrder', data, private=True)
            return result.get('result', {})
        except Exception as e:
            logger.error(f"Error placing market order: {e}")
            return None
    
    def place_limit_order(self, pair: str, side: str, amount: float, price: float) -> Optional[Dict]:
        """Place a limit order."""
        try:
            data = {
                'pair': pair,
                'type': side.lower(),
                'ordertype': 'limit',
                'volume': str(amount),
                'price': str(price)
            }
            
            result = self._make_request('/0/private/AddOrder', data, private=True)
            return result.get('result', {})
        except Exception as e:
            logger.error(f"Error placing limit order: {e}")
            return None
    
    def get_open_orders(self) -> List[Dict]:
        """Get all open orders."""
        try:
            result = self._make_request('/0/private/OpenOrders', private=True)
            return result.get('result', {}).get('open', {})
        except Exception as e:
            logger.error(f"Error getting open orders: {e}")
            return []
    
    def cancel_order(self, txid: str) -> bool:
        """Cancel an order by transaction ID."""
        try:
            data = {'txid': txid}
            result = self._make_request('/0/private/CancelOrder', data, private=True)
            return result.get('result', {}).get('count', 0) > 0
        except Exception as e:
            logger.error(f"Error canceling order {txid}: {e}")
            return False
    
    def get_trade_history(self, pair: Optional[str] = None) -> List[Dict]:
        """Get trade history."""
        try:
            data = {}
            if pair:
                data['pair'] = pair
                
            result = self._make_request('/0/private/TradesHistory', data, private=True)
            return result.get('result', {}).get('trades', {})
        except Exception as e:
            logger.error(f"Error getting trade history: {e}")
            return []
    
    def get_server_time(self) -> Dict[str, int]:
        """Get server time."""
        try:
            result = self._make_request('/0/public/Time')
            return result.get('result', {})
        except Exception as e:
            logger.error(f"Error getting server time: {e}")
            return {}