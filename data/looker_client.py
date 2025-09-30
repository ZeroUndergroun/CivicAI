# data/looker_client.py - FIXED VERSION
import os
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import pandas as pd
from datetime import datetime, timedelta
import json
import hashlib

logger = logging.getLogger(__name__)

# Try different import patterns for Looker SDK
try:
    # Newer Looker SDK versions
    from looker_sdk.sdk.api40 import methods, models
    looker_sdk_available = True
except ImportError:
    try:
        # Older Looker SDK versions
        import looker_sdk
        methods = looker_sdk.methods
        models = looker_sdk.models
        looker_sdk_available = True
    except (ImportError, AttributeError):
        looker_sdk_available = False
        logger.warning("Looker SDK not available, running in fallback mode")

@dataclass
class LookerConfig:
    """Configuration for Looker integration"""
    base_url: str = os.getenv("LOOKER_BASE_URL", "")
    client_id: str = os.getenv("LOOKER_CLIENT_ID", "")
    client_secret: str = os.getenv("LOOKER_CLIENT_SECRET", "")
    embed_secret: str = os.getenv("LOOKER_EMBED_SECRET", "")
    timeout: int = 30
    max_retries: int = 3

class LookerClient:
    """
    Enhanced Looker client with graceful fallbacks
    """
    
    def __init__(self, config: LookerConfig = None):
        self.config = config or LookerConfig()
        self.sdk = self._authenticate() if looker_sdk_available else None
        self._dashboard_cache = {}
        
        # Pre-defined dashboard IDs for different query types
        self.dashboard_templates = {
            "covid": os.getenv("LOOKER_COVID_DASHBOARD_ID", ""),
            "income": os.getenv("LOOKER_INCOME_DASHBOARD_ID", ""),
            "comparison": os.getenv("LOOKER_COMPARISON_DASHBOARD_ID", "")
        }
    
    def _authenticate(self):
        """Authenticate with Looker API"""
        try:
            if not all([self.config.base_url, self.config.client_id, self.config.client_secret]):
                logger.warning("Looker credentials incomplete, running in fallback mode")
                return None
            
            # Initialize Looker SDK
            import looker_sdk
            sdk = looker_sdk.init40(
                config_file=looker_sdk.rtl.settings.get_config_file(),
                base_url=self.config.base_url,
                client_id=self.config.client_id,
                client_secret=self.config.client_secret
            )
            
            # Test authentication
            sdk.me()
            logger.info("✅ Looker API authenticated successfully")
            return sdk
            
        except Exception as e:
            logger.error(f"❌ Looker authentication failed: {e}")
            return None
    
    def generate_embed_url(self, 
                          query_type: str, 
                          data: pd.DataFrame = None,
                          filters: Dict[str, Any] = None) -> str:
        """
        Generate embedded Looker URL - SIMPLIFIED VERSION
        """
        # Use cached URL if available
        cache_key = self._generate_cache_key(query_type, filters)
        if cache_key in self._dashboard_cache:
            return self._dashboard_cache[cache_key]
        
        # Always use static URLs for now (simpler for demo)
        embed_url = self._get_static_embed_url(query_type, filters)
        self._dashboard_cache[cache_key] = embed_url
        return embed_url
    
    def _get_static_embed_url(self, query_type: str, filters: Dict[str, Any]) -> str:
        """Fallback to static embedded URLs with parameter passing"""
        base_urls = {
            "covid": os.getenv("LOOKER_COVID_DASHBOARD_URL", "https://lookerstudio.google.com/embed/reporting/1"),
            "income": os.getenv("LOOKER_INCOME_DASHBOARD_URL", "https://lookerstudio.google.com/embed/reporting/1"), 
            "comparison": os.getenv("LOOKER_COMPARISON_DASHBOARD_URL", "https://lookerstudio.google.com/embed/reporting/1")
        }
        
        url = base_urls.get(query_type, base_urls["covid"])
        
        # Add URL parameters for filtering
        if filters:
            param_string = "&".join([f"params={k}:{v}" for k, v in filters.items()])
            url = f"{url}?{param_string}" if param_string else url
        
        return url
    
    def _generate_cache_key(self, query_type: str, filters: Dict[str, Any]) -> str:
        """Generate cache key for embed URLs"""
        key_data = f"{query_type}_{json.dumps(filters, sort_keys=True) if filters else 'no_filters'}"
        return hashlib.md5(key_data.encode()).hexdigest()

# Simplified version that definitely works
class SimpleLookerClient:
    """
    Ultra-simple Looker client that works without any SDK dependencies
    Perfect for your hackathon demo!
    """
    
    def __init__(self):
        self.base_urls = {
            "covid": os.getenv("LOOKER_COVID_DASHBOARD_URL", ""),
            "income": os.getenv("LOOKER_INCOME_DASHBOARD_URL", ""),
            "comparison": os.getenv("LOOKER_COMPARISON_DASHBOARD_URL", "")
        }
        
        # Fallback demo URLs (replace with your actual Looker Studio URLs)
        self.demo_urls = {
            "covid": "https://lookerstudio.google.com/embed/reporting/1",  # Replace with actual
            "income": "https://lookerstudio.google.com/embed/reporting/1",  # Replace with actual
            "comparison": "https://lookerstudio.google.com/embed/reporting/1"  # Replace with actual
        }
    
    def get_embed_url(self, query_type: str, state: str = None) -> str:
        """Simple method to get embed URL based on query type"""
        # Use configured URL or fallback to demo URL
        url = self.base_urls.get(query_type) or self.demo_urls.get(query_type, self.demo_urls["covid"])
        
        # Add state filter to URL if provided
        if state and url:
            if "?" in url:
                url += f"&params=state:{state}"
            else:
                url += f"?params=state:{state}"
        
        return url