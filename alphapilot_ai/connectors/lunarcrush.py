"""
LunarCrush API Connector

Fetches social sentiment metrics for crypto assets:
- Galaxy Score: Overall social health (0-100)
- AltRank: Relative performance vs other assets
- Social Volume: Number of social posts/mentions
- Social Engagement: Likes, shares, comments
- Sentiment: Bullish/bearish ratio
- Influencer Activity: Notable account mentions

This data is fed into Claude's decision prompt so it can factor
in social momentum when making trade decisions.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

import httpx

from utils.logger import get_logger

logger = get_logger(__name__)

# Cache to avoid hammering the API
_cache: dict[str, tuple[float, Any]] = {}
_CACHE_TTL = 300  # 5 minutes


@dataclass
class SocialMetrics:
    """Social sentiment data for a single asset."""
    symbol: str
    name: str
    
    # Core scores (0-100 scale)
    galaxy_score: float  # Overall social health
    alt_rank: int  # Rank among all tracked assets (1 = best)
    
    # Volume metrics
    social_volume: int  # Total social posts in period
    social_volume_change_24h: float  # % change
    
    # Engagement
    social_engagements: int  # Total interactions
    social_contributors: int  # Unique accounts posting
    
    # Sentiment
    sentiment_score: float  # -1 to 1 (bearish to bullish)
    bullish_pct: float  # % of posts that are bullish
    bearish_pct: float  # % of posts that are bearish
    
    # News/Influencers
    news_articles: int  # Recent news count
    influencer_mentions: int  # Notable account mentions
    
    # Price correlation
    correlation_rank: int  # How well social predicts price
    
    # Trend indicators
    social_volume_trend: str  # "rising", "falling", "stable"
    sentiment_trend: str  # "improving", "declining", "stable"
    
    @property
    def is_buzzing(self) -> bool:
        """High social activity - potential breakout signal."""
        return self.social_volume_change_24h > 50 and self.galaxy_score > 60
    
    @property
    def is_fading(self) -> bool:
        """Declining interest - caution signal."""
        return self.social_volume_change_24h < -30 and self.sentiment_score < 0
    
    @property
    def has_influencer_pump(self) -> bool:
        """Influencer activity spike - could be pump setup."""
        return self.influencer_mentions > 5 and self.social_volume_change_24h > 100
    
    def to_prompt_context(self) -> str:
        """Format for inclusion in Claude's decision prompt."""
        sentiment_label = "BULLISH" if self.sentiment_score > 0.2 else (
            "BEARISH" if self.sentiment_score < -0.2 else "NEUTRAL"
        )
        
        alerts = []
        if self.is_buzzing:
            alerts.append("HIGH BUZZ - social volume spiking")
        if self.is_fading:
            alerts.append("FADING INTEREST - declining engagement")
        if self.has_influencer_pump:
            alerts.append("INFLUENCER ACTIVITY - potential pump")
        
        return f"""
Social Sentiment ({self.symbol}):
  Galaxy Score: {self.galaxy_score:.0f}/100 (rank #{self.alt_rank})
  Sentiment: {sentiment_label} ({self.bullish_pct:.0f}% bullish, {self.bearish_pct:.0f}% bearish)
  Social Volume: {self.social_volume:,} posts ({self.social_volume_change_24h:+.1f}% 24h)
  Engagement: {self.social_engagements:,} interactions from {self.social_contributors:,} contributors
  News Articles: {self.news_articles} | Influencer Mentions: {self.influencer_mentions}
  Trends: Volume {self.social_volume_trend}, Sentiment {self.sentiment_trend}
  {' | '.join(alerts) if alerts else 'No alerts'}
""".strip()


class LunarCrushClient:
    """Client for LunarCrush API v2."""
    
    BASE_URL = "https://lunarcrush.com/api4/public"
    
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("LUNARCRUSH_API_KEY", "")
        if not self.api_key:
            logger.warning("LUNARCRUSH_API_KEY not set - social sentiment disabled")
    
    def _get(self, endpoint: str, params: dict | None = None) -> dict[str, Any]:
        """Make authenticated GET request with caching."""
        if not self.api_key:
            return {"error": "API key not configured"}
        
        cache_key = f"{endpoint}:{params}"
        now = time.time()
        
        # Check cache
        if cache_key in _cache:
            cached_at, data = _cache[cache_key]
            if now - cached_at < _CACHE_TTL:
                return data
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json",
            }
            
            with httpx.Client(timeout=10) as client:
                resp = client.get(
                    f"{self.BASE_URL}/{endpoint}",
                    params=params or {},
                    headers=headers,
                )
                resp.raise_for_status()
                data = resp.json()
                
                # Cache successful response
                _cache[cache_key] = (now, data)
                return data
                
        except httpx.HTTPStatusError as e:
            logger.error(f"LunarCrush API error: {e.response.status_code}")
            return {"error": f"HTTP {e.response.status_code}"}
        except Exception as e:
            logger.error(f"LunarCrush request failed: {e}")
            return {"error": str(e)}
    
    def get_coin_metrics(self, symbol: str) -> SocialMetrics | None:
        """
        Fetch social metrics for a single coin.
        
        Args:
            symbol: Coin symbol like "BTC", "ETH", "SOL"
                    (strips "-USD" suffix if present)
        """
        # Normalize symbol (BTC-USD -> BTC)
        clean_symbol = symbol.replace("-USD", "").replace("-USDT", "").upper()
        
        data = self._get("coins", {"symbol": clean_symbol})
        
        if "error" in data:
            logger.debug(f"LunarCrush error for {clean_symbol}: {data['error']}")
            return None
        
        # LunarCrush returns data in a "data" array
        coins = data.get("data", [])
        if not coins:
            return None
        
        coin = coins[0] if isinstance(coins, list) else coins
        
        # Calculate trends
        vol_change = float(coin.get("social_volume_24h_change", 0) or 0)
        if vol_change > 20:
            vol_trend = "rising"
        elif vol_change < -20:
            vol_trend = "falling"
        else:
            vol_trend = "stable"
        
        sent_score = float(coin.get("sentiment", 0) or 0)
        sent_prev = float(coin.get("sentiment_24h_previous", sent_score) or sent_score)
        if sent_score > sent_prev + 0.1:
            sent_trend = "improving"
        elif sent_score < sent_prev - 0.1:
            sent_trend = "declining"
        else:
            sent_trend = "stable"
        
        return SocialMetrics(
            symbol=clean_symbol,
            name=coin.get("name", clean_symbol),
            galaxy_score=float(coin.get("galaxy_score", 0) or 0),
            alt_rank=int(coin.get("alt_rank", 999) or 999),
            social_volume=int(coin.get("social_volume", 0) or 0),
            social_volume_change_24h=vol_change,
            social_engagements=int(coin.get("social_engagements", 0) or 0),
            social_contributors=int(coin.get("social_contributors", 0) or 0),
            sentiment_score=sent_score,
            bullish_pct=float(coin.get("bullish_sentiment", 50) or 50),
            bearish_pct=float(coin.get("bearish_sentiment", 50) or 50),
            news_articles=int(coin.get("news", 0) or 0),
            influencer_mentions=int(coin.get("influential_mentions", 0) or 0),
            correlation_rank=int(coin.get("correlation_rank", 0) or 0),
            social_volume_trend=vol_trend,
            sentiment_trend=sent_trend,
        )
    
    def get_trending(self, limit: int = 10) -> list[SocialMetrics]:
        """Get top trending coins by social activity."""
        data = self._get("coins/list", {"sort": "galaxy_score", "limit": limit})
        
        if "error" in data:
            return []
        
        results = []
        for coin in data.get("data", [])[:limit]:
            metrics = self._parse_coin(coin)
            if metrics:
                results.append(metrics)
        
        return results
    
    def get_top_gainers_social(self, limit: int = 10) -> list[SocialMetrics]:
        """Get coins with biggest social volume increase."""
        data = self._get("coins/list", {"sort": "social_volume_24h_change", "limit": limit})
        
        if "error" in data:
            return []
        
        results = []
        for coin in data.get("data", [])[:limit]:
            metrics = self._parse_coin(coin)
            if metrics:
                results.append(metrics)
        
        return results
    
    def _parse_coin(self, coin: dict) -> SocialMetrics | None:
        """Parse coin data into SocialMetrics."""
        try:
            vol_change = float(coin.get("social_volume_24h_change", 0) or 0)
            vol_trend = "rising" if vol_change > 20 else ("falling" if vol_change < -20 else "stable")
            
            sent_score = float(coin.get("sentiment", 0) or 0)
            sent_prev = float(coin.get("sentiment_24h_previous", sent_score) or sent_score)
            sent_trend = "improving" if sent_score > sent_prev + 0.1 else (
                "declining" if sent_score < sent_prev - 0.1 else "stable"
            )
            
            return SocialMetrics(
                symbol=coin.get("symbol", ""),
                name=coin.get("name", ""),
                galaxy_score=float(coin.get("galaxy_score", 0) or 0),
                alt_rank=int(coin.get("alt_rank", 999) or 999),
                social_volume=int(coin.get("social_volume", 0) or 0),
                social_volume_change_24h=vol_change,
                social_engagements=int(coin.get("social_engagements", 0) or 0),
                social_contributors=int(coin.get("social_contributors", 0) or 0),
                sentiment_score=sent_score,
                bullish_pct=float(coin.get("bullish_sentiment", 50) or 50),
                bearish_pct=float(coin.get("bearish_sentiment", 50) or 50),
                news_articles=int(coin.get("news", 0) or 0),
                influencer_mentions=int(coin.get("influential_mentions", 0) or 0),
                correlation_rank=int(coin.get("correlation_rank", 0) or 0),
                social_volume_trend=vol_trend,
                sentiment_trend=sent_trend,
            )
        except Exception as e:
            logger.debug(f"Failed to parse coin data: {e}")
            return None


# Singleton instance
_client: LunarCrushClient | None = None


def get_client() -> LunarCrushClient:
    """Get or create the LunarCrush client singleton."""
    global _client
    if _client is None:
        _client = LunarCrushClient()
    return _client


def get_social_metrics(symbol: str) -> SocialMetrics | None:
    """
    Convenience function to get social metrics for a symbol.
    
    Usage:
        metrics = get_social_metrics("BTC-USD")
        if metrics:
            print(metrics.to_prompt_context())
    """
    return get_client().get_coin_metrics(symbol)


def get_social_context_for_prompt(symbols: list[str]) -> str:
    """
    Get formatted social context for multiple symbols,
    ready to inject into Claude's decision prompt.
    """
    client = get_client()
    contexts = []
    
    for symbol in symbols[:5]:  # Limit to avoid prompt bloat
        metrics = client.get_coin_metrics(symbol)
        if metrics:
            contexts.append(metrics.to_prompt_context())
    
    if not contexts:
        return "Social sentiment data: unavailable"
    
    return "\n\n".join(contexts)
