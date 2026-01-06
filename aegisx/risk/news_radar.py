import feedparser
import time
from datetime import datetime, timedelta
import logging
from aegisx.config import settings
from aegisx.risk.news_sources import RSS_FEEDS, RISK_KEYWORDS

logger = logging.getLogger(__name__)

class NewsRadar:
    def __init__(self) -> None:
        self.last_fetch = 0.0
        self.cached_score = 0
        self.veto_active = False
        
    def check_risk(self) -> int:
        """
        Returns a risk score (0-100+).
        If score > THRESHOLD, should VETO trades.
        """
        if not settings.NEWS_ENABLED:
            return 0
            
        now = time.time()
        # Cache for 5 minutes to avoid spamming feeds
        if now - self.last_fetch < 300:
            return self.cached_score
            
        total_risk = 0
        
        try:
            for url in RSS_FEEDS:
                feed = feedparser.parse(url)
                for entry in feed.entries[:5]: # Check top 5 items per feed
                    # Parse time
                    # Logic simplified: if published within last COOLDOWN minutes
                    pub_struct = entry.get('published_parsed') or entry.get('updated_parsed')
                    if not pub_struct:
                        continue
                        
                    pub_ts = time.mktime(pub_struct)
                    age_mins = (now - pub_ts) / 60
                    
                    if age_mins < settings.NEWS_COOLDOWN_MIN:
                        title = entry.title.lower()
                        summary = entry.get('summary', '').lower()
                        text = f"{title} {summary}"
                        
                        for kw, score in RISK_KEYWORDS.items():
                            if kw in text:
                                logger.warning(f"NEWS RISK: '{kw}' found in '{entry.title}' ({age_mins:.1f}m ago)")
                                total_risk += score
                                
        except Exception as e:
            logger.error(f"News fetch error: {e}")
            
        self.last_fetch = now
        self.cached_score = total_risk
        
        if total_risk > settings.NEWS_RISK_THRESHOLD:
            self.veto_active = True
        else:
            self.veto_active = False
            
        return total_risk
