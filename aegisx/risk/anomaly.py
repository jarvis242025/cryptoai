import pandas as pd
from sklearn.ensemble import IsolationForest
import logging

logger = logging.getLogger(__name__)

class AnomalyDetector:
    def __init__(self) -> None:
        self.clf = IsolationForest(contamination=0.05, random_state=42)
        self.is_fitted = False
        
    def fit(self, df: pd.DataFrame) -> None:
        # Use returns and range for anomaly detection
        features = df[['returns', 'range_pct', 'vol_spike']].dropna()
        if len(features) < 50:
            logger.warning("Not enough data to fit AnomalyDetector.")
            return
            
        self.clf.fit(features)
        self.is_fitted = True
        
    def is_anomaly(self, row: pd.Series) -> bool:
        if not self.is_fitted:
            return False
            
        try:
            X = pd.DataFrame([row[['returns', 'range_pct', 'vol_spike']]])
            # -1 is anomaly, 1 is normal
            pred = self.clf.predict(X)[0]
            return pred == -1
        except Exception:
            return False
