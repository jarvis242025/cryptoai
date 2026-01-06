import json
import joblib
import glob
import os
import pandas as pd
import logging
from typing import Tuple, Optional
from aegisx.features.build_features import get_feature_names
from aegisx.config import settings

logger = logging.getLogger(__name__)

class Predictor:
    def __init__(self) -> None:
        self.model, self.metadata = self._load_latest_model()
        self.recommended_threshold = self.metadata.get("recommended_threshold") if self.metadata else None
        
    def validate_model(self, force: bool = False) -> None:
        """
        Validate model compatibility with current settings.
        """
        if not self.metadata:
            logger.warning("No metadata found in model. Validation skipped.")
            return

        errors = []
        
        # Check Exchange
        model_exchange = self.metadata.get("exchange_id")
        if model_exchange and model_exchange != settings.EXCHANGE_ID:
            errors.append(f"Exchange mismatch: Model={model_exchange}, Config={settings.EXCHANGE_ID}")
            
        # Check Symbol
        model_symbol = self.metadata.get("symbol")
        if model_symbol and model_symbol != settings.SYMBOL:
            errors.append(f"Symbol mismatch: Model={model_symbol}, Config={settings.SYMBOL}")
            
        # Check Timeframe
        model_tf = self.metadata.get("timeframe")
        if model_tf and model_tf != settings.TIMEFRAME:
            errors.append(f"Timeframe mismatch: Model={model_tf}, Config={settings.TIMEFRAME}")
            
        # Check Features (Optional but good safety)
        model_features = self.metadata.get("features")
        current_features = get_feature_names()
        if model_features and model_features != current_features:
            errors.append(f"Feature mismatch: Model has {len(model_features)}, Code has {len(current_features)}")

        if errors:
            msg = "Model Compatibility Errors:\n" + "\n".join(errors)
            if force:
                logger.warning(f"{msg}\n--force flag provided. Proceeding with caution.")
            else:
                logger.error(msg)
                raise RuntimeError(f"Model incompatibility detected. Use --force to override.\n{msg}")
        else:
            logger.info("Model compatibility verified.")

    def _load_latest_model(self) -> Tuple[object, Optional[dict]]:
        files = glob.glob("aegisx/models/*.joblib")
        if not files:
            raise FileNotFoundError("No trained models found in aegisx/models/")
        
        # Sort by name (timestamp included)
        latest_file = sorted(files)[-1]
        logger.info(f"Loading model: {latest_file}")
        model = joblib.load(latest_file)
        
        # Load metadata
        meta_file = latest_file.replace(".joblib", ".json")
        metadata = {}
        if os.path.exists(meta_file):
            try:
                with open(meta_file, "r") as f:
                    metadata = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load metadata: {e}")
                
        return model, metadata
        
    def predict_probability(self, row: pd.Series) -> float:
        """
        Predict probability for a single row (pd.Series).
        """
        features = get_feature_names()
        
        # Reshape for sklearn
        X = pd.DataFrame([row[features]])
        
        try:
            # predict_proba returns [prob_0, prob_1]
            prob = self.model.predict_proba(X)[0][1]
            return prob
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return 0.0
