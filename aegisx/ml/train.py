import os
import json
import joblib
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, brier_score_loss, average_precision_score
from sklearn.model_selection import train_test_split

from aegisx.config import settings
from aegisx.data.ohlcv import OHLCVFetcher
from aegisx.features.build_features import build_features, get_feature_names
from aegisx.ml.labeling import triple_barrier_label

from typing import Optional

logger = logging.getLogger(__name__)

def train_model(
    days: int = 365, 
    start_date: Optional[str] = None, 
    save: bool = True,
    threshold_policy: str = "max_precision",
    target_signals_per_day: float = 2.0
) -> None:
    # 1. Fetch
    fetcher = OHLCVFetcher()
    
    if start_date:
        logger.info(f"Starting training pipeline (From {start_date})...")
        try:
            dt_start = datetime.strptime(start_date, "%Y-%m-%d")
            df = fetcher.fetch_from(dt_start)
        except ValueError:
            logger.error("Invalid start-date format. Use YYYY-MM-DD.")
            return
    else:
        logger.info(f"Starting training pipeline ({days} days)...")
        df = fetcher.fetch_recent(lookback_days=days)
    
    if df.empty:
        logger.error("No data fetched. Aborting training.")
        return
        
    logger.info(f"Coverage: {df.index[0]} to {df.index[-1]} ({len(df)} rows)")
    logger.info(f"Labeling Horizon: {settings.HORIZON_BARS} bars")
    
    # 2. Features
    df = build_features(df)
    feature_cols = get_feature_names()
    
    # 3. Labeling
    logger.info("Generating labels...")
    df['target'] = triple_barrier_label(df)
    
    # Drop unlabeled end rows
    df = df.iloc[:-settings.HORIZON_BARS]
    
    # Label Distribution Stats
    n_pos = df['target'].sum()
    n_total = len(df)
    pos_rate = (n_pos / n_total) * 100 if n_total > 0 else 0
    
    logger.info(f"Label Stats: Positives={n_pos}, Total={n_total}, Rate={pos_rate:.2f}%")
    
    if pos_rate < 1.0:
        logger.warning("WARNING: Positive labels < 1% of dataset. TP/SL/Horizon settings may be too strict!")
    
    if df['target'].nunique() < 2:
        logger.error("Training aborted: Target has only 1 class. Need both wins and losses to train.")
        return

    X = df[feature_cols]
    y = df['target']
    
    # 4. Split (Time-based: Train first 80%, Val last 20%)
    split_idx = int(len(df) * 0.8)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
    
    logger.info(f"Train size: {len(X_train)}, Val size: {len(X_val)}")
    
    # Calculate Val Duration in Days for stats
    if len(X_val) > 1:
        # Robust duration using index timestamps if available
        try:
            # Assuming index is datetime or can be converted
            ts_start = pd.to_datetime(X_val.index[0])
            ts_end = pd.to_datetime(X_val.index[-1])
            val_days = (ts_end - ts_start).total_seconds() / 86400.0
            val_days = max(0.1, val_days) # Avoid zero div
        except Exception:
            # Fallback to bar count approximation
            val_days = len(X_val) * 15 / 1440.0 # 15m candles approximation
    else:
        val_days = 1.0
    logger.info(f"Validation Duration: {val_days:.2f} days")
    
    # 5. Train
    if settings.MODEL_TYPE == "xgb":
        try:
            from xgboost import XGBClassifier
        except ImportError:
            logger.error("XGBoost selected but not installed. Run: pip install xgboost")
            return

        base_model = XGBClassifier(
            n_estimators=settings.XGB_N_ESTIMATORS,
            max_depth=settings.XGB_MAX_DEPTH,
            learning_rate=settings.XGB_LEARNING_RATE,
            subsample=settings.XGB_SUBSAMPLE,
            colsample_bytree=settings.XGB_COLSAMPLE_BYTREE,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1
            # scale_pos_weight handled by calibration or manual balancing if needed,
            # but usually XGB handles imbalance well with appropriate metrics.
            # We'll stick to calibration for probability accuracy.
        )
        logger.info("Training XGBoost Model...")
    else:
        base_model = RandomForestClassifier(
            n_estimators=300, 
            max_depth=5, 
            min_samples_leaf=20,
            random_state=42,
            class_weight="balanced_subsample"
        )
        logger.info("Training Random Forest Model...")
    
    # Calibrate probabilities
    clf = CalibratedClassifierCV(base_model, cv=3, method='isotonic')
    clf.fit(X_train, y_train)
    
    # 6. Evaluate
    # Metrics at default 0.5 threshold
    y_pred_05 = clf.predict(X_val)
    y_prob = clf.predict_proba(X_val)[:, 1]
    
    prec_05 = precision_score(y_val, y_pred_05, zero_division=0)
    rec_05 = recall_score(y_val, y_pred_05, zero_division=0)
    f1_05 = f1_score(y_val, y_pred_05, zero_division=0)
    acc = accuracy_score(y_val, y_pred_05)
    brier = brier_score_loss(y_val, y_prob)
    pr_auc = average_precision_score(y_val, y_prob)
    
    logger.info(f"Eval (0.5 thr): Prec={prec_05:.2f}, Rec={rec_05:.2f}, F1={f1_05:.2f}, Acc={acc:.2f}")
    logger.info(f"Prob Metrics: Brier={brier:.4f}, PR-AUC={pr_auc:.4f}")

    # --- Automatic Threshold Selection ---
    logger.info("Computing optimal threshold...")
    prob_stats = {
        "min": float(y_prob.min()),
        "mean": float(y_prob.mean()),
        "max": float(y_prob.max()),
        "p50": float(np.percentile(y_prob, 50)),
        "p75": float(np.percentile(y_prob, 75)),
        "p90": float(np.percentile(y_prob, 90)),
        "p95": float(np.percentile(y_prob, 95)),
        "p99": float(np.percentile(y_prob, 99)),
    }
    logger.info(f"Prob Stats: {prob_stats}")

    best_thr = settings.MIN_PROB
    candidates = []
    
    # Grid: 5% to 95% + percentiles
    grid = np.linspace(0.05, 0.95, 19)
    # Add percentiles to grid to ensure we test high ranges
    grid = np.concatenate([grid, [prob_stats['p90'], prob_stats['p95'], prob_stats['p99']]])
    grid = np.unique(grid)
    grid = grid[grid <= 1.0]
    
    for thr in grid:
        y_hat = (y_prob >= thr).astype(int)
        pred_pos = int(y_hat.sum())
        
        # Calculate metrics even if pred_pos = 0 (precision=0)
        prec = precision_score(y_val, y_hat, zero_division=0)
        rec = recall_score(y_val, y_hat, zero_division=0)
        f1 = f1_score(y_val, y_hat, zero_division=0)
        signals_per_day = pred_pos / val_days if val_days > 0 else 0
        
        candidates.append({
            "threshold": float(thr),
            "n_signals": pred_pos,
            "signals_per_day": signals_per_day,
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1)
        })
        
    # Selection Policy
    # 1) Meets MIN_SIGNALS
    min_signals = settings.MIN_SIGNALS
    valid_candidates = [c for c in candidates if c['n_signals'] >= min_signals]
    
    selected_metrics = {}
    policy_msg = ""
    
    if threshold_policy == "target_rate":
        # Target Rate Policy
        # 1. Must meet absolute min signals to be statistically valid
        min_total = settings.MIN_SIGNALS_TOTAL
        pool = [c for c in candidates if c['n_signals'] >= min_total]
        
        # 2. Prefer candidates meeting MIN_VALIDATION_PRECISION
        high_quality_pool = [c for c in pool if c['precision'] >= settings.MIN_VALIDATION_PRECISION]
        
        # Select best pool
        selection_pool = high_quality_pool if high_quality_pool else pool
        
        if selection_pool:
            best_cand = min(selection_pool, key=lambda x: abs(x['signals_per_day'] - target_signals_per_day))
            best_thr = best_cand['threshold']
            selected_metrics = best_cand
            
            quality_tag = "High Quality" if high_quality_pool else "Standard (Precision < Floor)"
            policy_msg = f"Target Rate (~{target_signals_per_day}/day), {quality_tag}, Actual={best_cand['signals_per_day']:.2f}"
        else:
            # Fallback to max precision if we can't meet min_total
             policy_msg = "Fallback (Target Rate failed min signals)"
             logger.warning(f"Target Rate policy failed: no threshold had >= {min_total} signals.")
             # Fallthrough to max_precision logic below
             pass

    # If policy was max_precision OR target_rate failed and we didn't select yet
    if not selected_metrics:
        if valid_candidates:
            # 2) Highest precision among valid
            # In sniper mode, we prioritize precision above all else once minimal signal count is met
            best_cand = max(valid_candidates, key=lambda x: x['precision'])
            
            best_thr = best_cand['threshold']
            selected_metrics = best_cand
            policy_msg = f"Sniper (>= {min_signals} signals), Max Precision"
        else:
            # 3) Fallback: Max Precision (among any signals found)
            # If no threshold yielded >= MIN_SIGNALS, we take the one with highest precision
            # provided it had at least 1 signal.
            candidates_with_signals = [c for c in candidates if c['n_signals'] > 0]
            
            if candidates_with_signals:
                best_cand = max(candidates_with_signals, key=lambda x: x['precision'])
                best_thr = best_cand['threshold']
                selected_metrics = best_cand
                policy_msg = "Fallback (Max Precision, n < MIN_SIGNALS)"
                logger.warning(f"Sniper mode: no validation signals met threshold (>= {min_signals}). Picked max precision.")
            else:
                # Absolute fallback if literally 0 signals everywhere (e.g. extremely poor model)
                # We pick the highest threshold tested (conservative) or just default?
                # Let's pick p99 to be safe/conservative
                best_thr = prob_stats['p99']
                policy_msg = "Fallback (P99 - No Signals)"
                logger.warning("Model failed to produce signals at any tested threshold.")

    if selected_metrics:
        logger.info(f"Selected Threshold: {best_thr:.4f} ({policy_msg})")
        logger.info(f"Metrics at Thr: {selected_metrics}")
    else:
        logger.warning(f"Selected Fallback Threshold: {best_thr:.4f}")

    # 7. Save
    if save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = "aegisx/models"
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = f"{model_dir}/model_{timestamp}.joblib"
        meta_path = f"{model_dir}/model_{timestamp}.json"
        
        joblib.dump(clf, model_path)
        
        metadata = {
            "timestamp": timestamp,
            "exchange_id": settings.EXCHANGE_ID,
            "symbol": settings.SYMBOL,
            "timeframe": settings.TIMEFRAME,
            "features": feature_cols,
            "recommended_threshold": best_thr,
            "threshold_policy": threshold_policy,
            "prob_stats": prob_stats,
            "metrics": {
                "threshold_0.5": {
                    "precision": prec_05,
                    "recall": rec_05,
                    "f1": f1_05,
                    "accuracy": acc,
                    "brier": brier,
                    "pr_auc": pr_auc
                },
                "selected": selected_metrics
            },
            "train_rows": len(X_train)
        }
        
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
            
        logger.info(f"Saved model to {model_path}")
