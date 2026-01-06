import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from aegisx.config import settings
from aegisx.ml.train import train_model

@patch("aegisx.ml.train.average_precision_score", return_value=0.8)
@patch("aegisx.ml.train.brier_score_loss", return_value=0.1)
@patch("aegisx.ml.train.accuracy_score", return_value=0.8)
@patch("aegisx.ml.train.f1_score", return_value=0.6)
@patch("aegisx.ml.train.recall_score", return_value=0.5)
@patch("aegisx.ml.train.precision_score", return_value=0.75)
@patch("aegisx.ml.train.OHLCVFetcher")
@patch("aegisx.ml.train.build_features")
@patch("aegisx.ml.train.triple_barrier_label")
@patch("aegisx.ml.train.CalibratedClassifierCV")
@patch("aegisx.ml.train.joblib.dump")
@patch("aegisx.ml.train.json.dump")
@patch("aegisx.ml.train.open")
def test_calibration_logic(mock_open, mock_json, mock_joblib, mock_clf_cls, mock_label, mock_features, mock_fetcher,
                           mock_prec, mock_rec, mock_f1, mock_acc, mock_brier, mock_prauc):
    # Setup Mocks
    mock_fetcher_instance = mock_fetcher.return_value
    
    # Mock DF with dict and index
    class MockIndex:
        def __init__(self, size):
            self.size = size
        def __getitem__(self, idx):
            if idx == 0: return "2025-01-01"
            if idx == -1: return "2025-01-03" # 2 days
            return "2025-01-02"
        def __len__(self): return self.size

    class MockDF(dict):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.index = MockIndex(1000)
            self.empty = False
            
            # Mock iloc as an object with __getitem__
            class IlocIndexer:
                def __getitem__(s, idx):
                    return self # Return self (sliced df)
            self.iloc = IlocIndexer()
            
        def copy(self): return self
        def dropna(self, inplace=True): pass
        def __getitem__(self, key):
            # If key is list (cols), return self
            if isinstance(key, list): return self
            return super().get(key, MagicMock()) # Return mock for columns
        def __len__(self): return 1000

    df = MockDF()
    # Mock target column access specifically for sum()
    target_col = MagicMock()
    target_col.sum.return_value = 100
    target_col.nunique.return_value = 2
    df['target'] = target_col
    
    mock_fetcher_instance.fetch_recent.return_value = df
    mock_fetcher_instance.fetch_from.return_value = df
    mock_features.return_value = df
    mock_label.return_value = target_col # Re-use mock
    
    # Mock Model predictions
    mock_clf = mock_clf_cls.return_value
    
    # Validation size ~200 (20% of 1000)
    n_val = 1000 # Since iloc returns self (1000)
    
    probs = np.linspace(0.1, 0.9, n_val)
    proba_output = np.column_stack((1-probs, probs))
    
    mock_clf.predict_proba.return_value = proba_output
    mock_clf.predict.return_value = (probs > 0.5).astype(int)
    
    # RUN TEST: Target Rate
    train_model(days=30, threshold_policy="target_rate", target_signals_per_day=5.0)
    
    # Verify metadata
    args, _ = mock_json.call_args
    metadata = args[0]
    
    assert metadata["threshold_policy"] == "target_rate"
    assert "recommended_threshold" in metadata
    
    pass

@patch("aegisx.ml.train.OHLCVFetcher")
@patch("aegisx.ml.train.build_features")
@patch("aegisx.ml.train.triple_barrier_label")
@patch("aegisx.ml.train.CalibratedClassifierCV")
@patch("aegisx.ml.train.joblib.dump")
@patch("aegisx.ml.train.json.dump")
@patch("aegisx.ml.train.open")
def test_low_probability_distribution_selection(
    mock_open, mock_json, mock_joblib, mock_clf_cls, mock_label, mock_features, mock_fetcher
):
    from aegisx.config import settings
    from aegisx.ml.train import train_model
    
    np.random.seed(42)

    # Raw OHLCV DF
    idx = pd.date_range("2025-01-01", periods=1000, freq="15min")
    raw = pd.DataFrame(
        {
            "open": np.random.rand(len(idx)),
            "high": np.random.rand(len(idx)),
            "low": np.random.rand(len(idx)),
            "close": np.random.rand(len(idx)),
            "volume": np.random.rand(len(idx)),
        },
        index=idx,
    )
    mock_fetcher.return_value.fetch_recent.return_value = raw
    mock_fetcher.return_value.fetch_from.return_value = raw

    # Features DF (must support df['target'] assignment)
    feats = pd.DataFrame(
        {
            "f1": np.random.randn(len(idx)),
            "f2": np.random.randn(len(idx)),
        },
        index=idx,
    )
    mock_features.return_value = feats

    # Labels: provide BOTH classes to avoid early abort and precision/recall warnings
    labels = np.zeros(len(idx), dtype=int)
    labels[0:100] = 1 # positives in training slice
    labels[-200:] = 1 # positives in validation slice
    mock_label.return_value = pd.Series(labels, index=idx)

    # Calculate validation size expected by train_model (80/20 split)
    n_total = len(idx) - settings.HORIZON_BARS
    split_idx = int(n_total * 0.8)
    n_val = n_total - split_idx

    # Compressed probabilities: max 0.40
    probs = np.linspace(0.1, 0.4, n_val)
    mock_clf = mock_clf_cls.return_value
    mock_clf.predict_proba.return_value = np.column_stack((1 - probs, probs))
    mock_clf.predict.return_value = np.zeros(n_val)

    # Override settings for test
    settings.MIN_VALIDATION_PRECISION = 0.40
    settings.TARGET_SIGNALS_PER_DAY = 1.0
    settings.MIN_SIGNALS_TOTAL = 1

    # Ensure precision passes quality gate for thresholds we evaluate
    with patch("aegisx.ml.train.precision_score", return_value=0.45), \
         patch("aegisx.ml.train.get_feature_names", return_value=["f1", "f2"]):
        train_model(days=30, threshold_policy="target_rate", target_signals_per_day=1.0)

    # Assert saved metadata contains a "low" threshold (<0.5)
    assert mock_json.called, "json.dump was not called (training likely aborted early)"
    meta = mock_json.call_args[0][0]
    
    # Resilient key detection
    thr_key = "recommended_threshold" if "recommended_threshold" in meta else "entry_threshold"
    assert thr_key in meta, f"Metadata dict keys: {meta.keys()}"
    assert float(meta[thr_key]) < 0.5

@pytest.fixture(autouse=True)
def fix_metrics_mock():
    # Force metrics to return floats to avoid Mock comparison issues
    import sklearn.metrics
    if isinstance(sklearn.metrics.precision_score, MagicMock):
        sklearn.metrics.precision_score.return_value = 0.75 
        sklearn.metrics.recall_score.return_value = 0.5
        sklearn.metrics.f1_score.return_value = 0.6
        sklearn.metrics.accuracy_score.return_value = 0.8
        sklearn.metrics.brier_score_loss.return_value = 0.1
        sklearn.metrics.average_precision_score.return_value = 0.8
