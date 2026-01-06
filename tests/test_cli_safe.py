import pytest
from unittest.mock import patch, MagicMock
from aegisx.cli import main

@patch("aegisx.cli.train_model")
def test_cli_train(mock_train):
    with patch("sys.argv", ["aegisx", "train", "--days", "30"]):
        main()
    mock_train.assert_called_once()

@patch("aegisx.cli.SimBroker")
@patch("aegisx.cli.TradingEngine")
@patch("aegisx.cli.OHLCVFetcher")
def test_cli_sim(mock_fetcher, mock_engine, mock_broker):
    mock_fetcher.return_value.fetch_recent.return_value = MagicMock(empty=False)
    mock_broker.return_value.get_balance.return_value = 10000.0
    mock_broker.return_value.trades = []
    
    with patch("sys.argv", ["aegisx", "sim", "--days", "10"]):
        main()
    
    mock_engine.assert_called_once()
    mock_engine.return_value.run_simulation.assert_called_once()
