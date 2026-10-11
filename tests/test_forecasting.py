import numpy as np
import pandas as pd
import pytest

from src.forecasting import make_targets, make_windows, reconstruct_close


def test_targets_are_next_session_log_returns_and_direction():
    df = pd.DataFrame({"Close": [100.0, 110.0, 99.0, 99.0, 108.9]})
    returns, direction, target_indices = make_targets(df, lookback=2)
    assert target_indices.tolist() == [2, 3, 4]
    np.testing.assert_allclose(returns, np.log([99 / 110, 99 / 99, 108.9 / 99]), rtol=1e-6)
    assert direction.tolist() == [0.0, 0.0, 1.0]


def test_windows_do_not_include_target_row():
    features = np.arange(8, dtype=np.float32).reshape(-1, 1)
    windows = make_windows(features, np.asarray([3, 4]), lookback=3)
    np.testing.assert_array_equal(windows[0, :, 0], [0, 1, 2])
    np.testing.assert_array_equal(windows[1, :, 0], [1, 2, 3])


def test_reconstruct_close_from_log_return():
    result = reconstruct_close(np.asarray([100.0]), np.asarray([np.log(1.1)]))
    np.testing.assert_allclose(result, [110.0], rtol=1e-10)


def test_reconstruct_close_rejects_invalid_price():
    with pytest.raises(ValueError):
        reconstruct_close(np.asarray([0.0]), np.asarray([0.0]))
