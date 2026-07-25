from __future__ import annotations

import unittest

import numpy as np

from drag_conveyor.geometry_v2.tracking import KalmanConfig, TrackingError, initialize_from_two_measurements, predict, update


class KalmanTrackingTests(unittest.TestCase):
    def test_seed_predict_and_joseph_update_are_finite(self) -> None:
        config = KalmanConfig(chain_span_px=100.0)
        state = initialize_from_two_measurements(10.0, 1.0, 0.0, 20.0, 1.0, 0.1, config)
        prediction = predict(state, 0.2, config)
        result = update(prediction, 31.0, 1.0, config)

        self.assertAlmostEqual(prediction.state.mean[0], 30.0)
        self.assertAlmostEqual(result.innovation, 1.0)
        self.assertGreater(result.innovation_variance, 0.0)
        self.assertTrue(np.all(np.linalg.eigvalsh(result.state.covariance) >= -1e-9))

    def test_zero_dt_prediction_does_not_predict_twice(self) -> None:
        config = KalmanConfig(chain_span_px=100.0)
        state = initialize_from_two_measurements(10.0, 1.0, 0.0, 20.0, 1.0, 0.1, config)

        prediction = predict(state, 0.1, config)

        np.testing.assert_allclose(prediction.state.mean, state.mean)
        np.testing.assert_allclose(prediction.state.covariance, state.covariance)

    def test_time_cannot_run_backwards(self) -> None:
        config = KalmanConfig(chain_span_px=100.0)
        state = initialize_from_two_measurements(10.0, 1.0, 0.0, 20.0, 1.0, 0.1, config)

        with self.assertRaises(TrackingError):
            predict(state, 0.09, config)


if __name__ == "__main__":
    unittest.main()
