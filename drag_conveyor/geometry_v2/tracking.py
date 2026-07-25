from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


class TrackingError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class KalmanConfig:
    chain_span_px: float
    sigma_acceleration_ratio_per_sec2: float = 0.15
    minimum_measurement_sigma_px: float = 1.0
    minimum_velocity_sigma_ratio_per_sec: float = 0.02


@dataclass(frozen=True, slots=True)
class KalmanState:
    mean: np.ndarray
    covariance: np.ndarray
    timestamp_sec: float

    def __post_init__(self) -> None:
        if self.mean.shape != (2,) or self.covariance.shape != (2, 2):
            raise TrackingError("Kalman state must have [s,v] and a 2x2 covariance")
        if not np.isfinite(self.mean).all() or not np.isfinite(self.covariance).all():
            raise TrackingError("Kalman state must be finite")


@dataclass(frozen=True, slots=True)
class KalmanPrediction:
    state: KalmanState
    innovation: float | None = None
    innovation_variance: float | None = None


def initialize_from_two_measurements(
    first_s: float,
    first_sigma: float,
    first_time_sec: float,
    second_s: float,
    second_sigma: float,
    second_time_sec: float,
    config: KalmanConfig,
) -> KalmanState:
    dt = second_time_sec - first_time_sec
    if not dt > 0.0:
        raise TrackingError("Second seed measurement must have a later timestamp")
    _validate_measurement(first_s, first_sigma)
    _validate_measurement(second_s, second_sigma)
    var_first = max(first_sigma**2, config.minimum_measurement_sigma_px**2)
    var_second = max(second_sigma**2, config.minimum_measurement_sigma_px**2)
    velocity_variance = max(
        (var_first + var_second) / dt**2,
        (config.minimum_velocity_sigma_ratio_per_sec * config.chain_span_px) ** 2,
    )
    covariance = np.array([[var_second, var_second / dt], [var_second / dt, velocity_variance]], dtype=np.float64)
    return KalmanState(np.array([second_s, (second_s - first_s) / dt], dtype=np.float64), covariance, second_time_sec)


def predict(state: KalmanState, timestamp_sec: float, config: KalmanConfig) -> KalmanPrediction:
    dt = timestamp_sec - state.timestamp_sec
    if dt < 0.0:
        raise TrackingError("Prediction timestamps must be nondecreasing")
    transition = np.array([[1.0, dt], [0.0, 1.0]], dtype=np.float64)
    sigma_acceleration = config.sigma_acceleration_ratio_per_sec2 * config.chain_span_px
    process_noise = sigma_acceleration**2 * np.array(
        [[dt**4 / 4.0, dt**3 / 2.0], [dt**3 / 2.0, dt**2]], dtype=np.float64
    )
    covariance = transition @ state.covariance @ transition.T + process_noise
    covariance = 0.5 * (covariance + covariance.T)
    return KalmanPrediction(KalmanState(transition @ state.mean, covariance, timestamp_sec))


def update(prediction: KalmanPrediction, measurement_s: float, measurement_sigma: float, config: KalmanConfig) -> KalmanPrediction:
    _validate_measurement(measurement_s, measurement_sigma)
    predicted = prediction.state
    measurement_variance = max(measurement_sigma**2, config.minimum_measurement_sigma_px**2)
    innovation = measurement_s - float(predicted.mean[0])
    innovation_variance = float(predicted.covariance[0, 0] + measurement_variance)
    if not math.isfinite(innovation_variance) or innovation_variance <= 0.0:
        raise TrackingError("Innovation variance must be finite and positive")
    gain = predicted.covariance[:, 0] / innovation_variance
    mean = predicted.mean + gain * innovation
    identity_minus_kh = np.eye(2, dtype=np.float64) - np.outer(gain, np.array([1.0, 0.0]))
    covariance = identity_minus_kh @ predicted.covariance @ identity_minus_kh.T + np.outer(gain, gain) * measurement_variance
    covariance = 0.5 * (covariance + covariance.T)
    if np.linalg.eigvalsh(covariance).min() < -1e-9:
        raise TrackingError("Updated covariance is not positive semidefinite")
    return KalmanPrediction(KalmanState(mean, covariance, predicted.timestamp_sec), innovation, innovation_variance)


def _validate_measurement(value: float, sigma: float) -> None:
    if not math.isfinite(value) or not math.isfinite(sigma) or sigma < 0.0:
        raise TrackingError("Measurement and sigma must be finite; sigma must be nonnegative")
