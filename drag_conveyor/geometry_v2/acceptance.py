from __future__ import annotations

from dataclasses import dataclass

from .decision import FinalStatus


@dataclass(frozen=True, slots=True)
class GroundTruthEvent:
    event_id: str
    crossing_timestamp_sec: float
    status: FinalStatus
    partial_boundary: bool = False


@dataclass(frozen=True, slots=True)
class PredictedEvent:
    paddle_id: int
    crossing_timestamp_sec: float
    status: FinalStatus
    suspected_breakage: bool = False


@dataclass(frozen=True, slots=True)
class EventMatch:
    ground_truth_id: str
    predicted_paddle_id: int
    crossing_error_sec: float


@dataclass(frozen=True, slots=True)
class AcceptanceMetrics:
    matched_events: int
    missed_events: int
    extra_events: int
    precision: float
    recall: float
    status_pairs: tuple[tuple[str, str], ...]
    matches: tuple[EventMatch, ...] = ()
    duplicate_events: int = 0


@dataclass(frozen=True, slots=True)
class ClassificationMetrics:
    matched_events: int
    definitive_events: int
    uncertain_events: int
    coverage: float
    uncertain_rate: float
    selective_exact_accuracy: float
    broken_safety_alert_recall: float
    dangerous_false_normal_rate: float
    normal_recall: float
    false_break_alert_rate_on_normal: float
    confusion_pairs: tuple[tuple[str, str], ...]


def evaluate_events(
    predictions: tuple[PredictedEvent, ...],
    ground_truth: tuple[GroundTruthEvent, ...],
    *,
    maximum_crossing_delta_sec: float,
) -> AcceptanceMetrics:
    """One-to-one, deterministic event matching by crossing time.

    A prediction is never matched twice; ties are resolved by timestamp then IDs,
    making a replay comparison stable across runs.
    """
    if maximum_crossing_delta_sec <= 0:
        raise ValueError("maximum_crossing_delta_sec must be positive")
    truth = tuple(sorted((item for item in ground_truth if not item.partial_boundary), key=lambda item: (item.crossing_timestamp_sec, item.event_id)))
    predicted = tuple(sorted(predictions, key=lambda item: (item.crossing_timestamp_sec, item.paddle_id)))
    match_indexes = _order_preserving_matches(predicted, truth, maximum_crossing_delta_sec)
    matched_prediction_indexes = {prediction_index for _, prediction_index in match_indexes}
    matches = tuple(
        EventMatch(truth[truth_index].event_id, predicted[prediction_index].paddle_id, abs(truth[truth_index].crossing_timestamp_sec - predicted[prediction_index].crossing_timestamp_sec))
        for truth_index, prediction_index in match_indexes
    )
    pairs = tuple((truth[truth_index].status.value, predicted[prediction_index].status.value) for truth_index, prediction_index in match_indexes)
    duplicate_events = sum(
        prediction_index not in matched_prediction_indexes
        and any(abs(item.crossing_timestamp_sec - predicted[prediction_index].crossing_timestamp_sec) <= _event_match_gate(truth, truth_index, maximum_crossing_delta_sec) for truth_index, item in enumerate(truth))
        for prediction_index in range(len(predicted))
    )
    matched = len(matches)
    total_predicted = len(predicted)
    total_truth = len(truth)
    return AcceptanceMetrics(
        matched_events=matched,
        missed_events=total_truth - matched,
        extra_events=total_predicted - matched,
        precision=matched / total_predicted if total_predicted else 0.0,
        recall=matched / total_truth if total_truth else 0.0,
        status_pairs=pairs,
        matches=matches,
        duplicate_events=duplicate_events,
    )


def _order_preserving_matches(
    predictions: tuple[PredictedEvent, ...],
    truth: tuple[GroundTruthEvent, ...],
    maximum_crossing_delta_sec: float,
) -> tuple[tuple[int, int], ...]:
    """Exact DP: maximize matches, then minimize timing error, then stable IDs."""
    table: list[list[tuple[int, float, tuple[tuple[int, int], ...]]]] = [
        [(0, 0.0, ()) for _ in range(len(predictions) + 1)] for _ in range(len(truth) + 1)
    ]
    for truth_index in range(1, len(truth) + 1):
        for prediction_index in range(1, len(predictions) + 1):
            candidates = [table[truth_index - 1][prediction_index], table[truth_index][prediction_index - 1]]
            current_truth = truth[truth_index - 1]
            current_prediction = predictions[prediction_index - 1]
            error = abs(current_truth.crossing_timestamp_sec - current_prediction.crossing_timestamp_sec)
            if error <= _event_match_gate(truth, truth_index - 1, maximum_crossing_delta_sec):
                previous = table[truth_index - 1][prediction_index - 1]
                candidates.append((previous[0] + 1, previous[1] + error, (*previous[2], (truth_index - 1, prediction_index - 1))))
            table[truth_index][prediction_index] = min(
                candidates,
                key=lambda item: (-item[0], item[1], tuple((truth[t].event_id, predictions[p].paddle_id) for t, p in item[2])),
            )
    return table[-1][-1][2]


def _event_match_gate(truth: tuple[GroundTruthEvent, ...], index: int, maximum_crossing_delta_sec: float) -> float:
    intervals = [
        abs(truth[index].crossing_timestamp_sec - truth[neighbor].crossing_timestamp_sec)
        for neighbor in (index - 1, index + 1)
        if 0 <= neighbor < len(truth) and truth[index].crossing_timestamp_sec != truth[neighbor].crossing_timestamp_sec
    ]
    return min(maximum_crossing_delta_sec, 0.35 * min(intervals)) if intervals else maximum_crossing_delta_sec


def evaluate_classification(
    predictions: tuple[PredictedEvent, ...],
    ground_truth: tuple[GroundTruthEvent, ...],
    event_metrics: AcceptanceMetrics,
) -> ClassificationMetrics:
    """Evaluate matched events with explicit abstention and break-alert semantics."""
    predicted_by_id = {item.paddle_id: item for item in predictions}
    truth_by_id = {item.event_id: item for item in ground_truth if not item.partial_boundary}
    pairs = [
        (truth_by_id[match.ground_truth_id], predicted_by_id[match.predicted_paddle_id])
        for match in event_metrics.matches
        if match.ground_truth_id in truth_by_id and match.predicted_paddle_id in predicted_by_id
    ]
    matched = len(pairs)
    definitive = [(truth, predicted) for truth, predicted in pairs if predicted.status != FinalStatus.UNCERTAIN]
    uncertain = matched - len(definitive)
    exact = sum(truth.status == predicted.status for truth, predicted in definitive)
    broken_pairs = [(truth, predicted) for truth, predicted in pairs if truth.status in _BROKEN_STATUSES]
    normal_pairs = [(truth, predicted) for truth, predicted in pairs if truth.status == FinalStatus.NORMAL]
    safety_alerts = sum(_is_break_alert(predicted) for _, predicted in broken_pairs)
    dangerous_false_normals = sum(predicted.status == FinalStatus.NORMAL for _, predicted in broken_pairs)
    normal_recalled = sum(predicted.status == FinalStatus.NORMAL for _, predicted in normal_pairs)
    false_break_alerts = sum(_is_break_alert(predicted) for _, predicted in normal_pairs)
    return ClassificationMetrics(
        matched_events=matched,
        definitive_events=len(definitive),
        uncertain_events=uncertain,
        coverage=len(definitive) / matched if matched else 0.0,
        uncertain_rate=uncertain / matched if matched else 0.0,
        selective_exact_accuracy=exact / len(definitive) if definitive else 0.0,
        broken_safety_alert_recall=safety_alerts / len(broken_pairs) if broken_pairs else 0.0,
        dangerous_false_normal_rate=dangerous_false_normals / len(broken_pairs) if broken_pairs else 0.0,
        normal_recall=normal_recalled / len(normal_pairs) if normal_pairs else 0.0,
        false_break_alert_rate_on_normal=false_break_alerts / len(normal_pairs) if normal_pairs else 0.0,
        confusion_pairs=tuple((truth.status.value, predicted.status.value) for truth, predicted in pairs),
    )


_BROKEN_STATUSES = frozenset({FinalStatus.BROKEN_LEFT, FinalStatus.BROKEN_RIGHT, FinalStatus.BROKEN_CENTER})


def _is_break_alert(predicted: PredictedEvent) -> bool:
    return predicted.status in _BROKEN_STATUSES or (predicted.status == FinalStatus.UNCERTAIN and predicted.suspected_breakage)
