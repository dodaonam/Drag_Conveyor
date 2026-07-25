from __future__ import annotations

from dataclasses import dataclass

from .acceptance import AcceptanceMetrics, ClassificationMetrics, GroundTruthEvent, PredictedEvent
from .decision import FinalStatus
from .statistics import clopper_pearson_one_sided


_BROKEN = frozenset({FinalStatus.BROKEN_LEFT, FinalStatus.BROKEN_RIGHT, FinalStatus.BROKEN_CENTER})


@dataclass(frozen=True, slots=True)
class AcceptanceGate:
    name: str
    successes: int
    trials: int
    observed: float | None
    bound: float | None
    threshold: float
    direction: str
    passed: bool


@dataclass(frozen=True, slots=True)
class BootstrapAcceptanceReport:
    gates: tuple[AcceptanceGate, ...]
    passed: bool


def evaluate_bootstrap_acceptance(
    predictions: tuple[PredictedEvent, ...],
    ground_truth: tuple[GroundTruthEvent, ...],
    events: AcceptanceMetrics,
    classifications: ClassificationMetrics,
) -> BootstrapAcceptanceReport:
    """Evaluate the measurable bootstrap gates using one-sided 95% exact bounds."""
    predicted = {item.paddle_id: item for item in predictions}
    truth = {item.event_id: item for item in ground_truth if not item.partial_boundary}
    pairs = [(truth[item.ground_truth_id], predicted[item.predicted_paddle_id]) for item in events.matches if item.ground_truth_id in truth and item.predicted_paddle_id in predicted]
    gates = [
        _minimum("event_precision", events.matched_events, events.matched_events + events.extra_events, .99),
        _minimum("event_recall", events.matched_events, events.matched_events + events.missed_events, .99),
        _maximum("duplicate_event_rate", events.duplicate_events, events.matched_events + events.extra_events, .005),
        _minimum("selective_exact_label_precision", sum(a == b for a, b in classifications.confusion_pairs if b != FinalStatus.UNCERTAIN.value), classifications.definitive_events, .98),
        _minimum("broken_safety_alert_recall", sum(_break_alert(predicted_event) for truth_event, predicted_event in pairs if truth_event.status in _BROKEN), sum(truth_event.status in _BROKEN for truth_event, _ in pairs), .99),
        _maximum("dangerous_false_normal_rate", sum(predicted_event.status == FinalStatus.NORMAL for truth_event, predicted_event in pairs if truth_event.status in _BROKEN), sum(truth_event.status in _BROKEN for truth_event, _ in pairs), .01),
        _minimum("normal_recall", sum(predicted_event.status == FinalStatus.NORMAL for truth_event, predicted_event in pairs if truth_event.status == FinalStatus.NORMAL), sum(truth_event.status == FinalStatus.NORMAL for truth_event, _ in pairs), .95),
        _maximum("false_break_alert_rate_on_normal", sum(_break_alert(predicted_event) for truth_event, predicted_event in pairs if truth_event.status == FinalStatus.NORMAL), sum(truth_event.status == FinalStatus.NORMAL for truth_event, _ in pairs), .02),
        _maximum("overall_uncertain_rate", classifications.uncertain_events, classifications.matched_events, .10),
    ]
    for status in sorted(_BROKEN, key=lambda value: value.value):
        members = [(truth_event, predicted_event) for truth_event, predicted_event in pairs if truth_event.status == status]
        gates.append(_minimum(f"{status.value}_exact_location_recall", sum(predicted_event.status == status for _, predicted_event in members), len(members), .90))
        gates.append(_minimum(f"{status.value}_definitive_coverage", sum(predicted_event.status != FinalStatus.UNCERTAIN for _, predicted_event in members), len(members), .90))
    return BootstrapAcceptanceReport(tuple(gates), all(gate.passed for gate in gates))


def _minimum(name: str, successes: int, trials: int, threshold: float) -> AcceptanceGate:
    if trials == 0:
        return AcceptanceGate(name, successes, trials, None, None, threshold, "minimum", False)
    lower, _ = clopper_pearson_one_sided(successes, trials)
    return AcceptanceGate(name, successes, trials, successes / trials, lower, threshold, "minimum", lower >= threshold)


def _maximum(name: str, failures: int, trials: int, threshold: float) -> AcceptanceGate:
    if trials == 0:
        return AcceptanceGate(name, failures, trials, None, None, threshold, "maximum", False)
    _, upper = clopper_pearson_one_sided(failures, trials)
    return AcceptanceGate(name, failures, trials, failures / trials, upper, threshold, "maximum", upper <= threshold)


def _break_alert(prediction: PredictedEvent) -> bool:
    return prediction.status in _BROKEN or (prediction.status == FinalStatus.UNCERTAIN and prediction.suspected_breakage)
