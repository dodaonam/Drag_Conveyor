from __future__ import annotations

import math


def clopper_pearson_one_sided(successes: int, trials: int, *, confidence: float = .95) -> tuple[float, float]:
    """Exact one-sided Clopper--Pearson lower and upper bounds for a binomial rate."""
    if not isinstance(successes, int) or not isinstance(trials, int) or not 0 <= successes <= trials or trials <= 0:
        raise ValueError("successes and trials must be integers with 0 <= successes <= trials")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    alpha = 1.0 - confidence
    lower = 0.0 if successes == 0 else _beta_inverse(alpha, successes, trials - successes + 1)
    upper = 1.0 if successes == trials else _beta_inverse(1.0 - alpha, successes + 1, trials - successes)
    return lower, upper


def _beta_inverse(probability: float, a: float, b: float) -> float:
    low, high = 0.0, 1.0
    for _ in range(80):
        midpoint = (low + high) / 2.0
        if _regularized_beta(midpoint, a, b) < probability:
            low = midpoint
        else:
            high = midpoint
    return (low + high) / 2.0


def _regularized_beta(x: float, a: float, b: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    factor = math.exp(a * math.log(x) + b * math.log1p(-x) - math.lgamma(a) - math.lgamma(b) + math.lgamma(a + b))
    if x < (a + 1.0) / (a + b + 2.0):
        return factor * _beta_continued_fraction(x, a, b) / a
    return 1.0 - factor * _beta_continued_fraction(1.0 - x, b, a) / b


def _beta_continued_fraction(x: float, a: float, b: float) -> float:
    tiny = 1e-300
    c = 1.0
    d = 1.0 - (a + b) * x / (a + 1.0)
    d = tiny if abs(d) < tiny else d
    d = 1.0 / d
    result = d
    for iteration in range(1, 201):
        twice = 2 * iteration
        coefficient = iteration * (b - iteration) * x / ((a + twice - 1.0) * (a + twice))
        d = 1.0 + coefficient * d
        d = tiny if abs(d) < tiny else d
        c = 1.0 + coefficient / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        result *= d * c
        coefficient = -(a + iteration) * (a + b + iteration) * x / ((a + twice) * (a + twice + 1.0))
        d = 1.0 + coefficient * d
        d = tiny if abs(d) < tiny else d
        c = 1.0 + coefficient / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        delta = d * c
        result *= delta
        if abs(delta - 1.0) < 3e-14:
            break
    return result
