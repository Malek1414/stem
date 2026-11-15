"""Small-sample statistics for the benchmarks (stdlib only).

With 12-20 tasks a point score hides a wide range, so every report carries a 95% interval, and models
are compared task by task (paired), not by their averages. References: Miller 2024 "Adding error bars
to evals" (arXiv 2411.00640); Bowyer et al. 2025 (arXiv 2503.01747)."""
from __future__ import annotations

import math


def wilson(successes: float, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """95% Wilson interval for a proportion; works for fractional successes (mean of 0..1 scores)."""
    if n == 0:
        return None
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)


def mean_ci(values: list[float], z: float = 1.96) -> tuple[float, float] | None:
    """Normal-approximation 95% interval for a mean (used for 1-5 judge scores)."""
    n = len(values)
    if n < 2:
        return None
    m = sum(values) / n
    sd = math.sqrt(sum((v - m) ** 2 for v in values) / (n - 1))
    half = z * sd / math.sqrt(n)
    return round(m - half, 3), round(m + half, 3)


def sign_test(a: list[float], b: list[float]) -> dict:
    """Paired two-sided exact sign test over tasks both models ran. Ties are dropped."""
    wins = sum(1 for x, y in zip(a, b) if x > y)
    losses = sum(1 for x, y in zip(a, b) if x < y)
    n = wins + losses
    if n == 0:
        return {"wins": 0, "losses": 0, "ties": len(a), "p_value": 1.0}
    k = min(wins, losses)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return {"wins": wins, "losses": losses, "ties": len(a) - n, "p_value": round(p, 4)}
