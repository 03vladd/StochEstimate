"""
Claude Haiku narration — generates a plain-English summary of a pair's OU analysis.
Called at the end of the analysis pipeline; result cached in pairs.narration.
"""

from __future__ import annotations

import math
import logging
from typing import Optional

log = logging.getLogger(__name__)


def _build_prompt(
    ticker1: str,
    ticker2: str,
    confidence_level: str,
    tests_passed: int,
    theta: float,
    mu: float,
    sigma: float,
    theta_ci: tuple[float, float],
    sigma_ci: tuple[float, float],
    z_score: Optional[float],
    signal_type: Optional[str],
) -> str:
    half_life = math.log(2) / theta if theta > 0 else float("inf")
    if half_life > 30:
        half_life_str = f"roughly {round(half_life / 7)} weeks"
    else:
        half_life_str = f"roughly {round(half_life)} trading days"

    ci_width_theta = theta_ci[1] - theta_ci[0]
    ci_width_sigma = sigma_ci[1] - sigma_ci[0]

    signal_line = ""
    if z_score is not None and signal_type is not None:
        direction = {
            "LONG": "below",
            "SHORT": "above",
            "EXIT": "near",
            "NONE": "within",
        }.get(signal_type, "at")
        signal_line = (
            f"The most recent z-score is {z_score:.2f}σ — the spread is currently {direction} its "
            f"long-run mean (signal: {signal_type})."
        )

    return f"""StochEstimate models stock-pair spreads as Ornstein-Uhlenbeck (OU) processes. \
Write a plain-English summary of the {ticker1}/{ticker2} analysis below. \
Three short paragraphs, no headers, no bullet points, no markdown.

Paragraph 1 — model fit: State whether the pair passed the OU tests and what that means in \
plain terms. {tests_passed}/4 core tests passed (confidence: {confidence_level}).

Paragraph 2 — current position: Describe where the spread sits right now. \
{signal_line if signal_line else f"The z-score is unavailable."} \
Keep it factual — say what the number means, not what to do.

Paragraph 3 — uncertainty: Note that the parameter estimates carry uncertainty \
(θ 95% CI width = {ci_width_theta:.4f}, σ 95% CI width = {ci_width_sigma:.4f}) and that \
the model reflects historical behaviour, not a forecast.

Style rules:
- 2 sentences per paragraph, max.
- No "investors" or "traders" — address the reader as "you" if needed.
- No financial advice language. No certainty claims.
- Conversational but precise. Start directly, no preamble.

Data:
- θ = {theta:.5f} → half-life {half_life_str}
- μ = {mu:.5f}, σ = {sigma:.5f}
- θ CI: [{theta_ci[0]:.5f}, {theta_ci[1]:.5f}]
- σ CI: [{sigma_ci[0]:.5f}, {sigma_ci[1]:.5f}]
- Estimated via LSTM-robust (200 MC Dropout samples)"""


def generate_narration(
    ticker1: str,
    ticker2: str,
    confidence_level: str,
    tests_passed: int,
    theta: float,
    mu: float,
    sigma: float,
    theta_ci: tuple[float, float],
    sigma_ci: tuple[float, float],
    z_score: Optional[float] = None,
    signal_type: Optional[str] = None,
    api_key: str = "",
) -> Optional[str]:
    """
    Returns the generated narration string, or None if the API key is absent or the call fails.
    Runs synchronously inside asyncio.to_thread — do NOT await Anthropic calls here.
    """
    if not api_key:
        log.warning("ANTHROPIC_API_KEY not set — skipping narration generation")
        return None

    try:
        import anthropic  # imported lazily so missing SDK doesn't crash the whole service

        client = anthropic.Anthropic(api_key=api_key)
        prompt = _build_prompt(
            ticker1, ticker2, confidence_level, tests_passed,
            theta, mu, sigma, theta_ci, sigma_ci, z_score, signal_type,
        )
        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text.strip()
    except Exception as exc:
        log.error("Narration generation failed: %s", exc)
        return None
