from __future__ import annotations


def news_override_allowed(
    *,
    news_mode: str,
    probability: float,
    confluence: float,
    spread_points: float,
    spread_reference: float,
    high_conf_probability: float,
    high_conf_confluence: float,
    high_conf_spread_mult: float,
) -> tuple[bool, str]:
    mode = str(news_mode or "SAFE").upper()
    if mode == "OFF":
        return True, "news_mode_off"
    if mode != "ALLOW_HIGH_CONF":
        return False, "news_blocked"
    if probability < high_conf_probability:
        return False, "news_prob_below_override"
    if confluence < high_conf_confluence:
        return False, "news_confluence_below_override"
    if spread_points > (spread_reference * high_conf_spread_mult):
        return False, "news_spread_blocked"
    return True, "news_high_conf_override"
