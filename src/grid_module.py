from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GridDecision:
    add_allowed: bool
    next_level_price: float | None
    reason: str


@dataclass
class GridModule:
    enabled: bool
    max_levels: int
    spacing_atr_multiple: float
    max_total_risk_multiple: float

    def plan(
        self,
        side: str,
        current_levels: int,
        last_entry_price: float,
        atr_value: float,
        ai_approved: bool,
        total_risk_multiple: float,
    ) -> GridDecision:
        if not self.enabled:
            return GridDecision(False, None, "grid_disabled")
        if not ai_approved:
            return GridDecision(False, None, "ai_rejected_add")
        if current_levels >= self.max_levels:
            return GridDecision(False, None, "max_grid_levels")
        if total_risk_multiple >= self.max_total_risk_multiple:
            return GridDecision(False, None, "grid_risk_limit")
        spacing = atr_value * self.spacing_atr_multiple
        if spacing <= 0:
            return GridDecision(False, None, "invalid_atr_spacing")
        next_level_price = last_entry_price - spacing if side.upper() == "BUY" else last_entry_price + spacing
        return GridDecision(True, next_level_price, "approved")
