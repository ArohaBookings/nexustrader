from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GridPermission:
    allowed: bool
    reason: str


def evaluate_grid_permission(
    *,
    timeframe: str,
    setup: str,
    side: str,
    entry_price: float,
    spread_points: float,
    max_spread_points: float,
    requires_m5: bool,
    step_points: float,
    point_size: float,
    max_legs: int,
    state: dict,
) -> GridPermission:
    setup_upper = str(setup or "").upper()
    if requires_m5 and str(timeframe).upper() != "M5":
        return GridPermission(False, "xau_grid_requires_tf_m5")
    if spread_points > max_spread_points:
        return GridPermission(False, "grid_spread_block")
    current_legs = int(state.get("grid_leg_index", 0))
    grid_side = str(state.get("grid_side", "")).upper()
    if "START" in setup_upper:
        if int(state.get("open_positions_estimate", 0)) > 0 or current_legs > 0:
            return GridPermission(False, "grid_cycle_already_active")
        return GridPermission(True, "grid_start_allowed")
    if "ADD" in setup_upper:
        if current_legs <= 0:
            return GridPermission(False, "grid_add_without_active_cycle")
        if current_legs >= max_legs:
            return GridPermission(False, "grid_max_legs_reached")
        if grid_side and grid_side != str(side).upper():
            return GridPermission(False, "grid_side_mismatch")
        last_entry = float(state.get("last_entry_price", 0.0))
        if last_entry > 0 and entry_price > 0:
            step_price = max(point_size, point_size * step_points)
            if abs(entry_price - last_entry) < step_price:
                return GridPermission(False, "grid_step_not_reached")
        return GridPermission(True, "grid_add_allowed")
    return GridPermission(True, "grid_manage_allowed")
