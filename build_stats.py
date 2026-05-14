"""
Aggregate stats for a bricks payload: count, complexity, support score,
rough cost/time estimates. The numbers are heuristic — fine for back-of-the-
envelope, not for binding customer quotes.
"""

from __future__ import annotations

from collections import Counter

from parts_list import parts_list


# Rough BrickLink retail prices (USD per piece, new condition). These are
# blended averages across colors; the real range is $0.02 (1x1 in common
# color) up to $0.40 (rarer specialty parts). Tune as you collect real data.
PRICE_HINT = {
    "1x1": 0.05, "1x2": 0.07, "1x3": 0.10, "1x4": 0.12,
    "1x6": 0.18, "1x8": 0.25, "1x10": 0.30, "1x12": 0.40, "1x16": 0.60,
    "2x2": 0.10, "2x3": 0.15, "2x4": 0.20,
    "2x6": 0.30, "2x8": 0.40, "2x10": 0.55,
    "4x4": 0.30, "4x6": 0.50, "4x8": 0.75, "4x10": 1.00,
}

# Kind multipliers — plates/tiles are typically cheaper per unit footprint than
# bricks; slopes are typically more expensive.
KIND_PRICE_MULTIPLIER = {
    "brick":     1.0,
    "plate":     0.6,
    "tile":      0.7,
    "slope":     1.3,
    "slope_inv": 1.4,
}

# Assembly time: ~10 seconds per brick is the AFOL-builder rule of thumb,
# bumped to 15s for a novice and 25s for kids.
SEC_PER_BRICK = 15


def difficulty(total_bricks: int, unique_skus: int,
               distinct_colors: int, kind_counts: dict) -> tuple[str, int]:
    """Return ('Easy'|'Medium'|'Hard'|'Expert', score 0-100).

    Score combines piece count (most weight), color count, and SKU diversity.
    Slope / inverted-slope pieces nudge difficulty up because they require
    spatial orientation thinking when building.
    """
    score = 0
    score += min(40, total_bricks // 50)            # 1 point per 50 bricks, capped at 40
    score += min(20, unique_skus * 2)                # SKU variety
    score += min(15, distinct_colors * 2)            # color tracking
    score += min(15, (kind_counts.get("slope", 0)
                      + kind_counts.get("slope_inv", 0)) // 10)
    score += min(10, kind_counts.get("plate", 0) // 50)
    score = max(0, min(100, score))
    if score < 25:    label = "Easy"
    elif score < 50:  label = "Medium"
    elif score < 75:  label = "Hard"
    else:             label = "Expert"
    return label, score


def build_stats(payload: dict) -> dict:
    bricks = payload.get("bricks", [])
    rows = parts_list(payload)

    types = Counter(b["brick_type"] for b in bricks)
    colors = Counter(b["color"] for b in bricks)
    palette = {p["id"]: p for p in payload.get("palette", [])}
    color_names = {cid: palette.get(cid, {}).get("name", "Unknown") for cid in colors}

    est_cost = sum(
        PRICE_HINT.get(b["brick_type"], 0.10)
        * KIND_PRICE_MULTIPLIER.get(b.get("kind", "brick"), 1.0)
        for b in bricks
    )
    est_seconds = SEC_PER_BRICK * len(bricks)
    est_minutes = est_seconds // 60

    floating = _floating_count(bricks)
    support_score = 1.0 if not bricks else 1.0 - (floating / len(bricks))

    kind_counts = Counter(b.get("kind", "brick") for b in bricks)
    diff_label, diff_score = difficulty(
        len(bricks), len(rows), len(colors), dict(kind_counts),
    )

    return {
        "total_bricks": len(bricks),
        "unique_skus": len(rows),
        "by_brick_type": dict(types.most_common()),
        "by_kind": dict(kind_counts),
        "by_color": {color_names[c]: n for c, n in colors.most_common()},
        "distinct_colors": len(colors),
        "floating_bricks": floating,
        "support_score": round(support_score, 3),
        "estimated_cost_usd": round(est_cost, 2),
        "estimated_build_time_minutes": int(est_minutes),
        "difficulty": diff_label,
        "difficulty_score": diff_score,
        "grid_shape": payload.get("grid_shape"),
        "resolution": payload.get("resolution"),
    }


def _floating_count(bricks: list[dict]) -> int:
    occ = set()
    for b in bricks:
        for dx in range(b["size_x"]):
            for dy in range(b["size_y"]):
                occ.add((b["x"] + dx, b["y"] + dy, b["z"]))
    floating = 0
    for b in bricks:
        if b["z"] <= 0:
            continue
        supported = False
        for dx in range(b["size_x"]):
            for dy in range(b["size_y"]):
                if (b["x"] + dx, b["y"] + dy, b["z"] - 1) in occ:
                    supported = True
                    break
            if supported:
                break
        if not supported:
            floating += 1
    return floating
