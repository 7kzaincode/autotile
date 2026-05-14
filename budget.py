"""
Translate a single "budget" choice into the bag of pipeline parameters that
will land near that piece count / cost target.

Tuned against empirical bunny/pepsi outputs at various resolutions. Not
exact — there's no closed-form prediction for piece count — but in the
right neighborhood.
"""

# Each row: target piece count, resolution, hollow on, region colors, max colors
PROFILES = {
    "tiny":     dict(piece_target=300,   resolution=24, hollow=True,  region_colors=4, max_colors=4,
                     cost_usd=40),
    "compact":  dict(piece_target=800,   resolution=32, hollow=True,  region_colors=6, max_colors=5,
                     cost_usd=80),
    "standard": dict(piece_target=1500,  resolution=40, hollow=True,  region_colors=8, max_colors=6,
                     cost_usd=150),
    "detailed": dict(piece_target=3000,  resolution=48, hollow=False, region_colors=12, max_colors=8,
                     cost_usd=300),
    "premium":  dict(piece_target=5000,  resolution=64, hollow=False, region_colors=16, max_colors=10,
                     cost_usd=500),
}


def resolve(budget: str) -> dict:
    """Return the param dict for a budget label. Unknown labels fall through
    to standard."""
    return PROFILES.get(budget, PROFILES["standard"]).copy()
