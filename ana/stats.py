from __future__ import annotations

import math


def percentile(values: list[int | float], p: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(item) for item in values)
    rank = ((p / 100) * (len(ordered) - 1)) + 1
    if rank <= 1:
        return float(ordered[0])
    if rank >= len(ordered):
        return float(ordered[-1])
    floor_rank = math.floor(rank)
    ceil_rank = math.ceil(rank)
    if floor_rank == ceil_rank:
        return float(ordered[floor_rank - 1])
    low = ordered[floor_rank - 1]
    high = ordered[ceil_rank - 1]
    fraction = rank - floor_rank
    return float(low + ((high - low) * fraction))
