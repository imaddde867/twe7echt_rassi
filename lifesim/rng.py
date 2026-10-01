"""Environmental randomness that does not depend on what agents do.

Every draw is a pure function of (world seed, day, agent id, event kind, index),
built from a splitmix64 hash. So two runs with the same seed see the same
exogenous luck (illness rolls, layoffs, windfalls, mortality) even when their
trajectories diverge, when a conditional draw (severity, duration) fires in one
and not the other, or when a policy consumes random numbers. What still differs
between trajectories is how the same luck plays out, never the luck itself.
"""
import math

_MASK = (1 << 64) - 1
_INV = 1.0 / (1 << 53)

# event kinds: one independent stream each
ILLNESS, JOB_LOSS, WINDFALL, MORTALITY = 1, 2, 3, 4
SEVERITY, CHRONIC, UNEMPLOYED_DAYS, WINDFALL_AMOUNT = 5, 6, 7, 8
MATCH, MATCH_PICK, BIRTH, CHILD_SEX, CROSSOVER, MUTATION = 9, 10, 11, 12, 13, 14
MATCH_ORDER = 15


def _mix(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & _MASK
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _MASK
    return x ^ (x >> 31)


class EnvRng:
    def __init__(self, seed: int):
        self._seed_h = _mix(seed & _MASK)
        self._day_h = self._seed_h

    def begin_day(self, day: int) -> None:
        self._day_h = _mix(self._seed_h ^ (day & _MASK))

    def u(self, agent_id: int, kind: int, idx: int = 0) -> float:
        """Uniform [0, 1), a pure function of (seed, day, agent, kind, idx)."""
        h = _mix(self._day_h + agent_id)
        h = _mix(h + kind * 0x10001 + idx)
        return (h >> 11) * _INV

    def uniform(self, agent_id: int, kind: int, lo: float, hi: float) -> float:
        return lo + (hi - lo) * self.u(agent_id, kind)

    def integers(self, agent_id: int, kind: int, lo: int, hi: int) -> int:
        """Inclusive on both ends."""
        return lo + min(int(self.u(agent_id, kind) * (hi - lo + 1)), hi - lo)

    def normal(self, agent_id: int, kind: int, idx: int = 0) -> float:
        """Standard normal (Box-Muller from two keyed uniforms)."""
        u1, u2 = self.u(agent_id, kind, 2 * idx), self.u(agent_id, kind, 2 * idx + 1)
        return math.sqrt(-2.0 * math.log(1.0 - u1)) * math.cos(2.0 * math.pi * u2)

    def exponential(self, agent_id: int, kind: int, mean: float) -> float:
        return -mean * math.log(1.0 - self.u(agent_id, kind))
