from dataclasses import dataclass, field

import numpy as np

# Order of the observation vector. Policies index into this, so an RL network
# later sees exactly the same thing a rule-based policy does.
OBS_FIELDS = ("health", "energy", "satiety", "mood", "cash",
              "education", "career", "age", "is_female", "retired")
OBS = {name: i for i, name in enumerate(OBS_FIELDS)}


@dataclass
class Agent:
    id: int
    sex: str                 # "M" or "F"
    age_days: int
    health: float = 90.0     # 0..100, 0 = dead
    energy: float = 80.0     # 0..100
    satiety: float = 70.0    # 0..100
    mood: float = 60.0       # 0..100
    cash: float = 200.0
    education: float = 10.0  # 0..100
    career: float = 0.0      # 0..100
    alive: bool = True
    genes: dict = field(default_factory=dict)  # strategy parameters
    start_cash: float = 200.0
    start_age_days: int = 0
    retired: bool = False
    lifetime_earnings: float = 0.0
    pension_daily: float = 0.0

    @property
    def age(self) -> float:
        return self.age_days / 365.0

    def observe(self) -> np.ndarray:
        return np.array([self.health, self.energy, self.satiety, self.mood,
                         self.cash, self.education, self.career, self.age,
                         1.0 if self.sex == "F" else 0.0,
                         1.0 if self.retired else 0.0], dtype=np.float64)

    def clamp(self) -> None:
        for f in ("health", "energy", "satiety", "mood", "education", "career"):
            setattr(self, f, min(100.0, max(0.0, getattr(self, f))))
