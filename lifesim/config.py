from dataclasses import dataclass, field


@dataclass
class Config:
    seed: int = 0
    n_agents: int = 100
    years: int = 50
    slots_per_day: int = 3          # actions an agent can take per day
    log_every: int = 30             # snapshot interval in days
    start_age: tuple = (18, 40)     # uniform initial age range, years

    # economy
    base_wage: float = 25.0         # per WORK slot
    eat_cost: float = 15.0
    social_cost: float = 5.0
    daily_living_cost: float = 20.0  # rent etc., charged once per day
    start_cash: float = 200.0

    # life stages (layer 1). Turn off to get the v1 behaviour back.
    life_stages: bool = True
    retire_age: float = 65.0
    pension_frac: float = 0.5           # share of average lifetime daily earnings
    elder_energy_recovery: float = 20.0  # nightly energy gain once retired (normal: 30)

    # mortality: annual hazard = gompertz_a * exp(gompertz_b * age)
    gompertz_a: float = 0.00003
    gompertz_b: float = 0.095
    # Per-sex multiplier on mortality. Neutral on purpose: real-world gaps are
    # an assumption, so turn them on explicitly when you want to test them.
    sex_mortality_mult: dict = field(default_factory=lambda: {"M": 1.0, "F": 1.0})

    @property
    def days(self) -> int:
        return self.years * 365
