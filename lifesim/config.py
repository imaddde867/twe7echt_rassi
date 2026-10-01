from dataclasses import dataclass, field


@dataclass
class Config:
    seed: int = 0
    n_agents: int = 100
    years: int = 50
    slots_per_day: int = 3          # actions an agent can take per day
    log_every: int = 30             # snapshot interval in days
    start_age: tuple = (18, 40)     # uniform initial age range, years
    study_frac_range: tuple = (0.0, 1.0)     # gene ranges drawn uniformly at start
    cash_buffer_range: tuple = (50.0, 400.0)
    patience_range: tuple = (0.3, 2.0)       # UtilityPolicy genes
    wealth_weight_range: tuple = (0.5, 2.0)

    # economy
    base_wage: float = 25.0         # per WORK slot
    eat_cost: float = 15.0
    social_cost: float = 5.0
    daily_living_cost: float = 20.0  # rent etc., charged once per day
    start_cash: float = 200.0
    start_education: tuple = (10.0, 60.0)   # adults start with differing schooling

    # life stages (layer 1). Turn off to get the v1 behaviour back.
    life_stages: bool = True
    retire_age: float = 65.0
    pension_frac: float = 0.5           # share of average lifetime daily earnings
    elder_energy_recovery: float = 20.0  # nightly energy gain once retired (normal: 30)

    # random shocks (layer 2). Turn off to get the pre-shock behaviour back.
    shocks: bool = True
    illness_rate: float = 0.6            # per year at health 100, age <= 40
    illness_severity: tuple = (5, 30)    # uniform health damage if untreated
    healthcare_cost_per_severity: float = 25.0
    treatment_health_fraction: float = 0.3   # damage kept if you can pay for treatment
    chronic_severity: float = 20.0       # severity at/above which chronic illness can start
    chronic_prob: float = 0.01
    chronic_daily_cost: float = 4.0
    chronic_daily_health_drain: float = 0.02
    job_loss_rate: float = 0.06          # per year while working age and employed
    unemployed_days: tuple = (30, 180)
    unemployment_benefit: float = 10.0   # per day
    career_loss_on_layoff: float = 0.9   # career multiplier
    windfall_rate: float = 0.03          # per year
    windfall_mean: float = 3000.0

    # economy (layer 3). Turn off to get the pre-economy behaviour back.
    # With it on, the flat pay/rent rules are replaced by jobs, tax, lifestyle
    # creep, interest and a credit limit. Neutral values: inflation=0,
    # wage_growth=0, tax_brackets=(), lifestyle_frac=0, savings_rate=0.
    economy: bool = True
    # (min education, min career, pay multiplier) per job tier, lowest first
    job_tiers: tuple = ((0, 0, 0.6), (10, 0, 1.0), (35, 0, 1.6),
                        (60, 10, 2.6), (80, 30, 4.0))
    # (daily-earnings upper bound in day-0 money, marginal rate)
    tax_brackets: tuple = ((40, 0.10), (100, 0.30), (float("inf"), 0.45))
    lifestyle_frac: float = 0.4   # share of income above baseline that gets spent
    inflation: float = 0.02       # per year, on prices
    wage_growth: float = 0.02     # per year, on pay
    savings_rate: float = 0.015   # per year on positive cash
    debt_rate: float = 0.12       # per year on negative cash
    credit_limit: float = 2000.0  # cannot buy food past this much debt
    welfare_daily: float = 25.0   # safety net paid per day once debt hits the credit limit
    debt_stress: float = 0.3      # daily health loss while cash < 0 (1.0 without the economy)

    # mortality: annual hazard = gompertz_a * exp(gompertz_b * age)
    gompertz_a: float = 0.00003
    gompertz_b: float = 0.095
    # Per-sex multiplier on mortality. Neutral on purpose: real-world gaps are
    # an assumption, so turn them on explicitly when you want to test them.
    sex_mortality_mult: dict = field(default_factory=lambda: {"M": 1.0, "F": 1.0})

    @property
    def days(self) -> int:
        return self.years * 365
