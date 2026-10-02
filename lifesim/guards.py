"""Refusals for tools whose statistics assume a fixed population."""


def require_fixed_population(cfg_kwargs: dict, who: str, instead: str) -> None:
    """With reproduction, descendants enter during the run (so `years_lived` over the whole horizon
    means different things per generation), the sample size is the number of adults ever created,
    and the dead keep cash that was already passed on in an estate (so summing cash double counts)."""
    if cfg_kwargs.get("reproduction"):
        raise ValueError(f"reproduction=True is not supported by {who}: its statistics assume a fixed "
                         f"population. {instead}")


def require_nominal_money(cfg_kwargs: dict, who: str, instead: str) -> None:
    """Real-money mode exists to keep money-denominated quantities meaningful over a long horizon. The RL
    policy only observes nominal cash (no price index) and the reward is a fixed nominal scale, so in this
    mode the learner could not observe the quantity being stabilised and the reward would change meaning
    as prices rise."""
    if cfg_kwargs.get("real_money"):
        raise ValueError(f"real_money=True is not supported by {who}: the policy observes nominal cash "
                         f"without a price index and the reward uses a fixed nominal scale. {instead}")
