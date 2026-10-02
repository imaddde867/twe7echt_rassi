"""Refusals for tools whose statistics assume a fixed population."""


def require_fixed_population(cfg_kwargs: dict, who: str, instead: str) -> None:
    """With reproduction, descendants enter during the run (so `years_lived` over the whole horizon
    means different things per generation), the sample size is the number of adults ever created,
    and the dead keep cash that was already passed on in an estate (so summing cash double counts)."""
    if cfg_kwargs.get("reproduction"):
        raise ValueError(f"reproduction=True is not supported by {who}: its statistics assume a fixed "
                         f"population. {instead}")
