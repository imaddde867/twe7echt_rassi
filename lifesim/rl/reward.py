"""What "a good life" means to the learner. THIS IS THE ASSUMPTION THAT DECIDES
THE RESULT: whatever you weight here is what the agents will chase."""
from dataclasses import dataclass

import numpy as np


@dataclass
class Reward:
    w_survival: float = 1.0      # years lived / horizon
    w_wealth: float = 0.5        # tanh(final cash / cash_scale)
    w_health: float = 0.25       # final health / 100 (0 if dead)
    cash_scale: float = 20000.0

    def per_agent(self, outcomes: list[dict], years: float) -> np.ndarray:
        yrs = np.array([o["years_lived"] for o in outcomes]) / years
        cash = np.array([o["cash"] for o in outcomes])
        health = np.array([o["health"] for o in outcomes]) / 100
        return (self.w_survival * np.clip(yrs, 0, 1)
                + self.w_wealth * np.tanh(cash / self.cash_scale)
                + self.w_health * health)
