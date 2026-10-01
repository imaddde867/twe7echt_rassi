"""A tiny numpy MLP policy. No torch: the sim is CPU-bound Python, and ES
parallelises over plain processes, which suits a CPU cluster."""
import numpy as np

from ..actions import N_ACTIONS, Action
from ..agent import OBS, OBS_FIELDS

N_IN = len(OBS_FIELDS)
_SCALE = np.array([{"health": 100, "energy": 100, "satiety": 100, "mood": 100,
                    "cash": 1, "education": 100, "career": 100, "age": 80,
                    "job_level": 4}.get(name, 1) for name in OBS_FIELDS], dtype=np.float64)
_CASH = OBS["cash"]


def featurize(obs: np.ndarray) -> np.ndarray:
    """Roughly [-1, 1] inputs. Cash is log-squashed so 100 and 100000 are both usable."""
    x = obs / _SCALE
    c = obs[_CASH]
    x[_CASH] = np.sign(c) * np.log10(1 + abs(c)) / 5
    return x


class MLPPolicy:
    def __init__(self, hidden=(32, 32), rng: np.random.Generator | None = None):
        self.sizes = (N_IN, *hidden, N_ACTIONS)
        rng = rng or np.random.default_rng(0)
        self.W, self.b = [], []
        for i, o in zip(self.sizes[:-1], self.sizes[1:]):
            self.W.append(rng.normal(0, 1 / np.sqrt(i), (i, o)))
            self.b.append(np.zeros(o))

    # -- flat parameter vector (what ES perturbs) --------------------------
    @property
    def n_params(self) -> int:
        return sum(w.size + b.size for w, b in zip(self.W, self.b))

    def get_flat(self) -> np.ndarray:
        return np.concatenate([np.concatenate([w.ravel(), b]) for w, b in zip(self.W, self.b)])

    def set_flat(self, theta: np.ndarray) -> None:
        k = 0
        for j, (w, b) in enumerate(zip(self.W, self.b)):
            self.W[j] = theta[k:k + w.size].reshape(w.shape).copy()
            k += w.size
            self.b[j] = theta[k:k + b.size].copy()
            k += b.size

    @classmethod
    def from_flat(cls, theta: np.ndarray, hidden=(32, 32)) -> "MLPPolicy":
        p = cls(hidden)
        p.set_flat(np.asarray(theta, dtype=np.float64))
        return p

    # -- inference ---------------------------------------------------------
    def logits(self, x: np.ndarray) -> np.ndarray:
        for w, b in zip(self.W[:-1], self.b[:-1]):
            x = np.tanh(x @ w + b)
        return x @ self.W[-1] + self.b[-1]

    def act(self, obs: np.ndarray) -> Action:
        return Action(int(np.argmax(self.logits(featurize(obs)))))

    # -- supervised training (behaviour cloning) -----------------------------
    def fit(self, X: np.ndarray, y: np.ndarray, epochs=30, lr=3e-3, batch=256,
            rng: np.random.Generator | None = None) -> list[float]:
        """Cross-entropy with Adam, hand-written backprop. Returns loss per epoch."""
        rng = rng or np.random.default_rng(0)
        params = self.W + self.b
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        t, losses = 0, []
        n_layers = len(self.W)
        for _ in range(epochs):
            order = rng.permutation(len(X))
            total = 0.0
            for s in range(0, len(X), batch):
                idx = order[s:s + batch]
                xb, yb = X[idx], y[idx]
                acts = [xb]
                for j in range(n_layers - 1):
                    acts.append(np.tanh(acts[-1] @ self.W[j] + self.b[j]))
                z = acts[-1] @ self.W[-1] + self.b[-1]
                z -= z.max(1, keepdims=True)
                p = np.exp(z)
                p /= p.sum(1, keepdims=True)
                total += -np.log(p[np.arange(len(yb)), yb] + 1e-12).sum()
                d = p
                d[np.arange(len(yb)), yb] -= 1
                d /= len(yb)
                gW, gb = [None] * n_layers, [None] * n_layers
                for j in range(n_layers - 1, -1, -1):
                    gW[j] = acts[j].T @ d
                    gb[j] = d.sum(0)
                    if j > 0:
                        d = (d @ self.W[j].T) * (1 - acts[j] ** 2)
                t += 1
                for k, g in enumerate(gW + gb):
                    m[k] = 0.9 * m[k] + 0.1 * g
                    v[k] = 0.999 * v[k] + 0.001 * g * g
                    mh, vh = m[k] / (1 - 0.9 ** t), v[k] / (1 - 0.999 ** t)
                    params[k] -= lr * mh / (np.sqrt(vh) + 1e-8)
            losses.append(total / len(X))
        return losses


def save(path, theta, hidden=(32, 32), **extra) -> None:
    np.savez(path, theta=theta, hidden=np.array(hidden), **extra)


def load(path) -> MLPPolicy:
    d = np.load(path)
    return MLPPolicy.from_flat(d["theta"], tuple(int(h) for h in d["hidden"]))
