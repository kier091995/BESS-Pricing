# file: features.py
import numpy as np
from collections import deque
from typing import Deque

def cyclical_time_features(t: int, hours_per_step: int = 1) -> np.ndarray:
    """Erzeuge (sin,cos)-Features für Stunde, Woche, Monat."""
    # Stunde 0..23; Woche 1..52; Monat 1..12
    # Wir nehmen t in Stunden als Basis.
    h = (t // hours_per_step) % 24
    week = ((t // (24 * 7)) % 52) + 1
    month = ((t // (24 * 30)) % 12) + 1
    h_sin, h_cos = np.sin(2 * np.pi * h / 24.0), np.cos(2 * np.pi * h / 24.0)
    w_sin, w_cos = np.sin(2 * np.pi * week / 52.0), np.cos(2 * np.pi * week / 52.0)
    m_sin, m_cos = np.sin(2 * np.pi * month / 12.0), np.cos(2 * np.pi * month / 12.0)
    return np.array([h_sin, h_cos, w_sin, w_cos, m_sin, m_cos], dtype=np.float32)

class ObsStacker:
    """FIFO-Stack für Observation-Stacking (K=4 standardmäßig)."""
    def __init__(self, obs_dim: int, K: int = 4):
        self.K = K
        self.buffer: Deque[np.ndarray] = deque(maxlen=K)
        self._zero = np.zeros(obs_dim, dtype=np.float32)

    def reset(self, first_obs: np.ndarray) -> np.ndarray:
        self.buffer.clear()
        for _ in range(self.K):
            self.buffer.append(first_obs.astype(np.float32))
        return self._stack()

    def push(self, obs: np.ndarray) -> np.ndarray:
        self.buffer.append(obs.astype(np.float32))
        return self._stack()

    def _stack(self) -> np.ndarray:
        return np.concatenate(list(self.buffer), axis=0).astype(np.float32)
