# file: market_env.py
# Multi-Market-Environment mit:
# - Energie-Arbitrage (Preis pt)
# - Reserve/Ancillary Services: Vergütung über reserve_price[t] * rho
# - FR-Tracking-Penalty: δ * |rho * f_t - b_t|
# - Safety-Layer: SoC-/Leistungs-/Reserve-Margen

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Dict, Any, Tuple

from rainflow_sp import SwitchingPoints, InstantDegradation
from features import cyclical_time_features, ObsStacker

class BessMultiMarketEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        price: np.ndarray,                 # pt
        fr_signal: Optional[np.ndarray] = None,  # f_t in [-1,1]
        reserve_price: Optional[np.ndarray] = None,  # Vergütung je Schritt
        dt_hours: float = 1.0,
        c_min: float = 0.1,
        c_max: float = 0.9,
        c_init: float = 0.5,
        b_max: float = 0.08,               # max ΔSoC pro Schritt (≈ Pmax*dt/C)
        delta_fr: float = 5.0,             # FR-Penalty-Koeffizient δ
        alpha_d: float = 4.5e-3, beta: float = 1.3,  # Degradation
        stack_K: int = 4,
        seed: int = 0,
    ):
        super().__init__()
        assert price.ndim == 1
        self.price = price.astype(np.float32)
        self.T = len(price)
        self.fr = (fr_signal if fr_signal is not None else np.zeros_like(price)).astype(np.float32)
        self.res_price = (reserve_price if reserve_price is not None else np.zeros_like(price)).astype(np.float32)

        self.dt = float(dt_hours)
        self.c_min, self.c_max = float(c_min), float(c_max)
        self.c = float(c_init)
        self.b_max = float(b_max)
        self.delta_fr = float(delta_fr)

        # SP/Degradation
        self.sp = SwitchingPoints(c_min=self.c_min, c_max=self.c_max, c_init=self.c)
        self.deg = InstantDegradation(alpha_d=alpha_d, beta=beta)

        # Aktionsraum: 17 Leistungsstufen x 4 Reserve-Quoten
        self.power_levels = np.linspace(-self.b_max, self.b_max, 17, dtype=np.float32)
        self.rho_levels = np.array([0.0, 0.1, 0.2, 0.3], dtype=np.float32)
        self.nA = len(self.power_levels) * len(self.rho_levels)
        self.action_space = spaces.Discrete(self.nA)

        # Beobachtung (ohne Stacking): [SoC, p_t, f_t, c0, c1, c2, 6 Zeitfeatures] = 1+1+1+3+6 = 12
        self.obs_dim = 12
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim * stack_K,), dtype=np.float32
        )
        self.stacker = ObsStacker(obs_dim=self.obs_dim, K=stack_K)

        # Laufvariablen
        self.t = 0
        self.rng = np.random.default_rng(seed)

    def _decode_action(self, a: int) -> Tuple[float, float]:
        i_rho = a % len(self.rho_levels)
        i_pow = a // len(self.rho_levels)
        return float(self.power_levels[i_pow]), float(self.rho_levels[i_rho])

    def _safety_layer(self, b_set: float, rho: float) -> float:
        """Projiziere auf zulässige Aktion: SoC- und Reservemarge."""
        # Reservemarge: Headroom für ±rho*b_max; also nutzbare Bandbreite: (1 - rho)*b_max
        b_limited = np.clip(b_set, -(1.0 - rho) * self.b_max, (1.0 - rho) * self.b_max)

        # SoC-Grenzen mit Reservepuffer: c ∈ [c_min + rho*b_max, c_max - rho*b_max]
        c_lo = self.c_min + rho * self.b_max
        c_hi = self.c_max - rho * self.b_max
        # maximal zulässige Schritte in beide Richtungen
        max_up = max(0.0, c_hi - self.c)  # Laden (b>0)
        max_dn = max(0.0, self.c - c_lo)  # Entladen (b<0)
        if b_limited > 0:
            b_limited = min(b_limited, max_up)
        elif b_limited < 0:
            b_limited = -min(abs(b_limited), max_dn)
        return float(b_limited)

    def _base_obs(self) -> np.ndarray:
        time_feats = cyclical_time_features(self.t, hours_per_step=int(self.dt))
        obs = np.array(
            [self.c, self.price[self.t], self.fr[self.t], self.sp.c0, self.sp.c1, self.sp.c2, *time_feats],
            dtype=np.float32,
        )
        return obs

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.t = 0
        self.c = float(np.clip(0.5, self.c_min, self.c_max))
        self.sp = SwitchingPoints(c_min=self.c_min, c_max=self.c_max, c_init=self.c)
        first = self._base_obs()
        return self.stacker.reset(first), {}

    def step(self, action: int):
        assert self.t < self.T, "Episode already ended"
        b_set, rho = self._decode_action(int(action))
        # Safety
        b = self._safety_layer(b_set, rho)

        # Instantane Degradation (Eq. (12)) mit c2_t vor Update
        h_d = self.deg.increment(c_t=self.c, b_t=b, c2_t=self.sp.c2)

        # Arbitrage-Kosten he_t = p_t * b_t (Kwon & Zhu, Eq. (6)); Reward = -he
        he = self.price[self.t] * b

        # FR-Tracking-Penalty nur wenn Reserve gebucht ist (ρ>0): δ * |ρ f_t - b_t|
        fr_target = rho * self.fr[self.t] * self.b_max  # skaliere f_t∈[-1,1] auf ΔSoC und auf ρ
        hf = self.delta_fr * abs(fr_target - b)

        # Reserve-Vergütung (einfach linear)
        r_res = self.res_price[self.t] * rho

        # Reward (maximieren): Arbitrage-Gewinn - FR-Penalty - Degradation + Reserve
        reward = (-he) - hf - h_d + r_res

        # SP-Update nach Eq. (11) und SoC-Update (ct+1 = ct + bt)
        self.sp.maybe_add_sp(c_t=self.c, b_t=b)
        self.c = float(np.clip(self.c + b, self.c_min, self.c_max))

        # Zeit
        self.t += 1
        terminated = self.t >= self.T
        obs = self._base_obs()
        obs = self.stacker.push(obs)
        info = {
            "b": b,
            "rho": rho,
            "he_cost": he,
            "fr_penalty": hf,
            "deg_cost": h_d,
            "reserve_reward": r_res,
            "soc": self.c,
        }
        return obs, float(reward), terminated, False, info
