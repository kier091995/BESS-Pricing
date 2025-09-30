# file: rainflow_sp.py
# Quellen: Kwon & Zhu, IEEE TSG (accepted). SP-basierte, instantane Degradationskosten
from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np

@dataclass
class SwitchingPoints:
    """Hält die letzten Switching Points (SPs) der SoC-Trajektorie.
    Wir führen eine Stack-Implementierung; c2 = letzter SP.
    Für die instantanen Kosten genügt c2; c0/c1 dienen als Features.
    """
    c_min: float
    c_max: float
    c_init: float
    stack: List[float] = field(default_factory=list)

    def __post_init__(self):
        self.stack = [self.c_init]  # initialer SP
        self._trim()

    def _trim(self, keep: int = 3):
        if len(self.stack) > keep:
            self.stack = self.stack[-keep:]

    @property
    def c0(self) -> float:
        if len(self.stack) >= 3:
            return self.stack[-3]
        return self.stack[0]

    @property
    def c1(self) -> float:
        if len(self.stack) >= 2:
            return self.stack[-2]
        return self.stack[0]

    @property
    def c2(self) -> float:
        return self.stack[-1]

    def maybe_add_sp(self, c_t: float, b_t: float) -> None:
        """Neuen SP erkennen wie in Eq. (11): b_t * (c_t - c2) < 0.
        Danach aktuellen SoC als neuen SP pushen und auf die letzten 3 SPs trimmen.
        """
        if b_t == 0.0:
            return
        if b_t * (c_t - self.c2) < 0.0:
            self.stack.append(float(np.clip(c_t, self.c_min, self.c_max)))
            self._trim()

@dataclass
class InstantDegradation:
    """Instantane Degradationskosten h_t^d nach Eq. (12).
    Φ(d) = α_d * exp(β * d), d = Zykeltiefe.
    """
    alpha_d: float = 4.5e-3
    beta: float = 1.3

    def increment(self, c_t: float, b_t: float, c2_t: float) -> float:
        """h^d_t = α(e^{β|c_t+b_t - c2|} - e^{β|c_t - c2|})"""
        return self.alpha_d * (
            np.exp(self.beta * np.abs((c_t + b_t) - c2_t))
            - np.exp(self.beta * np.abs(c_t - c2_t))
        )

# ---- Offline-Rainflow nur für Tests/Validierung ----
def rainflow_cycle_depths(soc: np.ndarray) -> List[float]:
    """Ein einfacher Rainflow-Zähler (Turning-Points-Stack).
    Liefert volle Zykeltiefen auf Basis von lokalen Extrema.
    Hinweis: Für die Validierung ausreichend; Produktionscode kann spezialisierte libs nutzen.
    """
    # Turning points extrahieren
    x = np.asarray(soc, dtype=float)
    if len(x) < 3:
        return []
    # lokale Extrema
    tp = [x[0]]
    for i in range(1, len(x)-1):
        if (x[i] - x[i-1]) * (x[i+1] - x[i]) <= 0:
            if tp[-1] != x[i]:
                tp.append(x[i])
    if tp[-1] != x[-1]:
        tp.append(x[-1])

    stack = []
    depths = []
    for y in tp:
        stack.append(y)
        while len(stack) >= 3:
            y0, y1, y2 = stack[-3], stack[-2], stack[-1]
            r1 = abs(y1 - y0)
            r2 = abs(y2 - y1)
            # Rainflow-Kriterium (vereinfachte 4-Punkt-Regel)
            if r2 >= r1:
                depths.append(r1)
                # entferne mittleren und vorletzten Punkt -> lasse y0 fallen, y1 bleibt Kante
                stack.pop(-2)  # remove y1
                stack.pop(-2)  # remove y0 (jetzt an -2)
            else:
                break
    return depths

def rainflow_cost(soc: np.ndarray, alpha_d: float, beta: float) -> float:
    depths = rainflow_cycle_depths(soc)
    return float(np.sum([alpha_d * np.exp(beta * d) for d in depths]))

def test_instant_vs_rainflow(seed: int = 0) -> Tuple[float, float, float]:
    """Mini-Test: Summe der instantanen Kosten ≈ Rainflow-Summe auf einer synthetischen SoC-Trajektorie.
    Exaktheit ist in Kwon & Zhu formal bewiesen (Proposition 1). Hier eine numerische Plausibilisierung. :contentReference[oaicite:3]{index=3}
    """
    rng = np.random.default_rng(seed)
    T = 200
    c_min, c_max = 0.1, 0.9
    c = 0.5
    sp = SwitchingPoints(c_min=c_min, c_max=c_max, c_init=c)
    deg = InstantDegradation()
    hist = [c]
    h_sum = 0.0
    for _ in range(T):
        # zufällige Ladung/Entladung (kleine Schritte)
        b = float(np.clip(rng.normal(0, 0.03), -0.08, 0.08))
        b = float(np.clip(b, c_min - c, c_max - c))  # SoC-Grenzen
        # instantane Kosten auf Basis c2_t
        h_sum += deg.increment(c_t=c, b_t=b, c2_t=sp.c2)
        # SP-Update (nach Eq. (11))
        sp.maybe_add_sp(c_t=c, b_t=b)
        c = float(np.clip(c + b, c_min, c_max))
        hist.append(c)
    rf = rainflow_cost(np.array(hist), deg.alpha_d, deg.beta)
    return h_sum, rf, abs(h_sum - rf) / max(1e-9, rf)
