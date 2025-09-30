# file: test_degradation.py
from rainflow_sp import test_instant_vs_rainflow

if __name__ == "__main__":
    h_sum, rf, rel_err = test_instant_vs_rainflow(seed=0)
    print(f"Instant sum   : {h_sum:.6f}")
    print(f"Rainflow total: {rf:.6f}")
    print(f"Rel. error    : {100*rel_err:.4f}%  (sollte sehr klein sein; formale Gleichheit: Proposition 1) ")
