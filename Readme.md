# Rainflow‑aware DQN for Battery Dispatch

This repository implements a **single, production‑ready code path** for training and evaluating a Deep Q‑Network (DQN) to dispatch a battery while (1) accounting for **cycle‑based degradation** via rainflow and (2) using **experiment features that reliably improve DRL performance** (cyclic time encodings and observation stacking). The design merges two lines of evidence: a rainflow‑exact MDP formulation for degradation and a comparative study of DRL design choices for battery dispatch.

---

## Why this code looks the way it does (scientific background)

### Cycle‑based degradation, made Markov

Battery wear depends on **cycle depth**, not just instantaneous power. The rainflow algorithm identifies cycles using the last switching points (SPs) of the state‑of‑charge (SoC) trajectory. By **augmenting the MDP state** with the last three SP SoC values (c_t^{(0)}, c_t^{(1)}, c_t^{(2)}), we can compute an **instantaneous** degradation increment that is **exactly equivalent** to the original cycle cost (no linearization). The instantaneous term we implement is:

```
h_t^d(b_t,c_t,c_t^{(2)}) = α_d e^{β|c_t+b_t - c_t^{(2)}|} - α_d e^{β|c_t - c_t^{(2)}|}
```

which follows Eq. (12) and allows standard RL training with per‑step rewards. See **Eq. (12)** for the formula, **Table II** for the SP update rules (NRa, NRb, RA), and **Algorithm 1** for the training loop.
*Pointers:* Eq. (12) (p. 4); Table II (p. 5) shows how the SP states are advanced at a new switching point; Algorithm 1 (p. 6) lists the full DQN‑based control routine; Lemma 1 and Fig. 1 (p. 3) illustrate why three SPs suffice for rainflow.

The reward we maximize is: **energy revenue – FR deviation penalty – degradation increment**. The FR signal can be down‑sampled during training (e.g., from seconds to minutes) and replayed at native rate in evaluation, as suggested in **Remark 1**.

### Why DQN, cyclic time features, and stacking

Across two battery‑dispatch case studies, **DQN consistently outperformed PPO, SAC, and DDPG** when tuned well. On top of the algorithm choice, two *simple* design features improved results **reliably**:

1. **Cyclic time encodings** (sine/cosine for hour, week, month), and
2. **Observation stacking** (adding recent observations to the state).

Table 3 reports DQN at or near the top; Table 4 shows that adding sine/cosine time counters and stacking observations **increased returns**, while reward penalties mostly did **not** help. Figures B.6 and B.7 visualize the gains and how stacking depth matters; Fig. 4 shows stable learning curves for the best settings.
*Pointers:* Table 3 (p. 8) compares models; Table 4 (p. 9) summarizes the effect of time counters, stacking, and penalties; Fig. 4 (p. 10) shows training stability; Appendix B, **Fig. B.6/B.7** (p. 12) detail time‑feature and stacking effects; **Table C.7** (p. 13) lists DQN hyperparameters and ranges used during tuning.

### Safety layer and discrete actions

We use **discrete actions** (charge/idle/discharge or a finer grid) with a **safety layer** that clips illegal actions to the nearest feasible command (respecting power and SoC bounds). This exact mechanism appears in Eq. (17) and proved robust in the comparative study. Discretization is **not a disadvantage** here and often simplifies learning.

---

## What this repository contains

```
.
├─ configs/
│  └─ default.yaml                  # One canonical config (see below)
├─ data/                            # Your timeseries CSVs live here (you provide)
├─ src/
│  ├─ envs/
│  │  ├─ battery_env.py             # Gymnasium-style env with unified reward
│  │  └─ rainflow_state.py          # SP tracker + Table II state transitions
│  ├─ features/
│  │  ├─ time_encoding.py           # Sine/cos encoders for hour/week/month
│  │  └─ stacker.py                 # Observation stacking wrapper
│  ├─ utils/
│  │  ├─ safety_layer.py            # Action clipping (Eq. 17 logic)
│  │  └─ metrics.py                 # KPIs: revenue/cost, DoD cycles, wear, etc.
│  ├─ agents/
│  │  └─ dqn_agent.py               # DQN with replay + target network
│  ├─ training/
│  │  ├─ train.py                   # End-to-end training script
│  │  └─ evaluate.py                # Deterministic rollout + report
│  └─ io/
│     └─ loaders.py                 # CSV readers, schema checks
└─ README.md
```

### Script responsibilities (single path that runs end‑to‑end)

* **`src/envs/battery_env.py`**
  *What it does:* A Gymnasium‑compatible environment for two tasks behind one API:
  * **Arbitrage** (revenue maximization), optionally with **FR penalty**.
  * **Load‑following / RE utilization** (cost minimization as negative reward).
  
  *How it works:*
  * State (s_t): ([p_t, f_t, SoC_t, c_t^{(0)}, c_t^{(1)}, c_t^{(2)}, time features, (stacked obs)]) per Eq. (13), extended with cyclic time encodings and stacks. The three SPs make degradation Markovian.
  * Reward: r_t = market cashflow - δ|f_t - b_t| - h_t^d with h_t^d from Eq. (12). FR down‑sampling for training is supported as in Remark 1.
  * Termination: end of dataset (episodic); hard safety via the safety layer (no illegal transitions).

* **`src/envs/rainflow_state.py`**
  *What it does:* Encapsulates the **switching‑point tracker** and updates (c_t^{(0..2)}) using the **NRa/NRb/RA** rules (**Table II**) and exposes the exact **instantaneous wear increment** (h_t^d) (**Eq. (12)**).

* **`src/features/time_encoding.py`**
  *What it does:* Adds **sine/cosine encodings** for hour‑of‑day, week‑of‑year, month‑of‑year; these **increase returns** by helping the agent catch periodicity in prices/renewables (see **Table 4** and **Fig. B.6**).

* **`src/features/stacker.py`**
  *What it does:* Concatenates the last *k* observations to the current state. **Stacking improves performance** (see **Table 4** and **Fig. B.7**). We expose `k` in the config and use a tested value by default.

* **`src/utils/safety_layer.py`**
  *What it does:* Implements the **action‑correction layer**: clips actions to SoC and power limits (and RE charging limit for LF/REU) following **Eq. (17)**. This keeps the environment within physical bounds without extra reward penalties.

* **`src/agents/dqn_agent.py`**
  *What it does:* **One DQN** with experience replay and a fixed‑interval target network—mirroring **Algorithm 1**. Epsilon‑greedy decays during training. Hyperparameter ranges follow the comparative study (see **Table C.7**); defaults are set to robust values for both tasks.

* **`src/training/train.py`**
  *What it does:* Wires everything together: loads data, builds env with time encodings + stacking, wraps safety layer, and trains DQN; saves checkpoints and a metrics report.

* **`src/training/evaluate.py`**
  *What it does:* Loads a trained policy, runs a deterministic rollout at full temporal resolution, and prints per‑episode KPIs (revenues/cost, degradation, cycle counts), including plots if desired.

* **`src/io/loaders.py`**
  *What it does:* Validates and loads CSV time series; ensures columns exist for the chosen task (see below).

---

## Data you provide

Place CSVs in `./data`. Timestamps should be uniform and sorted.

* **Arbitrage (+/− FR):**
  `timestamp, price, [fr_signal]`
  * `price`: market price per timestep.
  * `fr_signal` (optional): normalized request (f_t ∈ [-1,1]) for FR penalty.

* **Load‑following / RE utilization:**
  `timestamp, price, demand, re_gen`
  * `demand`: site load; `re_gen`: available renewable generation.

---

## How to run

### 1) Install

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Configure (single canonical config)

`configs/default.yaml` (excerpt):

```yaml
task: arbitrage              # or: lf_reu
action_grid: [ -1.0, 0.0, 1.0 ]   # discrete actions (scaled to power bounds)
env:
  dt_hours: 1.0
  soc_min: 0.2
  soc_max: 0.8
  p_charge_max_mw: 2.5
  p_discharge_max_mw: 2.5
  eta_charge: 0.92
  eta_discharge: 0.92
  fr_penalty_delta: 0.0        # set >0 to activate FR deviation penalty
  alpha_d: 4.5e-3
  beta: 1.3
features:
  time_cyclic: {hour: true, week: true, month: true}
  stack_obs: 4
agent:
  # Defaults guided by Table C.7; chosen to work across tasks
  learning_rate: 0.0005
  batch_size: 256
  buffer_size: 500000
  gamma: 0.975
  target_update_interval: 1000
  tau: 0.3
  train_freq: 128
  learning_starts: 2500
  exploration:
    fraction: 0.2
    init_eps: 1.0
    final_eps: 0.01
training:
  total_steps: 1_000_000
  seed: 0
```

*Why these choices:* the **discrete action grid** with a **safety layer** mirrors the setups that worked best; **cyclic time features** (hour/week/month) and **stacked observations** are enabled because they **consistently improved returns** in controlled comparisons (Table 4, Fig. B.6/B.7).

### 3) Train

```bash
python -m src.training.train --config configs/default.yaml --data ./data/your_file.csv
```

### 4) Evaluate (full‑rate FR optional)

```bash
python -m src.training.evaluate --checkpoint runs/best.pt --data ./data/your_file.csv
```

---

## How the pieces interact

```
time series ─┐
             ├─ loaders → env (SoC, price, FR/RE, SPs) → safety layer → DQN
config  ─────┘                    ↑  rainflow_state (Eq. 12, Table II)
                                  ↑  time_encoding + stacker
```

* The **environment** exposes a Markov state with SPs and returns a per‑step reward that includes the **exact** instantaneous degradation increment (Eq. 12).
* The **DQN loop** uses replay and a target network (Algorithm 1).
* **Cyclic time features** and **stacking** improve the agent's anticipation of exogenous signals (prices, RE) (Table 4, Fig. B.6/B.7).

---

## Defaults and tuning notes

* **Hyperparameters.** We ship one set of defaults that works across the two task families. If you tune, use the **ranges** demonstrated in **Table C.7** (learning rate, batch size, target update interval, etc.) and keep the two features (cyclic time + stacking) on; they delivered **the largest consistent gains** (Table 4).
* **Reward penalties.** This code **does not** add extra penalty terms (e.g., "keep SoC high") because controlled tests showed **little or negative effect** and sometimes unintended behavior; the safety layer is sufficient (Table 4).
* **FR rate:** Train with a down‑sampled FR signal (faster learning); evaluate at native rate if needed (Remark 1).

---

## Outputs

* **Training logs:** episode return, moving averages.
* **Evaluation report:** total revenue/cost, FR penalty, **degradation cost**, cycle counts and depths, and SoC trace plots.
* **Reproducibility:** set `training.seed`.

---

## Limitations

* The degradation model implemented is the **DoD/rainflow exponential** with coefficients (α_d,β); adjust these to match your cell chemistry if known. The exact mapping is discussed in the underlying formulation (Eq. (8) with Φ(d)=α_d e^{βd}).
* Compute needs are modest for DQN; for context, GA‑MPC baselines with long horizons can be orders of magnitude slower (Table 5).

---

## Key references behind this implementation

* **Rainflow‑exact degradation in an MDP + DQN training loop.** State augmentation with three switching points; instantaneous wear increment (**Eq. 12**); **Table II** (SP updates); **Algorithm 1** (training). Pages 3–6.
* **What reliably helps in practice.** DQN baselines; **cyclic time counters** (hour/week/month) and **observation stacking** improve rewards across tasks; **penalties** generally unhelpful. See **Table 3** (model comparison), **Table 4** (design choices), **Fig. 4** (learning curves), **Fig. B.6/B.7** (ablation), **Table C.7** (DQN hyperparams). Pages 8–13.

---

*If anything in your dataset doesn't match the expected schemas above, the loader will raise a clear error. Everything else runs from this README as‑is.*