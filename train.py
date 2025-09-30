# file: train.py
# Training eines DQN-Agents (stable-baselines3) auf dem Environment.


import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import set_random_seed
import torch as th
import torch.nn as nn

from market_env import BessMultiMarketEnv

def make_synthetic_series(T: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    # Preis (€/MWh) mit Tagesmuster
    hours = np.arange(T)
    base = 60 + 20 * np.sin(2*np.pi*hours/24) + 10 * np.sin(2*np.pi*hours/168)
    noise = rng.normal(0, 5, size=T)
    price = np.clip(base + noise, 10, 200).astype(np.float32)
    # FR-Signal ~ weißes Rauschen in [-1,1] (vgl. Kwon & Zhu Remark 1) 
    fr = np.clip(rng.normal(0, 0.7, size=T), -1, 1).astype(np.float32)
    # Reserve-Preis (einfaches Profil)
    res_price = (2.0 + 1.0 * (np.sin(2*np.pi*hours/24) > 0)).astype(np.float32)
    return price, fr, res_price

def main():
    set_random_seed(42)
    T = 24 * 60  # 60 Tage stündlich
    price, fr, res_price = make_synthetic_series(T, seed=7)

    def _env_fn():
        return BessMultiMarketEnv(
            price=price, fr_signal=fr, reserve_price=res_price,
            dt_hours=1.0, c_min=0.1, c_max=0.9, c_init=0.5,
            b_max=0.08, delta_fr=5.0, alpha_d=4.5e-3, beta=1.3,  # Kwon & Zhu Defaults 
            stack_K=4, seed=123
        )

    env = DummyVecEnv([_env_fn])


    policy_kwargs = dict(
        activation_fn=nn.LeakyReLU,
        net_arch=[256, 512, 512, 256],
    )


    model = DQN(
        "MlpPolicy", env,
        learning_rate=4.8e-4,  # linearer Decay ist optional; konstanter Wert funktioniert zuverlässig
        buffer_size=500_000,
        learning_starts=2_500,
        batch_size=256,
        gamma=0.975,
        train_freq=128,
        target_update_interval=10_000,
        exploration_fraction=0.5,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.01,
        gradient_steps=-1,  # as in SB3 for online updates
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=42,
        tensorboard_log=None,
        device="auto",
    )

    print("Train...")
    model.learn(total_timesteps=250_000, log_interval=50)

    # Kurze Evaluation
    env_eval = _env_fn()
    mean_r, std_r = evaluate_policy(model, env_eval, n_eval_episodes=3)
    print(f"Eval reward mean={mean_r:.2f} ± {std_r:.2f}")

    # Beispielschritt für KPI-Ausgabe
    obs, _ = env_eval.reset()
    kpis = {"he": 0.0, "fr": 0.0, "deg": 0.0, "res": 0.0}
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, done, trunc, info = env_eval.step(int(action))
        kpis["he"] += info["he_cost"]
        kpis["fr"] += info["fr_penalty"]
        kpis["deg"] += info["deg_cost"]
        kpis["res"] += info["reserve_reward"]
        done = done or trunc
    print("KPIs over episode:")
    for k, v in kpis.items():
        print(f"  {k}: {v:.3f}")

if __name__ == "__main__":
    main()
