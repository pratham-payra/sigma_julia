"""
train.py
Standalone TF training script for SIGMA.
Mirrors the full Julia pipeline:
  1. Synthetic dataset generation (Poisson traffic + emergency injection)
  2. C4 rotational augmentation (4x dataset)
  3. Actor pretraining (CE + utility regularisation)
  4. Critic pretraining (Fitted Q-Iteration)
  5. Online actor-critic refinement

Usage:
  python train.py                        # full experiment
  python train.py --quick                # reduced episodes for testing
  python train.py --use-llm              # use LLaMA-2 for emergency encoding
  python train.py --save sigma_model     # save weights after training
  python train.py --eval sigma_model     # evaluate a saved model
"""

import argparse
import numpy as np
import tensorflow as tf
from scipy.stats import poisson as sp_poisson

from sigma_models import (ActorNetwork, CriticNetwork, ReplayBuffer,
                           STATE_DIM, N_PHASES, GAMMA, TAU_SOFT,
                           ATR, ATR_tf, compute_reward, compute_utilities,
                           LAMBDA_VEC, ALPHA_VEC, soft_update,
                           augment_transition)
from llm_encoder import EmergencyEncoder, encode_emergency_rule

# ─── Traffic simulation constants (matching Julia) ────────────────────────────
Q_MAX        = 80.0
DT           = 30.0
PE_EMERGENCY = 1.0 / 20.0
NU0          = 0.5
MU0          = 0.4
R_HARM       = 3
KR           = [0.1, 0.05, 0.08]
KAPPA_R      = [0.08, 0.04, 0.06]
WR_BASE      = [2*np.pi/3600, 2*np.pi/1800, 2*np.pi/900]
PHI_R        = [0.0, np.pi/4, np.pi/2]
PSI_R        = [np.pi/6, np.pi/3, 0.0]


def arrival_rate(t):
    nu = NU0 + sum(KR[r] * np.sin(WR_BASE[r] * t + PHI_R[r]) for r in range(R_HARM))
    return max(nu, 0.0)

def departure_rate(t):
    mu = MU0 + sum(KAPPA_R[r] * np.sin(WR_BASE[r] * t + PSI_R[r]) for r in range(R_HARM))
    return max(mu, 0.0)

def build_pressure_mask(l_in, l_out):
    Mq_in  = np.repeat(l_in,  4)
    Mq_out = np.tile(l_out,   4)
    Mp     = Mq_in - Mq_out
    Mq     = Mp - Mp.min()
    return Mq.astype(np.float32)

def step_traffic(l_in, l_out, gamma, phase_idx, t, rng):
    nu = arrival_rate(t)
    mu = departure_rate(t)
    phase_vec = ATR[phase_idx]

    for i in range(4):
        arr = rng.poisson(nu * DT)
        l_in[i] = min(l_in[i] + arr, Q_MAX)

    for j in range(4):
        served = any(phase_vec[(i * 4) + j] > 0 for i in range(4))
        if served:
            dep = rng.poisson(mu * DT)
            l_out[j] = min(l_out[j] + dep, Q_MAX)
            released = min(dep, sum(l_in[i] * phase_vec[i * 4 + j] for i in range(4)))
            for i in range(4):
                if phase_vec[i * 4 + j] > 0 and l_in[i] > 0:
                    rel_i = min(released, l_in[i])
                    l_in[i] -= rel_i
                    released -= rel_i
                    if released <= 0:
                        break

    Mq = build_pressure_mask(l_in, l_out)
    for i in range(4):
        gamma[i] = l_in[i] * DT
    return Mq

# ─── Transition dataclass ─────────────────────────────────────────────────────
class Transition:
    __slots__ = ['state','action_oh','reward','rewards_vec',
                 'prev_phase_16','Mq','l_in','gamma','theta']
    def __init__(self, state, action_oh, reward, rewards_vec,
                 prev_phase_16, Mq, l_in, gamma, theta):
        self.state         = np.array(state,         dtype=np.float32)
        self.action_oh     = np.array(action_oh,     dtype=np.float32)
        self.reward        = float(reward)
        self.rewards_vec   = np.array(rewards_vec,   dtype=np.float32)
        self.prev_phase_16 = np.array(prev_phase_16, dtype=np.float32)
        self.Mq            = np.array(Mq,            dtype=np.float32)
        self.l_in          = np.array(l_in,          dtype=np.float32)
        self.gamma         = np.array(gamma,         dtype=np.float32)
        self.theta         = np.array(theta,         dtype=np.float32)


# ─── Dataset generation ───────────────────────────────────────────────────────
def generate_episode(T, rng, encoder: EmergencyEncoder, alpha=ALPHA_VEC):
    l_in  = np.zeros(4, dtype=np.float32)
    l_out = np.zeros(4, dtype=np.float32)
    gamma = np.zeros(4, dtype=np.float32)

    has_emergency = rng.random() < PE_EMERGENCY
    if has_emergency:
        orig = rng.integers(0, 4)
        dest = rng.integers(0, 4)
        while dest == orig:
            dest = rng.integers(0, 4)
        dirs = ["East", "North", "West", "South"]
        instr = f"Emergency vehicle approaching from {dirs[orig]} toward {dirs[dest]}"
        theta = encoder.encode(instr)
    else:
        theta = np.zeros(16, dtype=np.float32)

    prev_phase_oh = np.zeros(N_PHASES, dtype=np.float32)
    prev_phase_oh[0] = 1.0
    transitions = []
    t = 0.0

    for _ in range(T):
        prev_phase_16 = (prev_phase_oh @ ATR).astype(np.float32)
        Mq = build_pressure_mask(l_in, l_out)
        s  = np.concatenate([prev_phase_16, Mq, gamma, theta])

        alpha_rand = rng.random(5).astype(np.float32)
        alpha_rand = alpha_rand / alpha_rand.sum() * 5.0

        r_vec = np.zeros(N_PHASES, dtype=np.float32)
        for a_idx in range(N_PHASES):
            pi_a = np.zeros(N_PHASES, dtype=np.float32); pi_a[a_idx] = 1.0
            pi_tf = tf.constant([pi_a], dtype=tf.float32)
            Mq_tf = tf.constant([Mq],    dtype=tf.float32)
            G_tf  = tf.constant([gamma], dtype=tf.float32)
            th_tf = tf.constant([theta], dtype=tf.float32)
            utils = compute_utilities(pi_tf, pi_tf, Mq_tf, G_tf, th_tf)
            r_vec[a_idx] = float(sum(alpha_rand[i] * utils[i][0].numpy()
                                     for i in range(5)))

        a_star = int(np.argmax(r_vec))
        a_oh   = np.zeros(N_PHASES, dtype=np.float32); a_oh[a_star] = 1.0

        pi_tf  = tf.constant([a_oh],  dtype=tf.float32)
        Mq_tf  = tf.constant([Mq],    dtype=tf.float32)
        G_tf   = tf.constant([gamma], dtype=tf.float32)
        th_tf  = tf.constant([theta], dtype=tf.float32)
        utils  = compute_utilities(pi_tf, pi_tf, Mq_tf, G_tf, th_tf)
        rv5    = np.array([float(utils[i][0].numpy()) for i in range(5)],
                          dtype=np.float32)

        transitions.append(Transition(s, a_oh, r_vec[a_star], rv5,
                                       prev_phase_16, Mq,
                                       l_in.copy(), gamma.copy(), theta))

        Mq = step_traffic(l_in, l_out, gamma, a_star, t, rng)
        prev_phase_oh = a_oh
        t += DT

    return transitions


def generate_dataset(N_episodes, T_steps, encoder, seed=42, augment=True):
    rng = np.random.default_rng(seed)
    D_prime = []
    for ep in range(N_episodes):
        D_prime.extend(generate_episode(T_steps, rng, encoder))
        if (ep + 1) % 20 == 0:
            print(f"  Episode {ep+1}/{N_episodes}  |  D' size: {len(D_prime)}")

    if not augment:
        return D_prime, D_prime

    DF = []
    for tr in D_prime:
        augs = augment_transition(
            tr.state, tr.action_oh, tr.reward, tr.rewards_vec,
            tr.prev_phase_16, tr.Mq, tr.l_in, tr.gamma, tr.theta)
        for (rs, ra, rew, rv5, rp, rMq, rl, rg, rt) in augs:
            DF.append(Transition(rs, ra, rew, rv5, rp, rMq, rl, rg, rt))
    return DF, D_prime


# ─── Actor pretraining ────────────────────────────────────────────────────────
def pretrain_actor(actor: ActorNetwork, DF, epochs=100, batch_size=256,
                   lv=LAMBDA_VEC, seed=0, verbose=True):
    rng = np.random.default_rng(seed)
    N   = len(DF)
    lv_tf = tf.constant(lv, dtype=tf.float32)

    for epoch in range(1, epochs + 1):
        idx   = rng.permutation(N)
        total = 0.0
        steps = 0
        for start in range(0, N, batch_size):
            b = [DF[i] for i in idx[start:start + batch_size]]
            s  = tf.constant(np.stack([t.state     for t in b]), dtype=tf.float32)
            a  = tf.constant(np.stack([t.action_oh for t in b]), dtype=tf.float32)
            Mq = tf.constant(np.stack([t.Mq        for t in b]), dtype=tf.float32)
            G  = tf.constant(np.stack([t.gamma     for t in b]), dtype=tf.float32)
            th = tf.constant(np.stack([t.theta     for t in b]), dtype=tf.float32)

            with tf.GradientTape() as tape:
                pi    = actor(s, training=True)
                ce    = -tf.reduce_mean(tf.reduce_sum(
                    a * tf.math.log(pi + 1e-9), axis=-1))
                utils = compute_utilities(pi, pi, Mq, G, th)
                u_reg = -sum(lv_tf[i] * tf.reduce_mean(utils[i]) for i in range(5))
                loss  = ce + u_reg

            grads = tape.gradient(loss, actor.trainable_variables)
            actor.optimizer.apply_gradients(zip(grads, actor.trainable_variables))
            total += float(loss.numpy())
            steps += 1

        if verbose and (epoch % 20 == 0 or epoch == 1):
            print(f"  [Actor Pretrain] Epoch {epoch:3d}/{epochs}  "
                  f"loss={total/steps:.4f}")


# ─── Critic pretraining (FQI) ─────────────────────────────────────────────────
def pretrain_critic(critic: ActorNetwork, DF, max_iter=30, gamma=GAMMA,
                    alpha=ALPHA_VEC, k=5, seed=0, verbose=True):
    rng = np.random.default_rng(seed)
    N   = len(DF)
    states  = np.stack([t.state     for t in DF])
    actions = np.stack([t.action_oh for t in DF])
    act_idx = np.array([int(np.argmax(t.action_oh)) for t in DF])

    # k-NN next states
    next_states = np.zeros_like(states)
    for i in range(N):
        same = np.where(act_idx == act_idx[i])[0]
        same = same[same != i]
        if len(same) == 0:
            next_states[i] = states[(i + 1) % N]
            continue
        dists  = np.linalg.norm(states[same] - states[i], axis=1)
        nn_idx = same[np.argsort(dists)[:k]]
        next_states[i] = states[(nn_idx + 1) % N].mean(axis=0)

    alpha_np = np.array(alpha, dtype=np.float32)
    Q_vals   = np.array([float((alpha_np * t.rewards_vec).sum()) for t in DF],
                        dtype=np.float32)

    ctarget = CriticNetwork()
    soft_update(ctarget, critic, tau=1.0)

    for it in range(1, max_iter + 1):
        Q_prev = Q_vals.copy()

        # Bellman targets
        for i in range(N):
            _, Qn    = ctarget.best_action(
                tf.constant(next_states[i], dtype=tf.float32))
            Q_vals[i] = float((alpha_np * DF[i].rewards_vec).sum()) + gamma * Qn

        # Regression step
        idx  = rng.permutation(N)
        loss_total = 0.0
        for i in idx:
            s_t = tf.constant([states[i]],  dtype=tf.float32)
            a_t = tf.constant([actions[i]], dtype=tf.float32)
            q_t = tf.constant([Q_vals[i]],  dtype=tf.float32)
            with tf.GradientTape() as tape:
                q_pred = critic(s_t, a_t, training=True)
                loss   = tf.reduce_mean(tf.square(q_t - q_pred))
            grads = tape.gradient(loss, critic.trainable_variables)
            critic.optimizer.apply_gradients(zip(grads, critic.trainable_variables))
            loss_total += float(loss.numpy())

        delta = float(np.linalg.norm(Q_vals - Q_prev))
        soft_update(ctarget, critic, tau=0.1)

        if verbose and (it % 10 == 0 or it == 1):
            print(f"  [Critic FQI] Iter {it:3d}/{max_iter}  "
                  f"ΔQ={delta:.6f}  loss={loss_total/N:.4f}")
        if delta < 1e-4:
            if verbose:
                print(f"  [Critic FQI] Converged at iter {it}")
            break


# ─── Online training ──────────────────────────────────────────────────────────
def run_online(actor, critic, actor_target, critic_target, replay,
               encoder, n_episodes=1000, T_steps=120, batch_size=64,
               seed=0, verbose=True):
    rng  = np.random.default_rng(seed)
    step = 0
    ep_rewards = []

    for ep in range(1, n_episodes + 1):
        l_in  = np.zeros(4, dtype=np.float32)
        l_out = np.zeros(4, dtype=np.float32)
        gamma = np.zeros(4, dtype=np.float32)

        has_em = rng.random() < PE_EMERGENCY
        if has_em:
            orig = int(rng.integers(0, 4))
            dest = int(rng.integers(0, 4))
            while dest == orig:
                dest = int(rng.integers(0, 4))
            dirs  = ["East", "North", "West", "South"]
            theta = encoder.encode(
                f"Emergency vehicle from {dirs[orig]} toward {dirs[dest]}")
        else:
            theta = np.zeros(16, dtype=np.float32)

        prev_oh = np.zeros(N_PHASES, dtype=np.float32); prev_oh[0] = 1.0
        total_r = 0.0
        t = 0.0

        for _ in range(T_steps):
            prev_16 = (prev_oh @ ATR).astype(np.float32)
            Mq      = build_pressure_mask(l_in, l_out)
            s       = np.concatenate([prev_16, Mq, gamma, theta])

            # Stochastic action sampling
            pi = actor(tf.constant([s], dtype=tf.float32), training=False)[0].numpy()
            a_idx = int(np.random.choice(N_PHASES, p=pi))
            a_oh  = np.zeros(N_PHASES, dtype=np.float32); a_oh[a_idx] = 1.0

            old_lin = l_in.copy()
            Mq_next = step_traffic(l_in, l_out, gamma, a_idx, t, rng)
            next_16 = (a_oh @ ATR).astype(np.float32)
            s_next  = np.concatenate([next_16, Mq_next, gamma, theta])

            pi_next = actor(tf.constant([s_next], dtype=tf.float32),
                            training=False)[0].numpy()
            pi_tf   = tf.constant([pi],      dtype=tf.float32)
            pn_tf   = tf.constant([pi_next], dtype=tf.float32)
            Mq_tf   = tf.constant([Mq],      dtype=tf.float32)
            G_tf    = tf.constant([gamma],   dtype=tf.float32)
            th_tf   = tf.constant([theta],   dtype=tf.float32)
            r = float(compute_reward(pi_tf, pn_tf, Mq_tf, G_tf, th_tf)[0].numpy())

            replay.push(s, a_oh, r, s_next)
            total_r += r

            if replay.size >= batch_size:
                sb, ab, rb, snb = replay.sample(batch_size)

                with tf.GradientTape() as tape:
                    a_next = tf.one_hot(
                        [critic_target.best_action(snb[i])[0]
                         for i in range(sb.shape[0])], N_PHASES, dtype=tf.float32)
                    Q_next = critic_target(snb, a_next, training=False)
                    y      = rb + GAMMA * Q_next
                    Q_pred = critic(sb, ab, training=True)
                    c_loss = tf.reduce_mean(tf.square(y - Q_pred))
                grads = tape.gradient(c_loss, critic.trainable_variables)
                critic.optimizer.apply_gradients(
                    zip(grads, critic.trainable_variables))

                with tf.GradientTape() as tape:
                    pi_b      = actor(sb, training=True)
                    Q_sa      = critic(sb, pi_b,  training=False)
                    pi_next_b = tf.stop_gradient(actor(snb, training=False))
                    Q_sn      = critic(snb, pi_next_b, training=False)
                    delta_b   = rb + GAMMA * Q_sn - Q_sa
                    a_loss    = -tf.reduce_mean(
                        delta_b * tf.reduce_sum(
                            pi_b * tf.math.log(pi_b + 1e-9), axis=-1))
                grads = tape.gradient(a_loss, actor.trainable_variables)
                actor.optimizer.apply_gradients(
                    zip(grads, actor.trainable_variables))

                step += 1
                if step % 10 == 0:
                    soft_update(actor_target,  actor,  TAU_SOFT)
                    soft_update(critic_target, critic, TAU_SOFT)

            prev_oh = a_oh
            t += DT

        ep_rewards.append(total_r)
        if verbose and (ep % 100 == 0 or ep == 1):
            print(f"  [Online] Ep {ep:4d}/{n_episodes}  R={total_r:.2f}  "
                  f"avg100={np.mean(ep_rewards[-100:]):.2f}")

    return ep_rewards


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="SIGMA TF Training")
    parser.add_argument("--quick",     action="store_true")
    parser.add_argument("--use-llm",   action="store_true")
    parser.add_argument("--llm-model", type=str,
                        default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--device",    type=str, default="cpu")
    parser.add_argument("--save",      type=str, default=None)
    parser.add_argument("--eval",      type=str, default=None)
    parser.add_argument("--seed",      type=int, default=42)
    args = parser.parse_args()

    if args.quick:
        N_DATA, T_S, A_EP, FQI, N_ON = 30, 60, 15, 8, 100
    else:
        N_DATA, T_S, A_EP, FQI, N_ON = 200, 120, 100, 30, 1000

    encoder = EmergencyEncoder(use_llm=args.use_llm,
                                model_name=args.llm_model,
                                device=args.device)

    actor         = ActorNetwork()
    critic        = CriticNetwork()
    actor_target  = ActorNetwork()
    critic_target = CriticNetwork()
    soft_update(actor_target,  actor,  tau=1.0)
    soft_update(critic_target, critic, tau=1.0)
    replay = ReplayBuffer()

    if args.eval:
        actor.load_weights(args.eval + "_actor.weights.h5")
        critic.load_weights(args.eval + "_critic.weights.h5")
        print(f"Model loaded from {args.eval}")
        return

    print("\n[1/4] Generating synthetic dataset...")
    DF, D_prime = generate_dataset(N_DATA, T_S, encoder, seed=args.seed)
    print(f"      D'={len(D_prime)}  DF={len(DF)} (after C4 augmentation)")

    print("\n[2/4] Pretraining Actor...")
    pretrain_actor(actor, DF, epochs=A_EP, seed=args.seed)

    print("\n[3/4] Pretraining Critic (FQI)...")
    pretrain_critic(critic, DF, max_iter=FQI, seed=args.seed)

    print("\n[4/4] Online Actor-Critic refinement...")
    run_online(actor, critic, actor_target, critic_target, replay,
               encoder, n_episodes=N_ON, T_steps=T_S, seed=args.seed + 1)

    if args.save:
        actor.save_weights(args.save + "_actor.weights.h5")
        critic.save_weights(args.save + "_critic.weights.h5")
        print(f"\nModels saved to {args.save}_actor/critic.weights.h5")

    print("\nDone.")


if __name__ == "__main__":
    main()
