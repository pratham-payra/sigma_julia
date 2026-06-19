"""
sigma_models.py
Actor and Critic networks for SIGMA implemented in TensorFlow/Keras.
Matches the Julia architecture exactly:
  Actor:  52 -> 128 -> 64 -> 64 -> 8  (ReLU, softmax output)
  Critic: 60 -> 128 -> 64 -> 1        (ReLU)

Also contains:
  - ATR matrix (8x16) and C4 rotation operators
  - All 5 utility functions (vectorised, TF-differentiable)
  - Replay buffer
  - C4 rotational augmentation
"""

import numpy as np
import tensorflow as tf

# ─── Constants ────────────────────────────────────────────────────────────────
STATE_DIM = 52
N_PHASES  = 8
N_MOVE    = 16
GAMMA     = 0.99
TAU_SOFT  = 0.01

# ATR matrix (8x16) — row order: AE1,AE2-W,AE3-W,AW1,AN1,AN2-S,AN3-S,AS1
# col order: EE,EN,EW,ES, NE,NN,NW,NS, WE,WN,WW,WS, SE,SN,SW,SS
ATR = np.array([
    [1,1,1,1, 0,0,0,0, 0,0,0,0, 0,0,0,0],  # AE1
    [0,1,1,0, 0,0,0,0, 1,0,0,1, 0,0,0,0],  # AE2-W
    [1,0,0,1, 0,0,0,0, 0,1,1,0, 0,0,0,0],  # AE3-W
    [0,0,0,0, 0,0,0,0, 1,1,1,1, 0,0,0,0],  # AW1
    [0,0,0,0, 1,1,1,1, 0,0,0,0, 0,0,0,0],  # AN1
    [0,0,0,0, 0,0,1,1, 0,0,0,0, 1,1,0,0],  # AN2-S
    [0,0,0,0, 1,1,0,0, 0,0,0,0, 0,0,1,1],  # AN3-S
    [0,0,0,0, 0,0,0,0, 0,0,0,0, 1,1,1,1],  # AS1
], dtype=np.float32)

ATR_tf = tf.constant(ATR, dtype=tf.float32)

# Q_TRANSITION (8x8): cyclic Markov prior
_QP = np.array([
    [0,1,0,0,0,0,0,0],
    [1,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,1],
    [0,0,0,0,1,0,0,0],
    [0,0,0,1,0,0,0,0],
    [0,0,0,0,0,0,1,0],
    [0,0,0,0,0,1,0,0],
    [0,0,1,0,0,0,0,0],
], dtype=np.float32)
_eps = 0.1
Q_TRANSITION = tf.constant((1.0 - _eps) * _QP + (_eps / 8.0), dtype=tf.float32)

# C4 rotation operators
RHO_PRIME = np.array([[0,1,0,0],[0,0,1,0],[0,0,0,1],[1,0,0,0]], dtype=np.float32)

def _make_rho16():
    R = np.zeros((16, 16), dtype=np.float32)
    for i in range(16):
        R[i, (i + 4) % 16] = 1.0
    return R

def _make_rho_a():
    # Block-diagonal: P for G1 (cyclic 4-cycle), S for G2+G3 (2-cycles)
    # G1 indices: 0,1,2,3 (AE1,AE2W,AE3W,AW1 -> cycle: 0->4->3->7->0 in full)
    # Matches paper Appendix B.2
    R = np.zeros((8, 8), dtype=np.float32)
    # G1 cycle: AE1(0)->AN1(4)->AW1(3)->AS1(7)->AE1(0)
    R[4, 0] = 1.0; R[3, 4] = 1.0; R[7, 3] = 1.0; R[0, 7] = 1.0
    # G2 swap: AE2-W(1)<->AN2-S(5)
    R[5, 1] = 1.0; R[1, 5] = 1.0
    # G3 swap: AE3-W(2)<->AN3-S(6)
    R[6, 2] = 1.0; R[2, 6] = 1.0
    return R

RHO16  = tf.constant(_make_rho16(), dtype=tf.float32)
RHO_A  = tf.constant(_make_rho_a(), dtype=tf.float32)
RHO_P  = tf.constant(RHO_PRIME, dtype=tf.float32)

# Reward weights
LAMBDA_VEC = np.array([1.0, 0.5, 2.0, 1.0, 3.0], dtype=np.float32)
ALPHA_VEC  = np.array([1.0, 0.5, 2.0, 1.0, 3.0], dtype=np.float32)

# ─── Actor Network ────────────────────────────────────────────────────────────
class ActorNetwork(tf.keras.Model):
    def __init__(self, state_dim=STATE_DIM, n_phases=N_PHASES,
                 hidden=(128, 64, 64), lr=3e-4):
        super().__init__()
        self.fc1 = tf.keras.layers.Dense(hidden[0], activation='relu',
                                          kernel_initializer='glorot_uniform')
        self.fc2 = tf.keras.layers.Dense(hidden[1], activation='relu',
                                          kernel_initializer='glorot_uniform')
        self.fc3 = tf.keras.layers.Dense(hidden[2], activation='relu',
                                          kernel_initializer='glorot_uniform')
        self.out = tf.keras.layers.Dense(n_phases,
                                          kernel_initializer='glorot_uniform')
        self.optimizer = tf.keras.optimizers.Adam(lr)
        self.build((None, state_dim))

    def call(self, s, training=False):
        x = self.fc1(s)
        x = self.fc2(x)
        x = self.fc3(x)
        logits = self.out(x)
        return tf.nn.softmax(logits, axis=-1)

    def get_logits(self, s):
        x = self.fc1(s)
        x = self.fc2(x)
        x = self.fc3(x)
        return self.out(x)


# ─── Critic Network ───────────────────────────────────────────────────────────
class CriticNetwork(tf.keras.Model):
    def __init__(self, state_dim=STATE_DIM, n_phases=N_PHASES,
                 hidden=(128, 64), lr=3e-4):
        super().__init__()
        self.fc1 = tf.keras.layers.Dense(hidden[0], activation='relu',
                                          kernel_initializer='glorot_uniform')
        self.fc2 = tf.keras.layers.Dense(hidden[1], activation='relu',
                                          kernel_initializer='glorot_uniform')
        self.out = tf.keras.layers.Dense(1,
                                          kernel_initializer='glorot_uniform')
        self.optimizer = tf.keras.optimizers.Adam(lr)
        self.build((None, state_dim + n_phases))

    def call(self, s, a, training=False):
        x = tf.concat([s, a], axis=-1)
        x = self.fc1(x)
        x = self.fc2(x)
        return tf.squeeze(self.out(x), axis=-1)

    def best_action(self, s):
        """Returns (best_action_idx, best_Q) for a single state."""
        s_batch = tf.tile(tf.expand_dims(s, 0), [N_PHASES, 1])
        a_batch = tf.eye(N_PHASES, dtype=tf.float32)
        q_vals  = self.call(s_batch, a_batch)
        best    = tf.argmax(q_vals).numpy()
        return int(best), float(q_vals[best].numpy())


# ─── Utility Functions (differentiable) ──────────────────────────────────────
def U_M(pi_t, pi_t1):
    """Markovian consistency: -||pi_{t+1} - Q^T pi_t||^2"""
    expected = tf.linalg.matvec(tf.transpose(Q_TRANSITION), pi_t)
    return -tf.reduce_sum(tf.square(pi_t1 - expected), axis=-1)

def U_S(pi_t, pi_t1):
    """Action smoothness: -||ATR^T pi_{t+1} - ATR^T pi_t||^2"""
    m_t  = tf.linalg.matvec(tf.transpose(ATR_tf), pi_t)
    m_t1 = tf.linalg.matvec(tf.transpose(ATR_tf), pi_t1)
    return -tf.reduce_sum(tf.square(m_t1 - m_t), axis=-1)

def U_Q(pi_t1, Mq):
    """Queue pressure: (ATR^T pi_{t+1})^T Mq / (max(Mq)+1)"""
    served = tf.linalg.matvec(tf.transpose(ATR_tf), pi_t1)
    return tf.reduce_sum(served * Mq, axis=-1) / (tf.reduce_max(Mq, axis=-1) + 1.0)

def U_W(pi_t1, Gamma):
    """Waiting time fairness: -||dir_served - Gamma/max(Gamma)||^2"""
    move = tf.linalg.matvec(tf.transpose(ATR_tf), pi_t1)  # (16,)
    move_4x4 = tf.reshape(move, [-1, 4, 4])
    dir_served = tf.reduce_sum(move_4x4, axis=-1)          # (batch, 4)
    max_g = tf.reduce_max(Gamma, axis=-1, keepdims=True) + 1e-6
    return -tf.reduce_sum(tf.square(dir_served - Gamma / max_g), axis=-1)

def U_E(pi_t, theta):
    """Emergency alignment: -||ATR^T pi_t - theta||^2"""
    move = tf.linalg.matvec(tf.transpose(ATR_tf), pi_t)
    no_emergency = tf.reduce_all(tf.equal(theta, 0.0), axis=-1)
    loss = -tf.reduce_sum(tf.square(move - theta), axis=-1)
    return tf.where(no_emergency, tf.zeros_like(loss), loss)

def compute_utilities(pi_t, pi_t1, Mq, Gamma, theta):
    return (U_M(pi_t, pi_t1), U_S(pi_t, pi_t1),
            U_Q(pi_t1, Mq),   U_W(pi_t1, Gamma), U_E(pi_t, theta))

def compute_reward(pi_t, pi_t1, Mq, Gamma, theta, alpha=ALPHA_VEC):
    utils = compute_utilities(pi_t, pi_t1, Mq, Gamma, theta)
    alpha_tf = tf.constant(alpha, dtype=tf.float32)
    return sum(alpha_tf[i] * utils[i] for i in range(5))


# ─── C4 Rotational Augmentation ───────────────────────────────────────────────
def rotate4(v, k):
    """Rotate 4-dim per-approach vector by k*90deg anticlockwise."""
    k = k % 4
    R = tf.linalg.matrix_power(RHO_P, k)
    return tf.linalg.matvec(R, v)

def rotate16(v, k):
    """Rotate 16-dim movement vector by k*90deg anticlockwise."""
    k = k % 4
    R = tf.linalg.matrix_power(RHO16, k)
    return tf.linalg.matvec(R, v)

def rotate_action(a, k):
    """Rotate 8-dim one-hot action vector by k*90deg anticlockwise."""
    k = k % 4
    R = tf.linalg.matrix_power(RHO_A, k)
    return tf.linalg.matvec(R, a)

def augment_transition(state, action_oh, reward, rewards_vec,
                       prev_phase_16, Mq, l_in, gamma, theta):
    """Returns list of 4 augmented transitions (k=0,1,2,3)."""
    augmented = []
    for k in range(4):
        ra  = rotate_action(action_oh, k).numpy()
        rp  = rotate16(prev_phase_16, k).numpy()
        rl  = rotate4(l_in, k).numpy()
        rg  = rotate4(gamma, k).numpy()
        rt  = rotate16(theta, k).numpy()
        rMq = rotate16(Mq, k).numpy()
        rMq = rMq - rMq.min()
        rs  = np.concatenate([rp, rMq, rg, rt])
        augmented.append((rs, ra, reward, rewards_vec, rp, rMq, rl, rg, rt))
    return augmented


# ─── Replay Buffer ─────────────────────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.states      = np.zeros((capacity, STATE_DIM), dtype=np.float32)
        self.actions     = np.zeros((capacity, N_PHASES),  dtype=np.float32)
        self.rewards     = np.zeros(capacity,              dtype=np.float32)
        self.next_states = np.zeros((capacity, STATE_DIM), dtype=np.float32)
        self.ptr  = 0
        self.size = 0

    def push(self, s, a, r, s_next):
        self.states[self.ptr]      = s
        self.actions[self.ptr]     = a
        self.rewards[self.ptr]     = r
        self.next_states[self.ptr] = s_next
        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.choice(self.size, size=min(batch_size, self.size), replace=False)
        return (tf.constant(self.states[idx]),
                tf.constant(self.actions[idx]),
                tf.constant(self.rewards[idx]),
                tf.constant(self.next_states[idx]))


# ─── Soft target update ────────────────────────────────────────────────────────
def soft_update(target: tf.keras.Model, source: tf.keras.Model, tau=TAU_SOFT):
    for t_var, s_var in zip(target.trainable_variables, source.trainable_variables):
        t_var.assign(tau * s_var + (1.0 - tau) * t_var)
