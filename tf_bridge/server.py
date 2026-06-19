"""
server.py
JSON socket server — bridges Julia ↔ Python TF.

Julia calls this server for:
  - encode_emergency(instruction)  → 16-dim θ
  - actor_forward(state)           → 8-dim π
  - critic_forward(state, action)  → scalar Q
  - actor_train_step(batch)        → loss
  - critic_train_step(batch)       → loss
  - save_model(path)
  - load_model(path)

Protocol: newline-delimited JSON over TCP.
  Request:  {"cmd": "...", "args": {...}}\n
  Response: {"ok": true, "result": ...}\n  or  {"ok": false, "error": "..."}\n

Start server:
  python server.py [--port 9876] [--use-llm] [--llm-model meta-llama/...]
"""

import argparse
import json
import os
import socket
import threading
import traceback

import numpy as np
import tensorflow as tf

from sigma_models import (ActorNetwork, CriticNetwork, ReplayBuffer,
                           STATE_DIM, N_PHASES, GAMMA, TAU_SOFT,
                           compute_reward, compute_utilities,
                           LAMBDA_VEC, ALPHA_VEC, ATR_tf, Q_TRANSITION,
                           soft_update, augment_transition)
from llm_encoder import EmergencyEncoder

DEFAULT_PORT = 9876


class SIGMAServer:
    def __init__(self, use_llm=False, llm_model="meta-llama/Llama-2-7b-chat-hf",
                 device="cpu"):
        print("[Server] Initialising TF models...")
        self.actor        = ActorNetwork()
        self.actor_target = ActorNetwork()
        self.critic        = CriticNetwork()
        self.critic_target = CriticNetwork()
        soft_update(self.actor_target,  self.actor,  tau=1.0)
        soft_update(self.critic_target, self.critic, tau=1.0)

        self.replay  = ReplayBuffer()
        self.encoder = EmergencyEncoder(use_llm=use_llm, model_name=llm_model,
                                         device=device)
        self._update_step = 0
        print("[Server] Ready.")

    # ── Command handlers ──────────────────────────────────────────────────────

    def encode_emergency(self, instruction: str):
        theta = self.encoder.encode(instruction)
        return theta.tolist()

    def actor_forward(self, state: list):
        s = tf.constant([state], dtype=tf.float32)
        pi = self.actor(s, training=False)[0]
        return pi.numpy().tolist()

    def critic_forward(self, state: list, action: list):
        s = tf.constant([state],  dtype=tf.float32)
        a = tf.constant([action], dtype=tf.float32)
        q = self.critic(s, a, training=False)
        return float(q.numpy()[0])

    def push_transition(self, state, action, reward, next_state):
        self.replay.push(
            np.array(state,      dtype=np.float32),
            np.array(action,     dtype=np.float32),
            float(reward),
            np.array(next_state, dtype=np.float32)
        )
        return self.replay.size

    def train_step(self, batch_size=64):
        if self.replay.size < batch_size:
            return {"actor_loss": 0.0, "critic_loss": 0.0}

        s, a, r, s_next = self.replay.sample(batch_size)

        # ── Critic TD update ──────────────────────────────────────────────────
        with tf.GradientTape() as tape:
            a_next_idx = [self.critic_target.best_action(s_next[i])[0]
                          for i in range(batch_size)]
            a_next = tf.one_hot(a_next_idx, N_PHASES, dtype=tf.float32)
            Q_next = self.critic_target(s_next, a_next, training=False)
            y      = r + GAMMA * Q_next
            Q_pred = self.critic(s, a, training=True)
            c_loss = tf.reduce_mean(tf.square(y - Q_pred))

        grads = tape.gradient(c_loss, self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(
            zip(grads, self.critic.trainable_variables))

        # ── Actor PG update ───────────────────────────────────────────────────
        with tf.GradientTape() as tape:
            pi      = self.actor(s, training=True)
            Q_sa    = self.critic(s, pi,    training=False)
            Q_snext = self.critic(s_next, tf.stop_gradient(
                self.actor(s_next, training=False)), training=False)
            delta   = r + GAMMA * Q_snext - Q_sa
            a_loss  = -tf.reduce_mean(delta * tf.reduce_sum(
                pi * tf.math.log(pi + 1e-9), axis=-1))

        grads = tape.gradient(a_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(
            zip(grads, self.actor.trainable_variables))

        # ── Soft target update ────────────────────────────────────────────────
        self._update_step += 1
        if self._update_step % 10 == 0:
            soft_update(self.actor_target,  self.actor,  TAU_SOFT)
            soft_update(self.critic_target, self.critic, TAU_SOFT)

        return {"actor_loss": float(a_loss.numpy()),
                "critic_loss": float(c_loss.numpy())}

    def pretrain_actor_batch(self, states, actions, Mqs, gammas, thetas):
        s  = tf.constant(states,  dtype=tf.float32)
        a  = tf.constant(actions, dtype=tf.float32)
        Mq = tf.constant(Mqs,    dtype=tf.float32)
        G  = tf.constant(gammas,  dtype=tf.float32)
        th = tf.constant(thetas,  dtype=tf.float32)
        lv = tf.constant(LAMBDA_VEC, dtype=tf.float32)

        with tf.GradientTape() as tape:
            pi     = self.actor(s, training=True)
            # Cross-entropy loss
            ce     = -tf.reduce_mean(tf.reduce_sum(a * tf.math.log(pi + 1e-9), axis=-1))
            # Utility regularisation
            utils  = compute_utilities(pi, pi, Mq, G, th)
            u_loss = -sum(lv[i] * tf.reduce_mean(utils[i]) for i in range(5))
            loss   = ce + u_loss

        grads = tape.gradient(loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(
            zip(grads, self.actor.trainable_variables))
        return float(loss.numpy())

    def pretrain_critic_batch(self, states, actions, q_targets):
        s  = tf.constant(states,   dtype=tf.float32)
        a  = tf.constant(actions,  dtype=tf.float32)
        qt = tf.constant(q_targets, dtype=tf.float32)

        with tf.GradientTape() as tape:
            q_pred = self.critic(s, a, training=True)
            loss   = tf.reduce_mean(tf.square(qt - q_pred))

        grads = tape.gradient(loss, self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(
            zip(grads, self.critic.trainable_variables))
        return float(loss.numpy())

    def save_model(self, path: str):
        self.actor.save_weights(path + "_actor.weights.h5")
        self.critic.save_weights(path + "_critic.weights.h5")
        return path

    def load_model(self, path: str):
        self.actor.load_weights(path + "_actor.weights.h5")
        self.critic.load_weights(path + "_critic.weights.h5")
        soft_update(self.actor_target,  self.actor,  tau=1.0)
        soft_update(self.critic_target, self.critic, tau=1.0)
        return path

    # ── Dispatch ──────────────────────────────────────────────────────────────
    def handle(self, request: dict) -> dict:
        cmd  = request.get("cmd", "")
        args = request.get("args", {})
        try:
            if cmd == "encode_emergency":
                return {"ok": True, "result": self.encode_emergency(args["instruction"])}
            elif cmd == "actor_forward":
                return {"ok": True, "result": self.actor_forward(args["state"])}
            elif cmd == "critic_forward":
                return {"ok": True, "result": self.critic_forward(args["state"], args["action"])}
            elif cmd == "push_transition":
                n = self.push_transition(args["state"], args["action"],
                                         args["reward"], args["next_state"])
                return {"ok": True, "result": n}
            elif cmd == "train_step":
                return {"ok": True, "result": self.train_step(args.get("batch_size", 64))}
            elif cmd == "pretrain_actor_batch":
                loss = self.pretrain_actor_batch(
                    args["states"], args["actions"], args["Mqs"],
                    args["gammas"], args["thetas"])
                return {"ok": True, "result": loss}
            elif cmd == "pretrain_critic_batch":
                loss = self.pretrain_critic_batch(
                    args["states"], args["actions"], args["q_targets"])
                return {"ok": True, "result": loss}
            elif cmd == "save_model":
                return {"ok": True, "result": self.save_model(args["path"])}
            elif cmd == "load_model":
                return {"ok": True, "result": self.load_model(args["path"])}
            elif cmd == "ping":
                return {"ok": True, "result": "pong"}
            else:
                return {"ok": False, "error": f"Unknown command: {cmd}"}
        except Exception:
            return {"ok": False, "error": traceback.format_exc()}


def handle_client(conn, server: SIGMAServer):
    buf = b""
    with conn:
        while True:
            chunk = conn.recv(65536)
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                try:
                    req = json.loads(line.decode("utf-8"))
                    resp = server.handle(req)
                except json.JSONDecodeError as e:
                    resp = {"ok": False, "error": f"JSON decode error: {e}"}
                conn.sendall((json.dumps(resp) + "\n").encode("utf-8"))


def run_server(port=DEFAULT_PORT, use_llm=False,
               llm_model="meta-llama/Llama-2-7b-chat-hf", device="cpu"):
    server = SIGMAServer(use_llm=use_llm, llm_model=llm_model, device=device)
    sock   = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", port))
    sock.listen(5)
    print(f"[Server] Listening on 127.0.0.1:{port}")
    while True:
        conn, addr = sock.accept()
        print(f"[Server] Connection from {addr}")
        t = threading.Thread(target=handle_client, args=(conn, server), daemon=True)
        t.start()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SIGMA TF Bridge Server")
    parser.add_argument("--port",      type=int,  default=DEFAULT_PORT)
    parser.add_argument("--use-llm",   action="store_true")
    parser.add_argument("--llm-model", type=str,
                        default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--device",    type=str,  default="cpu")
    args = parser.parse_args()
    run_server(args.port, args.use_llm, args.llm_model, args.device)
