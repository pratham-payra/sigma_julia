# SIGMA TF Bridge

Python/TensorFlow backend for SIGMA. Provides:

1. **Actor + Critic as `tf.keras.Model`** — proper DL training with Adam optimiser and `GradientTape`
2. **LLaMA-2 emergency encoder** — HuggingFace `transformers` runs the real LLM to produce the 16-dim θ vector from natural language; falls back to rule-based encoder if model unavailable
3. **Socket server** — Julia calls Python over a local TCP socket (JSON protocol)
4. **Standalone training script** — full pipeline without Julia

---

## Setup

```bash
pip install -r requirements.txt
```

For GPU (optional):
```bash
pip install tensorflow[and-cuda]
```

For LLaMA-2, request access at https://huggingface.co/meta-llama/Llama-2-7b-chat-hf and login:
```bash
huggingface-cli login
```

---

## Option A — Standalone Python Training

Runs the full SIGMA pipeline (data gen → pretrain → online RL) entirely in Python/TF:

```bash
# Full experiment
python train.py

# Quick test
python train.py --quick

# With real LLaMA-2 emergency encoder
python train.py --use-llm

# Save model
python train.py --save sigma_tf_model

# Evaluate saved model
python train.py --eval sigma_tf_model
```

---

## Option B — Julia + Python Bridge

Julia handles all SUMO/environment logic; Python TF handles Actor, Critic, and LLM.

### Step 1: Start the Python server

```bash
# Rule-based encoder (no GPU needed)
python server.py

# With real LLaMA-2
python server.py --use-llm --device cuda

# Custom port
python server.py --port 9876
```

### Step 2: Use from Julia

```julia
include("src/LLMBridge.jl")
using .LLMBridge

bridge = TFBridge()          # connects to 127.0.0.1:9876

# Encode emergency via LLM
θ = encode_emergency(bridge, "Ambulance approaching from North toward East")

# Get action distribution from TF actor
π = actor_forward(bridge, state_vec)

# Push experience
push_transition!(bridge, s, a, r, s_next)

# Train step
losses = train_step!(bridge; batch_size=64)

# Save weights
save_model(bridge, "sigma_model")
```

---

## File Overview

| File | Description |
|------|-------------|
| `sigma_models.py` | Actor + Critic Keras models, ATR matrix, C4 augmentation, utility functions, replay buffer |
| `llm_encoder.py` | LLaMA-2 (HuggingFace) emergency encoder with rule-based fallback |
| `server.py` | JSON-over-TCP socket server — bridges Julia ↔ Python |
| `train.py` | Standalone full training pipeline (no Julia needed) |
| `requirements.txt` | Python dependencies |

The Julia client is at `Sigma_manual/src/LLMBridge.jl`.

---

## Architecture

```
Julia (environment, SUMO, state construction)
        │  JSON over TCP (port 9876)
        ▼
Python server.py
        ├── sigma_models.py  →  Actor (Keras), Critic (Keras), GradientTape updates
        └── llm_encoder.py   →  LLaMA-2-7B → 16-dim θ
```
