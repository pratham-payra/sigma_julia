# SIGMA — Signal Intelligence via Guided Multi-objective Actor-Critic

**Julia implementation of the SIGMA framework for adaptive traffic signal control.**

> *SIGMA:Signal Intelligence via Guided Multi-objective Actor-Critic*
> Indian Statistical Institute, Kolkata & Bangalore — June 2026

---

## Overview

SIGMA is a hierarchical actor-critic architecture for traffic signal control that jointly optimizes five competing objectives:

- **Emergency preemption** — LLM-encoded priority vectors steer green time toward emergency routes
- **Queue pressure reduction** — net pressure mask drives max-pressure phase selection
- **Waiting time fairness** — penalizes approaches with disproportionately long delays
- **Action smoothness** — limits abrupt phase switching
- **Markovian consistency** — enforces cyclic signal patterns

Key technical contributions:
- **Pure Julia** implementation with no deep learning framework dependencies
- **C4 rotational augmentation** for orientation-invariant policies (one model, any rotation)
- **Offline pretraining → online refinement** pipeline for cold-start stability
- **Native TraCI client** — controls SUMO over TCP with no Python bridge

---

## Repository Structure

```
sigma_julia/
├── Sigma_manual/          # Synthetic simulation (no SUMO required)
│   ├── main.jl
│   ├── Project.toml
│   ├── README.md
│   └── src/
│       ├── SigmaManual.jl       # Top-level module
│       ├── Constants.jl         # ATR matrix, C4 operators, hyperparameters
│       ├── Environment.jl       # Poisson traffic model + 5 utility functions
│       ├── Augmentation.jl      # C4 rotational augmentation (ρ′, ρ, ρₐ)
│       ├── Networks.jl          # Actor + Critic (pure Julia, from scratch)
│       ├── DataGeneration.jl    # Offline dataset generation
│       ├── Pretraining.jl       # Actor CE+utility loss, Critic FQI
│       ├── OnlineExecution.jl   # Online actor-critic loop + replay buffer
│       ├── Baselines.jl         # Fixed-Time, Actuated, DQN controllers
│       └── Evaluation.jl        # KPI aggregation, benchmark table, ablation
│
└── Sigma_sumo/            # SUMO simulation via TraCI
    ├── main.jl
    ├── Project.toml
    ├── README.md
    ├── src/
    │   ├── SigmaSumo.jl         # Top-level module
    │   ├── Constants.jl         # ATR matrix, C4 operators, SUMO constants
    │   ├── TraCI.jl             # Julia TraCI TCP client (no Python)
    │   ├── SumoInterface.jl     # SUMO process management + state extraction
    │   ├── Networks.jl          # Actor + Critic networks
    │   ├── UtilityFunctions.jl  # 5 utility objectives + reward
    │   ├── Pretraining.jl       # Offline synthetic pretraining
    │   ├── SumoAgent.jl         # Online agent + replay buffer + baselines
    │   └── Evaluation.jl        # KPI metrics + comparison table
    └── sumo_nets/
        ├── kolkata.sumocfg      # SUMO configuration
        ├── kolkata.net.xml      # 4-way intersection (Kolkata)
        ├── kolkata.rou.xml      # Poisson traffic demand + emergency vehicles
        └── kolkata.add.xml      # Lane detectors + output definitions
```

---

## Quick Start

### Sigma_manual (no external dependencies)

```bash
cd Sigma_manual

# Full experiment — train SIGMA + all baselines, print comparison table
julia main.jl

# Quick smoke test (reduced episode counts)
julia main.jl --quick

# Ablation study (Tables 4/5 from paper)
julia main.jl --ablation

# Train only, save model
julia main.jl --train-only
```

### Sigma_sumo (mock mode — no SUMO required)

```bash
cd Sigma_sumo

# Mock experiment — full pipeline using internal Poisson model
julia main.jl

# Quick mock test
julia main.jl --quick
```

### Sigma_sumo (with real SUMO)

1. Install SUMO: https://sumo.dlr.de/docs/Installing/index.html
2. Ensure `sumo` (or `sumo-gui`) is on `PATH`

```bash
cd Sigma_sumo

julia main.jl --sumo                     # headless
julia main.jl --sumo-gui                 # with GUI
julia main.jl --sumo --quick             # quick test
julia main.jl --eval sigma_model.jls    # evaluate saved model
```

---

## Architecture

### State Vector (52-dim)

| Component     | Dim | Description                                    |
|---------------|-----|------------------------------------------------|
| p_{t-1} · ATR | 16  | Previous phase projected to movement space     |
| M^q           | 16  | Net pressure mask (incoming − outgoing queues) |
| Γ^t           |  4  | Max waiting time per approach (s)              |
| θ^t           | 16  | Emergency priority vector (LLM-encoded)        |

### Action Space (8 admissible phases)

| # | Phase  | Group | Movements                              |
|---|--------|-------|----------------------------------------|
| 1 | AE1    | G1    | All from East: E→N, E→W, E→S          |
| 2 | AE2-W  | G2    | E left+straight + W straight+left     |
| 3 | AE3-W  | G3    | E right + W right                     |
| 4 | AW1    | G1    | All from West: W→S, W→E, W→N          |
| 5 | AN1    | G1    | All from North: N→W, N→S, N→E         |
| 6 | AN2-S  | G2    | N left+straight + S straight+left     |
| 7 | AN3-S  | G3    | N right + S right                     |
| 8 | AS1    | G1    | All from South: S→E, S→N, S→W         |

### Training Pipeline

```
Synthetic Dataset D′
        │
        ▼
C4 Augmentation → D_F (4× size)
        │
        ├──▶ Actor Pretraining
        │    Loss = L_CE + λ·[L_M, L_S, L_Q, L_W, L_E]
        │
        ├──▶ Critic Pretraining
        │    Fitted Q-Iteration + k-NN next-state (k=5)
        │
        ▼
Online Actor-Critic (SUMO or synthetic)
    TD updates · soft target networks · replay buffer
        │
        ▼
Evaluation vs Fixed-Time · Actuated · DQN
```

### Utility Functions (Table 1)

| Symbol | Objective              | Formula                                      |
|--------|------------------------|----------------------------------------------|
| U_M    | Markovian consistency  | −‖π(t+1) − Q⊤π(t)‖²                        |
| U_S    | Action smoothness      | −‖ATR⊤π(t+1) − ATR⊤π(t)‖²                 |
| U_Q    | Queue pressure         | (ATR⊤π(t+1))⊤ M^q / (max M^q + 1)         |
| U_W    | Waiting time fairness  | −‖dir_served − Γ/max(Γ)‖²                  |
| U_E    | Emergency alignment    | −‖ATR⊤π(t) − θ‖²                           |

---

## Evaluation Metrics (Table 2)

| Metric | Description                          | Target |
|--------|--------------------------------------|--------|
| AMWT   | Average Maximum Waiting Time (s)     | ↓      |
| AWT    | Average Waiting Time (s)             | ↓      |
| AEWT   | Average Emergency Waiting Time (s)   | ↓      |
| AQL    | Average Queue Length (vehicles)      | ↓      |
| APC    | Average Phase Change magnitude       | ↓      |
| TC     | Transition Consistency               | ↑      |
| ATP    | Average Throughput (veh/step)        | ↑      |

---

## Dependencies

Both projects require only Julia standard library + `Distributions.jl`. No GPU, no PyTorch, no Python.

```
Julia ≥ 1.6
Distributions.jl
```

Sigma_sumo additionally uses `Sockets.jl` (stdlib) for the TraCI TCP client.

---

## Citation

```bibtex
@article{payra2026sigma,
  title   = {SIGMA: Signal Intelligence via Guided Multi-objective Actor-Critic for Traffic Control},
  author  = {Payra, Pratham and B, Jagadish and Sen, Tanmay},
  year    = {2026},
  institution = {Indian Statistical Institute}
}
```
