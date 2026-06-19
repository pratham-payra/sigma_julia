# SIGMA Manual — Signal Intelligence via Guided Multi-objective Actor-Critic

Julia implementation of SIGMA for **manual synthetic simulation** (no SUMO required).

## Structure

```
Sigma_manual/
├── main.jl                  # Entry point
├── Project.toml
└── src/
    ├── SigmaManual.jl       # Top-level module
    ├── Constants.jl         # Hyperparameters, ATR matrix, C4 rotation operators
    ├── Environment.jl       # Synthetic intersection: Poisson traffic + utilities
    ├── Augmentation.jl      # C4 rotational augmentation framework
    ├── Networks.jl          # Actor + Critic neural networks (pure Julia)
    ├── DataGeneration.jl    # Offline dataset generation
    ├── Pretraining.jl       # Actor pretraining (CE+utility) + Critic FQI
    ├── OnlineExecution.jl   # Online actor-critic loop + replay buffer
    ├── Baselines.jl         # Fixed-Time, Actuated, DQN baselines
    └── Evaluation.jl        # KPI metrics, benchmark table, ablation study
```

## Quick Start

```bash
# Full experiment (train all methods, print comparison table)
julia main.jl

# Quick smoke test
julia main.jl --quick

# Ablation study (Table 4/5 from paper)
julia main.jl --ablation

# Train and save only
julia main.jl --train-only
```

## Dependencies

Requires only standard Julia + `Distributions.jl`:
```julia
using Pkg; Pkg.activate("."); Pkg.instantiate()
```

## Pipeline (matching paper Section 4)

1. **Synthetic Data Generation** (Appendix A.7/A.8)
   - Non-homogeneous Poisson arrivals: ν(t) = ν₀ + Σ kᵣ sin(ωᵣt + φᵣ), R=12 harmonics
   - Emergency vehicle injection: Z ~ Bernoulli(1/20) per episode
   - Best action selected by evaluating all 8 phases via reward function

2. **C4 Rotational Augmentation** (Appendix B)
   - Operators: ρ′ (4×4), ρ (16×16), ρₐ (8×8)
   - 4× dataset expansion: D_F = ⋃_{k=0}^{3} g_{kπ/2}(D′)

3. **Actor Pretraining** (Appendix A.1)
   - Loss: L = L_CE + λ_M·L_M + λ_S·L_S + λ_Q·L_Q + λ_W·L_W + λ_E·L_E

4. **Critic Pretraining** (Appendix A.2)
   - Fitted Q-Iteration with k-NN next-state approximation (k=5)

5. **Online Execution** (Appendix A.3)
   - Stochastic action sampling, TD learning, soft target network updates

## Evaluation Metrics (Table 2)

| Metric | Description                            |
|--------|----------------------------------------|
| AMWT   | Average Maximum Waiting Time (s) ↓     |
| AWT    | Average Waiting Time (s) ↓             |
| AEWT   | Average Emergency Waiting Time (s) ↓   |
| AQL    | Average Queue Length ↓                 |
| APC    | Average Phase Change magnitude ↓       |
| TC     | Transition Consistency ↑               |
| ATP    | Average Throughput (veh/step) ↑        |
