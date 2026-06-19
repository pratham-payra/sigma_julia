# SIGMA SUMO — Signal Intelligence via Guided Multi-objective Actor-Critic

Julia implementation of SIGMA for SUMO traffic simulation.

## Structure

```
Sigma_sumo/
├── main.jl                  # Entry point
├── Project.toml
├── src/
│   ├── SigmaSumo.jl         # Top-level module
│   ├── Constants.jl         # Hyperparameters, ATR matrix, rotation operators
│   ├── TraCI.jl             # Julia TraCI TCP client (no Python needed)
│   ├── SumoInterface.jl     # SUMO environment wrapper
│   ├── Networks.jl          # Actor + Critic neural networks
│   ├── UtilityFunctions.jl  # 5 utility objectives + reward
│   ├── Pretraining.jl       # Offline synthetic pretraining
│   ├── SumoAgent.jl         # Online actor-critic + baselines
│   └── Evaluation.jl        # KPI metrics + benchmark table
└── sumo_nets/
    ├── kolkata.sumocfg      # SUMO configuration
    ├── kolkata.net.xml      # 4-way intersection network
    ├── kolkata.rou.xml      # Traffic demand (Poisson + emergency)
    └── kolkata.add.xml      # Detectors and output definitions
```

## Quick Start (No SUMO required)

```bash
julia main.jl              # mock experiment — no SUMO installation needed
julia main.jl --quick      # reduced episode count for quick testing
```

## With Real SUMO

1. Install SUMO: https://sumo.dlr.de/docs/Installing/index.html
2. Ensure `sumo` is on PATH
3. Copy `sumo_nets/` files to working directory or update `sumo_cfg` path

```bash
julia main.jl --sumo                    # headless SUMO
julia main.jl --sumo-gui                # with GUI
julia main.jl --sumo --quick            # quick test
julia main.jl --eval sigma_model.jls   # evaluate saved model
```

## Pipeline

1. **Offline Pretraining** — synthetic Poisson traffic with harmonic demand variation
   - Non-homogeneous Poisson arrivals/departures (Table 11-13 parameters)
   - Emergency vehicle injection (p = 1/20 per episode)
   - C4 rotational augmentation (4× dataset)
   - Actor: cross-entropy + 5-objective utility loss
   - Critic: fitted Q-iteration with k-NN next-state approximation

2. **Online SUMO Training** — pretrained policy refined against live SUMO
   - TraCI interface reads real queue lengths, waiting times
   - Emergency detection via vehicle type
   - Yellow-phase transitions between signal changes
   - TD-learning actor-critic updates

3. **Evaluation** — SIGMA vs Fixed-Time vs Actuated
   - AMWT, AWT, AEWT, AQL, APC, TC, ATP metrics

## State Vector (52-dim)

| Component       | Dim | Description                               |
|-----------------|-----|-------------------------------------------|
| p_{t-1} · ATR   | 16  | Previous phase movement mask              |
| M^q             | 16  | Net pressure mask (incoming - outgoing)   |
| Γ^t             |  4  | Max waiting time per approach (s)         |
| θ^t             | 16  | Emergency priority vector (LLM-encoded)   |

## Action Space (8 phases)

| Index | Phase  | Description                        |
|-------|--------|------------------------------------|
| 1     | AE1    | All East movements green           |
| 2     | AE2-W  | East + West straight & left        |
| 3     | AE3-W  | East + West right & U-turn         |
| 4     | AW1    | All West movements green           |
| 5     | AN1    | All North movements green          |
| 6     | AN2-S  | North + South straight & left      |
| 7     | AN3-S  | North + South right & U-turn       |
| 8     | AS1    | All South movements green          |
