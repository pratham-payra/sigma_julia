"""
Constants.jl - Shared constants for SIGMA SUMO simulation.
Identical mathematical definitions as Sigma_manual; SUMO-specific parameters added.
"""
module Constants

using LinearAlgebra

# Direction indices (E=1, N=2, W=3, S=4)
const E, N, W, S = 1, 2, 3, 4
const DIR_NAMES = ["E", "N", "W", "S"]

const N_DIR     = 4
const N_PHASES  = 8
const N_MOVE    = 16
const STATE_DIM = 52   # 16(Mq) + 16(prev_phase) + 4(Gamma) + 16(theta)

# ─── ATR: (8×16) one-hot phase → movement mask ───────────────────────────────
# Row order: [AE1, AE2-W, AE3-W, AW1, AN1, AN2-S, AN3-S, AS1]
# Col order: EE,EN,EW,ES, NE,NN,NW,NS, WE,WN,WW,WS, SE,SN,SW,SS
const ATR = Float64[
   1  1  1  1   0  0  0  0   0  0  0  0   0  0  0  0;  # AE1
   0  1  1  0   0  0  0  0   1  0  0  1   0  0  0  0;  # AE2-W
   1  0  0  1   0  0  0  0   0  1  1  0   0  0  0  0;  # AE3-W
   0  0  0  0   0  0  0  0   1  1  1  1   0  0  0  0;  # AW1
   0  0  0  0   1  1  1  1   0  0  0  0   0  0  0  0;  # AN1
   0  0  0  0   0  0  1  1   0  0  0  0   1  1  0  0;  # AN2-S
   0  0  0  0   1  1  0  0   0  0  0  0   0  0  1  1;  # AN3-S
   0  0  0  0   0  0  0  0   0  0  0  0   1  1  1  1;  # AS1
]

# ─── Transition prior Qp (8×8) ───────────────────────────────────────────────
const QP_RAW = Float64[
   0  1  0  0  0  0  0  0;
   1  0  0  0  0  0  0  0;
   0  0  0  0  0  0  0  1;
   0  0  0  0  1  0  0  0;
   0  0  0  1  0  0  0  0;
   0  0  0  0  0  0  1  0;
   0  0  0  0  0  1  0  0;
   0  0  1  0  0  0  0  0;
]
make_Q(ε=0.1) = (1.0-ε) .* QP_RAW .+ (ε/8.0)
const Q_TRANSITION = make_Q()

# ─── C4 rotation operators ────────────────────────────────────────────────────
const RHO_PRIME = Float64[0 1 0 0; 0 0 1 0; 0 0 0 1; 1 0 0 0]

function _make_rho16()
    R = zeros(Float64,16,16)
    for i in 1:16; R[i, mod1(i+4,16)] = 1.0; end
    return R
end
const RHO16 = _make_rho16()

const _P  = Float64[0 0 0 1; 1 0 0 0; 0 1 0 0; 0 0 1 0]
const _S4 = Float64[0 1 0 0; 0 0 0 1; 1 0 0 0; 0 0 1 0]
function _make_rho_a()
    R = zeros(Float64,8,8)
    R[1:4,1:4] .= _P; R[5:8,5:8] .= _S4; return R
end
const RHO_A = _make_rho_a()

# ─── Hyperparameters ──────────────────────────────────────────────────────────
const DT              = 30.0     # seconds per decision step
const Q_MAX           = 80       # max queue per approach (SUMO lanes typically larger)
const PE_EMERGENCY    = 1/20.0

# SUMO-specific
const SUMO_STEP_LENGTH = 1.0     # SUMO simulation step (seconds)
const SUMO_DELTA_T     = 30      # SUMO steps per RL decision step
const SUMO_PORT        = 8813    # TraCI port
const SUMO_HOST        = "localhost"
const YELLOW_DURATION  = 3       # seconds of yellow phase between changes

# Actor
const ACTOR_HIDDEN  = (128, 64, 64)
const ACTOR_LR      = 1e-3
const ACTOR_EPOCHS  = 200
const ACTOR_BATCH   = 256

# Critic
const CRITIC_HIDDEN = (128, 64)
const CRITIC_LR     = 1e-3
const KNN_K         = 5
const GAMMA         = 0.95
const FQI_MAX_ITER  = 50
const FQI_EPS       = 1e-4

# Reward weights
const ALPHA_M = 0.5; const ALPHA_S = 0.45; const ALPHA_Q = 0.7
const ALPHA_W = 1.3; const ALPHA_E = 2.0
const ALPHA_VEC = [ALPHA_M, ALPHA_S, ALPHA_Q, ALPHA_W, ALPHA_E]

const LAMBDA_M = 0.5; const LAMBDA_S = 0.5; const LAMBDA_Q = 0.7
const LAMBDA_W = 1.3; const LAMBDA_E = 2.0
const LAMBDA_VEC = [LAMBDA_M, LAMBDA_S, LAMBDA_Q, LAMBDA_W, LAMBDA_E]

# Online
const ONLINE_LR_Q    = 1e-4
const ONLINE_NBATCH  = 128
const ONLINE_REPLAY  = 1024
const TARGET_UPDATE  = 100
const TAU_SOFT       = 0.001

# Harmonic traffic params (for synthetic pre-training before SUMO)
const NU0   = 4.0; const MU0 = 4.0; const R_HARM = 12
const KR    = [4.5,3.6,2.7,2.2,1.8,1.4,1.3,1.1,0.9,0.7,0.5,0.3]
const WR_BASE = [2π*r/2880.0 for r in 1:12]
const PHI_R = [0,π/6,π/4,π/3,π/2,2π/3,3π/4,5π/6,π,7π/6,5π/4,4π/3]
const KAPPA_R = [3.5,2.8,2.1,1.7,1.4,1.1,1.0,0.8,0.7,0.5,0.4,0.2]
const PSI_R   = [0,π/6,π/4,π/3,π/2,2π/3,3π/4,5π/6,π,7π/6,5π/4,4π/3]

end # module
