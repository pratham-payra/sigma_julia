"""
Constants.jl - All fixed matrices, hyperparameters, and indices for SIGMA.

Key concepts:
  - 4 directions: E=1, N=2, W=3, S=4
  - 8 admissible phases (one-hot dim=8)
  - 16-dim movement vector (4x4 flattened row-major: EE,EN,EW,ES,NE,NN,NW,NS,WE,WN,WW,WS,SE,SN,SW,SS)
  - State dim = 16 (pressure) + 16 (prev phase) + 4 (max wait) + 16 (emergency) = 52
"""
module Constants

using LinearAlgebra

# Direction indices
const E, N, W, S = 1, 2, 3, 4
const DIR_NAMES = ["E", "N", "W", "S"]

# Dimensions
const N_DIR     = 4     # number of directions
const N_PHASES  = 8     # number of admissible signal phases
const N_MOVE    = 16    # movement space dim (4x4)
const STATE_DIM = 52    # 16 + 16 + 4 + 16

# ─────────────────────────────────────────────────────────────────────────────
# A_TR: transition matrix  (8 × 16)  maps one-hot phase → 16-dim movement mask
# Row order (SO):  [AE1, AE2-W, AE3-W, AW1, AN1, AN2-S, AN3-S, AS1]
# Column order (MS): EE,EN,EW,ES, NE,NN,NW,NS, WE,WN,WW,WS, SE,SN,SW,SS
# Phase definitions (paper Fig 2 + Appendix B):
#   AE1  (G1, pivot=E): all movements FROM E → EE,EN,EW,ES = cols 1-4
#   AW1  (G1, pivot=W): all movements FROM W → WE,WN,WW,WS = cols 9-12
#   AN1  (G1, pivot=N): all movements FROM N → NE,NN,NW,NS = cols 5-8
#   AS1  (G1, pivot=S): all movements FROM S → SE,SN,SW,SS = cols 13-16
#   AN2-S (G2, NS pair): N straight+left + S straight+left
#          N straight=NS(col8), N left=NW(col7); S straight=SN(col14), S left=SE(col13)
#   AE2-W (G2, EW pair): E straight+left + W straight+left
#          E straight=EW(col3), E left=EN(col2); W straight=WE(col9), W left=WS(col12)
#   AN3-S (G3, NS right+U): N right=NE(col5), N U=NN(col6); S right=SW(col15), S U=SS(col16)
#   AE3-W (G3, EW right+U): E right=ES(col4), E U=EE(col1); W right=WN(col10), W U=WW(col11)
# ─────────────────────────────────────────────────────────────────────────────
const ATR = Float64[
# EE EN EW ES  NE NN NW NS  WE WN WW WS  SE SN SW SS
   1  1  1  1   0  0  0  0   0  0  0  0   0  0  0  0;  # AE1
   0  1  1  0   0  0  0  0   1  0  0  1   0  0  0  0;  # AE2-W  (E straight+left, W straight+left)
   1  0  0  1   0  0  0  0   0  1  1  0   0  0  0  0;  # AE3-W  (E right+U, W right+U)
   0  0  0  0   0  0  0  0   1  1  1  1   0  0  0  0;  # AW1
   0  0  0  0   1  1  1  1   0  0  0  0   0  0  0  0;  # AN1
   0  0  0  0   0  0  1  1   0  0  0  0   1  1  0  0;  # AN2-S  (N straight+left, S straight+left)
   0  0  0  0   1  1  0  0   0  0  0  0   0  0  1  1;  # AN3-S  (N right+U, S right+U)
   0  0  0  0   0  0  0  0   0  0  0  0   1  1  1  1;  # AS1
]

# ─────────────────────────────────────────────────────────────────────────────
# Transition prior matrix Q_p (8×8): expected next-phase transitions
# Encodes cyclic pattern: each phase most likely transitions to the "next" one
# Q = (1-ε)·Qp + (ε/8)·ones  used during actor training
# ─────────────────────────────────────────────────────────────────────────────
const QP_RAW = Float64[
# AE1 AE2W AE3W AW1  AN1 AN2S AN3S AS1
   0    1    0    0    0    0    0    0;   # from AE1
   1    0    0    0    0    0    0    0;   # from AE2W
   0    0    0    0    0    0    0    1;   # from AE3W
   0    0    0    0    1    0    0    0;   # from AW1
   0    0    0    1    0    0    0    0;   # from AN1
   0    0    0    0    0    0    1    0;   # from AN2S
   0    0    0    0    0    1    0    0;   # from AN3S
   0    0    1    0    0    0    0    0;   # from AS1
]

function make_Q(ε::Float64=0.1)
    return (1.0 - ε) .* QP_RAW .+ (ε / 8.0)
end
const Q_TRANSITION = make_Q()

# ─────────────────────────────────────────────────────────────────────────────
# C4 rotation operators
# ρ′ (4×4): anticlockwise 90° on 4-dim per-approach vectors [E,N,W,S]
#   maps E→N, N→W, W→S, S→E  ⟹  [Ye,Yn,Yw,Ys] → [Yn,Yw,Ys,Ye]
# ρ  (16×16): anticlockwise 90° on 16-dim movement vectors
#   permutes direction blocks cyclically
# ρa (8×8):  anticlockwise 90° on 8-dim one-hot action vectors
# ─────────────────────────────────────────────────────────────────────────────
const RHO_PRIME = Float64[
    0  1  0  0;
    0  0  1  0;
    0  0  0  1;
    1  0  0  0;
]

# 16-dim block-cyclic permutation
function _make_rho16()
    R = zeros(Float64, 16, 16)
    # blocks of 4: rows 1-4 (E), 5-8 (N), 9-12 (W), 13-16 (S)
    # anticlockwise: E→N, N→W, W→S, S→E
    # So block that was at N goes to where E was, etc.
    # Row i in new vector = Row (i+4) mod 16 in old (shift blocks by one)
    for i in 1:16
        j = mod1(i + 4, 16)
        R[i, j] = 1.0
    end
    return R
end
const RHO16 = _make_rho16()

# 8-dim action rotation
# Phase order: [AE1, AE2W, AE3W, AW1, AN1, AN2S, AN3S, AS1]
# P (4×4) permutes single-pivot G1: AE1→AN1→AW1→AS1 (cyclic)
# S (4×4) permutes complementary pairs
const _P = Float64[
    0  0  0  1;
    1  0  0  0;
    0  1  0  0;
    0  0  1  0;
]
const _S4 = Float64[
    0  1  0  0;
    0  0  0  1;
    1  0  0  0;
    0  0  1  0;
]

function _make_rho_a()
    R = zeros(Float64, 8, 8)
    R[1:4, 1:4] .= _P
    R[5:8, 5:8] .= _S4
    return R
end
const RHO_A = _make_rho_a()

# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameters
# ─────────────────────────────────────────────────────────────────────────────

# Data generation
const DT       = 30.0   # seconds per step
const NU0      = 4.0    # base arrival rate (veh/s per approach)
const MU0      = 4.0    # base departure rate (veh/s per direction)
const R_HARM   = 12     # number of harmonics
const KR       = [4.5, 3.6, 2.7, 2.2, 1.8, 1.4, 1.3, 1.1, 0.9, 0.7, 0.5, 0.3]
const WR_BASE  = [2π * r / 2880.0 for r in 1:12]  # ωr = 2πr/2880
const PHI_R    = [0, π/6, π/4, π/3, π/2, 2π/3, 3π/4, 5π/6, π, 7π/6, 5π/4, 4π/3]
const KAPPA_R  = [3.5, 2.8, 2.1, 1.7, 1.4, 1.1, 1.0, 0.8, 0.7, 0.5, 0.4, 0.2]
const PSI_R    = [0, π/6, π/4, π/3, π/2, 2π/3, 3π/4, 5π/6, π, 7π/6, 5π/4, 4π/3]
const PE_EMERGENCY = 1.0 / 20.0  # emergency probability per episode
const Q_MAX    = 50     # max queue capacity per approach

# Actor pretraining
const ACTOR_HIDDEN  = (128, 64, 64)
const ACTOR_LR      = 1e-3
const ACTOR_EPOCHS  = 200
const ACTOR_BATCH   = 256
const LAMBDA_E      = 2.0
const LAMBDA_W      = 1.3
const LAMBDA_Q      = 0.7
const LAMBDA_S      = 0.5
const LAMBDA_M      = 0.5
const LAMBDA_VEC    = [LAMBDA_M, LAMBDA_S, LAMBDA_Q, LAMBDA_W, LAMBDA_E]

# Critic pretraining
const CRITIC_HIDDEN = (128, 64)
const CRITIC_LR     = 1e-3
const KNN_K         = 5
const GAMMA         = 0.95
const FQI_MAX_ITER  = 50
const FQI_EPS       = 1e-4
const ALPHA_M       = 0.5
const ALPHA_S       = 0.45
const ALPHA_Q       = 0.7
const ALPHA_W       = 1.3
const ALPHA_E       = 2.0
const ALPHA_VEC     = [ALPHA_M, ALPHA_S, ALPHA_Q, ALPHA_W, ALPHA_E]

# Online execution
const ONLINE_LR_Q    = 1e-4
const ONLINE_LR_PI   = 1e-4
const ONLINE_NBATCH  = 128
const ONLINE_REPLAY  = 512
const TARGET_UPDATE  = 100
const TAU_SOFT       = 0.001

# DQN baseline
const DQN_EPS   = 0.1
const DQN_LR    = 1e-4
const DQN_BATCH = 128

end # module
