"""
Environment.jl - Synthetic traffic intersection simulation.

Implements:
  - Non-homogeneous Poisson arrival/departure model with harmonic intensities
  - Pressure mask construction (16-dim)
  - Emergency vehicle injection and encoding
  - State construction: s = [p_{t-1}, M^q, Γ^t, θ^t]  (dim=52)
  - Reward computation via all 5 utility functions
"""
module Environment

using Random, Distributions, LinearAlgebra
using ..Constants

# ─────────────────────────────────────────────────────────────────────────────
# State struct
# ─────────────────────────────────────────────────────────────────────────────
mutable struct IntersectionState
    l_in  ::Vector{Float64}   # incoming queue lengths [E,N,W,S]
    l_out ::Vector{Float64}   # outgoing queue lengths [E,N,W,S]
    Mq    ::Vector{Float64}   # net pressure mask (16-dim, non-negative)
    Gamma ::Vector{Float64}   # max waiting times per approach (4-dim)
    prev_phase::Vector{Float64}  # one-hot prev phase (8-dim) → via ATR → 16-dim
    theta ::Vector{Float64}   # emergency priority vector (16-dim)
    vehicle_wait::Matrix{Float64}  # waiting time tracker per vehicle slot per approach
    t     ::Float64           # current simulation time
    step  ::Int
    emergency_active::Bool
    emergency_origin::Int     # approach index (1-4) or 0
    emergency_dest  ::Int
end

function IntersectionState()
    return IntersectionState(
        zeros(4), zeros(4), zeros(16), zeros(4),
        zeros(16), zeros(16),
        zeros(4, 100),   # up to 100 tracked vehicles per approach
        0.0, 0, false, 0, 0
    )
end

# ─────────────────────────────────────────────────────────────────────────────
# Harmonic intensity functions
# ─────────────────────────────────────────────────────────────────────────────
function arrival_rate(t::Float64)
    ν = NU0
    for r in 1:R_HARM
        ν += KR[r] * sin(WR_BASE[r] * t + PHI_R[r])
    end
    return max(ν, 0.0)
end

function departure_rate(t::Float64)
    μ = MU0
    for r in 1:R_HARM
        μ += KAPPA_R[r] * sin(WR_BASE[r] * t + PSI_R[r])
    end
    return max(μ, 0.0)
end

# ─────────────────────────────────────────────────────────────────────────────
# Pressure mask construction (equations 50-53)
# M^q_{in}: replicate each approach count 4 times → [l_e,l_e,l_e,l_e,l_n,…] (16-dim)
# M^q_{out}: replicate each outgoing count 4 times, interleaved per-row
# ─────────────────────────────────────────────────────────────────────────────
function build_pressure_mask(l_in::Vector{Float64}, l_out::Vector{Float64})
    # M^q_{in}(i,j) = l_in[i]  for all j  (row = incoming dir)
    Mq_in = Float64[]
    for i in 1:4
        append!(Mq_in, fill(l_in[i], 4))
    end
    # M^q_{out}(i,j) = l_out[j]  for all i  (col = outgoing dir)
    Mq_out = Float64[]
    for i in 1:4
        append!(Mq_out, l_out)
    end
    Mprime = Mq_in .- Mq_out
    Mq     = Mprime .- minimum(Mprime)   # shift to non-negative
    return Mq
end

# ─────────────────────────────────────────────────────────────────────────────
# Emergency encoding (rule-based LLM surrogate)
# In the manual simulation we use a deterministic encoder instead of LLaMA-2
# to produce the same 16-dim binary priority vector.
# ─────────────────────────────────────────────────────────────────────────────
function encode_emergency(origin::Int, dest::Int)
    # origin = incoming direction (1-4), dest = outgoing direction (1-4)
    # "exact path" rule: θ_{i,j}=1 only for specified movement
    θ = zeros(Float64, 16)
    if origin > 0 && dest > 0
        idx = (origin - 1) * 4 + dest   # row-major index in 4×4
        θ[idx] = 1.0
    elseif origin > 0
        # incoming-only: set all j for direction origin
        for j in 1:4
            θ[(origin-1)*4 + j] = 1.0
        end
    end
    return θ
end

# ─────────────────────────────────────────────────────────────────────────────
# Sample one step of traffic arrivals/departures
# ─────────────────────────────────────────────────────────────────────────────
function step_traffic!(state::IntersectionState, phase_idx::Int, rng::AbstractRNG)
    t = state.t
    ν = arrival_rate(t)
    μ = departure_rate(t)

    for i in 1:4
        arr = rand(rng, Poisson(ν * DT))
        state.l_in[i]  = min(state.l_in[i] + arr, Q_MAX)
    end

    # Departures allowed only on active phase movements
    phase_vec = ATR[phase_idx, :]   # 16-dim movement vector
    for j in 1:4
        # can depart from direction j if any movement involving incoming that serves j
        served = any(phase_vec[(i-1)*4 + j] > 0 for i in 1:4)
        if served
            dep = rand(rng, Poisson(μ * DT))
            state.l_out[j] = min(state.l_out[j] + dep, Q_MAX)
            # release vehicles from incoming that have green for this outgoing
            released = min(dep, sum(state.l_in[i] * phase_vec[(i-1)*4+j] for i in 1:4))
            for i in 1:4
                if phase_vec[(i-1)*4+j] > 0 && state.l_in[i] > 0
                    rel_i = min(released, state.l_in[i])
                    state.l_in[i] -= rel_i
                    released -= rel_i
                    if released <= 0; break; end
                end
            end
        end
    end

    state.Mq = build_pressure_mask(state.l_in, state.l_out)

    # Update max waiting times (proxy: proportional to queue length)
    for i in 1:4
        state.Gamma[i] = state.l_in[i] * DT
    end

    state.t += DT
    state.step += 1
end

# ─────────────────────────────────────────────────────────────────────────────
# Build state vector: s = [p_{t-1}·ATR (16), Mq (16), Γ (4), θ (16)] = 52-dim
# ─────────────────────────────────────────────────────────────────────────────
function build_state(state::IntersectionState, prev_phase_onehot::Vector{Float64})
    p_hist = prev_phase_onehot' * ATR   # 1×16 → flatten to 16-dim
    return vcat(vec(p_hist), state.Mq, state.Gamma, state.theta)
end

# ─────────────────────────────────────────────────────────────────────────────
# Utility functions (Table 1 in paper)
# ─────────────────────────────────────────────────────────────────────────────

"""
U_M: Markovian consistency — penalty for deviating from expected transition Q
π(t): current action distribution (8-dim softmax)
π(t+1): next action distribution
Q: transition prior matrix (8×8)
"""
function U_M(pi_t::Vector{Float64}, pi_t1::Vector{Float64}, Q::Matrix{Float64})
    expected_next = Q' * pi_t   # 8-dim expected distribution
    return -norm(pi_t1 .- expected_next)^2
end

"""
U_S: Action smoothness — penalize distributional shift between consecutive actions
Uses ATR to project into movement space for comparison
"""
function U_S(pi_t::Vector{Float64}, pi_t1::Vector{Float64})
    move_t  = ATR' * pi_t    # 16-dim
    move_t1 = ATR' * pi_t1
    return -norm(move_t1 .- move_t)^2
end

"""
U_Q: Queue pressure reduction — differentiable max-pressure
Favors actions that serve high net-pressure movements
"""
function U_Q(pi_t1::Vector{Float64}, Mq::Vector{Float64})
    # project next action onto pressure mask via ATR
    served_pressure = (ATR' * pi_t1)' * Mq   # dot product of served movements with pressure
    return served_pressure / (max(maximum(Mq), 1e-6) + 1.0)
end

"""
U_W: Waiting time fairness — penalize max waiting time deviation
"""
function U_W(pi_t1::Vector{Float64}, Gamma::Vector{Float64})
    # 4-dim waiting times: project 8-dim action to 4-dim direction via grouping
    dir_served = zeros(4)
    for i in 1:4
        for j in 1:4
            dir_served[i] += (ATR' * pi_t1)[(i-1)*4+j]
        end
    end
    # penalize unserved directions with high waiting time
    return -norm(dir_served .- Gamma ./ (max(maximum(Gamma), 1e-6)))^2
end

"""
U_E: Emergency alignment — align action distribution with emergency priority vector θ
"""
function U_E(pi_t::Vector{Float64}, theta::Vector{Float64})
    if all(theta .== 0)
        return 0.0
    end
    move_pi = ATR' * pi_t
    return -norm(move_pi .- theta)^2
end

"""
Compute all 5 utility values for a given (state, action distribution, next dist)
Returns NamedTuple with M, S, Q, W, E utilities
"""
function compute_utilities(
    pi_t   ::Vector{Float64},
    pi_t1  ::Vector{Float64},
    Mq     ::Vector{Float64},
    Gamma  ::Vector{Float64},
    theta  ::Vector{Float64},
    Q      ::Matrix{Float64} = Q_TRANSITION
)
    uM = U_M(pi_t, pi_t1, Q)
    uS = U_S(pi_t, pi_t1)
    uQ = U_Q(pi_t1, Mq)
    uW = U_W(pi_t1, Gamma)
    uE = U_E(pi_t, theta)
    return (M=uM, S=uS, Q=uQ, W=uW, E=uE)
end

"""
Reward for online execution: r = α⊤ · [uM, uS, uQ, uW, uE]
"""
function compute_reward(
    pi_t   ::Vector{Float64},
    pi_t1  ::Vector{Float64},
    Mq     ::Vector{Float64},
    Gamma  ::Vector{Float64},
    theta  ::Vector{Float64},
    alpha  ::Vector{Float64} = ALPHA_VEC
)
    u = compute_utilities(pi_t, pi_t1, Mq, Gamma, theta)
    return alpha[1]*u.M + alpha[2]*u.S + alpha[3]*u.Q + alpha[4]*u.W + alpha[5]*u.E
end

"""
Evaluate all 8 actions and return reward vector (for dataset construction)
"""
function evaluate_all_actions(
    state_vec::Vector{Float64},
    Mq     ::Vector{Float64},
    Gamma  ::Vector{Float64},
    theta  ::Vector{Float64},
    alpha  ::Vector{Float64}
)
    rewards = zeros(N_PHASES)
    for a in 1:N_PHASES
        pi_a = zeros(N_PHASES); pi_a[a] = 1.0
        # approximate next as same distribution (greedy)
        rewards[a] = compute_reward(pi_a, pi_a, Mq, Gamma, theta, alpha)
    end
    return rewards
end

end # module
