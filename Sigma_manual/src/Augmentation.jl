"""
Augmentation.jl - C4 rotational augmentation framework.

Implements the rotation operators defined in Appendix B:
  - ρ′  (4×4):  anticlockwise 90° on per-approach (4-dim) vectors
  - ρ   (16×16): anticlockwise 90° on movement (16-dim) vectors
  - ρa  (8×8):  anticlockwise 90° on one-hot action (8-dim) vectors

The augmented dataset DF = ⋃_{k=0}^{3} g_{kπ/2}(D')
where g_{kπ/2}(at, p_{t-1}, Lt, Γt, θt) = (ρa^k at, ρ^k p_{t-1}, ρ'^k Lt, ρ'^k Γt, ρ^k θt)
"""
module Augmentation

using LinearAlgebra
using ..Constants

# ─────────────────────────────────────────────────────────────────────────────
# Rotate a 4-dim per-approach vector by k×90° anticlockwise
# k=0: identity, k=1: E→N→W→S→E, k=2: 180°, k=3: 270°
# ─────────────────────────────────────────────────────────────────────────────
function rotate4(v::Vector{Float64}, k::Int)
    k = mod(k, 4)
    k == 0 && return copy(v)
    R = RHO_PRIME^k
    return R * v
end

# ─────────────────────────────────────────────────────────────────────────────
# Rotate a 16-dim movement vector by k×90° anticlockwise
# ─────────────────────────────────────────────────────────────────────────────
function rotate16(v::Vector{Float64}, k::Int)
    k = mod(k, 4)
    k == 0 && return copy(v)
    R = RHO16^k
    return R * v
end

# ─────────────────────────────────────────────────────────────────────────────
# Rotate an 8-dim one-hot action vector by k×90° anticlockwise
# ─────────────────────────────────────────────────────────────────────────────
function rotate_action(a::Vector{Float64}, k::Int)
    k = mod(k, 4)
    k == 0 && return copy(a)
    R = RHO_A^k
    return R * a
end

# ─────────────────────────────────────────────────────────────────────────────
# Apply g_{kπ/2} to a single transition tuple
# Returns rotated (action_oh, prev_phase_16, l_in, gamma, theta)
# ─────────────────────────────────────────────────────────────────────────────
function rotate_transition(
    action_oh      ::Vector{Float64},  # 8-dim one-hot
    prev_phase_16  ::Vector{Float64},  # 16-dim (p_{t-1}·ATR)
    l_in           ::Vector{Float64},  # 4-dim
    gamma          ::Vector{Float64},  # 4-dim
    theta          ::Vector{Float64},  # 16-dim
    k              ::Int
)
    ra  = rotate_action(action_oh, k)
    rp  = rotate16(prev_phase_16, k)
    rl  = rotate4(l_in, k)
    rg  = rotate4(gamma, k)
    rt  = rotate16(theta, k)
    return ra, rp, rl, rg, rt
end

# ─────────────────────────────────────────────────────────────────────────────
# Augment a dataset by applying all 4 rotations
# Dataset D' is a vector of NamedTuples with fields:
#   action_oh, prev_phase_16, l_in, gamma, theta, reward, state_vec
# Returns DF with 4× the entries
# ─────────────────────────────────────────────────────────────────────────────
struct Transition
    state    ::Vector{Float64}  # 52-dim full state
    action_oh::Vector{Float64}  # 8-dim one-hot
    reward   ::Float64
    rewards_vec::Vector{Float64} # 5-dim utility rewards
    # raw components for re-building rotated states
    prev_phase_16::Vector{Float64}
    Mq           ::Vector{Float64}
    l_in         ::Vector{Float64}
    gamma        ::Vector{Float64}
    theta        ::Vector{Float64}
end

function build_state_from_parts(
    prev_phase_16::Vector{Float64},
    Mq           ::Vector{Float64},
    gamma        ::Vector{Float64},
    theta        ::Vector{Float64}
)
    return vcat(prev_phase_16, Mq, gamma, theta)
end

function augment_dataset(D::Vector{Transition})
    DF = Transition[]
    for tr in D
        for k in 0:3
            ra, rp, rl, rg, rt = rotate_transition(
                tr.action_oh, tr.prev_phase_16, tr.l_in, tr.gamma, tr.theta, k
            )
            # Rebuild Mq from rotated l_in (l_out is also rotated via rotate4)
            # We rotate the Mq directly as a 16-dim vector
            rMq = rotate16(tr.Mq, k)
            rMq = rMq .- minimum(rMq)   # re-shift
            rs  = build_state_from_parts(rp, rMq, rg, rt)
            push!(DF, Transition(rs, ra, tr.reward, tr.rewards_vec,
                                 rp, rMq, rl, rg, rt))
        end
    end
    return DF
end

end # module
