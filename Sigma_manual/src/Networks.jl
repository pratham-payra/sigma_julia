"""
Networks.jl - Actor and Critic neural networks implemented from scratch in Julia.

Actor:  4-layer FF with ReLU + softmax output (STATE_DIM → 128 → 64 → 64 → N_PHASES)
Critic: 3-layer FF with ReLU output        (STATE_DIM+N_PHASES → 128 → 64 → 1)

All forward passes and gradient computation done manually (no AD framework needed
for this scale — paper uses standard gradient descent).
"""
module Networks

using LinearAlgebra, Random
using ..Constants

# ─────────────────────────────────────────────────────────────────────────────
# Activation functions
# ─────────────────────────────────────────────────────────────────────────────
relu(x)        = max.(x, 0.0)
relu_grad(x)   = Float64.(x .> 0)

function softmax(x::Vector{Float64})
    shifted = x .- maximum(x)
    ex = exp.(shifted)
    return ex ./ sum(ex)
end

# ─────────────────────────────────────────────────────────────────────────────
# Dense layer struct
# ─────────────────────────────────────────────────────────────────────────────
mutable struct Dense
    W ::Matrix{Float64}
    b ::Vector{Float64}
    dW::Matrix{Float64}
    db::Vector{Float64}
end

function Dense(in_dim::Int, out_dim::Int; scale=0.01)
    W  = randn(out_dim, in_dim) .* scale
    b  = zeros(out_dim)
    return Dense(W, b, zeros(size(W)), zeros(size(b)))
end

function forward(layer::Dense, x::Vector{Float64})
    return layer.W * x .+ layer.b
end

# ─────────────────────────────────────────────────────────────────────────────
# Actor network: STATE_DIM → h1 → h2 → h3 → N_PHASES (softmax)
# ─────────────────────────────────────────────────────────────────────────────
mutable struct ActorNetwork
    l1::Dense
    l2::Dense
    l3::Dense
    l4::Dense
    lr::Float64
end

function ActorNetwork(; lr=ACTOR_LR)
    return ActorNetwork(
        Dense(STATE_DIM, ACTOR_HIDDEN[1]),
        Dense(ACTOR_HIDDEN[1], ACTOR_HIDDEN[2]),
        Dense(ACTOR_HIDDEN[2], ACTOR_HIDDEN[3]),
        Dense(ACTOR_HIDDEN[3], N_PHASES),
        lr
    )
end

function copy_actor(a::ActorNetwork)
    c = ActorNetwork(lr=a.lr)
    for (dst, src) in [(c.l1,a.l1),(c.l2,a.l2),(c.l3,a.l3),(c.l4,a.l4)]
        dst.W .= src.W; dst.b .= src.b
    end
    return c
end

"""Forward pass; returns (logits, softmax_dist, cache)"""
function actor_forward(net::ActorNetwork, s::Vector{Float64})
    z1 = forward(net.l1, s)
    h1 = relu(z1)
    z2 = forward(net.l2, h1)
    h2 = relu(z2)
    z3 = forward(net.l3, h2)
    h3 = relu(z3)
    z4 = forward(net.l4, h3)
    pi = softmax(z4)
    return pi, (s=s, z1=z1, h1=h1, z2=z2, h2=h2, z3=z3, h3=h3, z4=z4)
end

"""Predict without storing cache (inference only)."""
function actor_predict(net::ActorNetwork, s::Vector{Float64})
    pi, _ = actor_forward(net, s)
    return pi
end

"""
Backward pass for actor pretraining.
Loss = cross_entropy(a, π) + λ⊤·L_R  where L_R are utility losses.
We compute d(Loss)/d(z4) and backprop.
"""
function actor_backward!(net::ActorNetwork,
                          cache::NamedTuple,
                          pi::Vector{Float64},
                          a_oh::Vector{Float64},
                          grad_from_utilities::Vector{Float64})
    # Cross-entropy grad w.r.t. logits z4: d(CE)/dz4 = π - a
    dz4 = pi .- a_oh .+ grad_from_utilities  # combined gradient
    
    # Layer 4
    net.l4.dW .= dz4 * cache.h3'
    net.l4.db .= dz4
    
    dh3 = net.l4.W' * dz4
    dz3 = dh3 .* relu_grad(cache.z3)
    
    net.l3.dW .= dz3 * cache.h2'
    net.l3.db .= dz3
    
    dh2 = net.l3.W' * dz3
    dz2 = dh2 .* relu_grad(cache.z2)
    
    net.l2.dW .= dz2 * cache.h1'
    net.l2.db .= dz2
    
    dh1 = net.l2.W' * dz2
    dz1 = dh1 .* relu_grad(cache.z1)
    
    net.l1.dW .= dz1 * cache.s'
    net.l1.db .= dz1
    
    # Gradient descent update
    for l in [net.l1, net.l2, net.l3, net.l4]
        l.W .-= net.lr .* l.dW
        l.b .-= net.lr .* l.db
    end
end

"""Policy gradient update for online actor."""
function actor_pg_update!(net::ActorNetwork,
                           s::Vector{Float64},
                           a_oh::Vector{Float64},
                           delta::Float64)
    pi, cache = actor_forward(net, s)
    # ∇ψ log π(a|s) = a/π(a) projected through softmax jacobian
    # Simplified: grad w.r.t z4 = delta · (a_oh - pi)
    grad_logits = delta .* (a_oh .- pi)
    actor_backward!(net, cache, pi, a_oh, -grad_logits)  # negate since we maximize
end

# ─────────────────────────────────────────────────────────────────────────────
# Critic network: (STATE_DIM + N_PHASES) → h1 → h2 → 1
# ─────────────────────────────────────────────────────────────────────────────
mutable struct CriticNetwork
    l1::Dense
    l2::Dense
    l3::Dense
    lr::Float64
end

function CriticNetwork(; lr=CRITIC_LR)
    in_dim = STATE_DIM + N_PHASES
    return CriticNetwork(
        Dense(in_dim, CRITIC_HIDDEN[1]),
        Dense(CRITIC_HIDDEN[1], CRITIC_HIDDEN[2]),
        Dense(CRITIC_HIDDEN[2], 1),
        lr
    )
end

function copy_critic(c::CriticNetwork)
    cp = CriticNetwork(lr=c.lr)
    for (dst, src) in [(cp.l1,c.l1),(cp.l2,c.l2),(cp.l3,c.l3)]
        dst.W .= src.W; dst.b .= src.b
    end
    return cp
end

"""Forward pass; returns (Q_value, cache)"""
function critic_forward(net::CriticNetwork, s::Vector{Float64}, a::Vector{Float64})
    x  = vcat(s, a)
    z1 = forward(net.l1, x)
    h1 = relu(z1)
    z2 = forward(net.l2, h1)
    h2 = relu(z2)
    z3 = forward(net.l3, h2)
    Qval = z3[1]
    return Qval, (x=x, z1=z1, h1=h1, z2=z2, h2=h2)
end

function critic_predict(net::CriticNetwork, s::Vector{Float64}, a::Vector{Float64})
    Q, _ = critic_forward(net, s, a)
    return Q
end

"""
Bellman TD update: minimize (Q(s,a) - y)² where y = r + γ·max_a' Q_target(s',a')
"""
function critic_td_update!(net::CriticNetwork,
                            s::Vector{Float64},
                            a::Vector{Float64},
                            target::Float64)
    Qval, cache = critic_forward(net, s, a)
    err = Qval - target
    
    dz3 = [err]
    net.l3.dW .= dz3 * cache.h2'
    net.l3.db .= dz3
    
    dh2 = net.l3.W' * dz3
    dz2 = dh2 .* relu_grad(cache.z2)
    
    net.l2.dW .= dz2 * cache.h1'
    net.l2.db .= dz2
    
    dh1 = net.l2.W' * dz2
    dz1 = dh1 .* relu_grad(cache.z1)
    
    net.l1.dW .= dz1 * cache.x'
    net.l1.db .= dz1
    
    for l in [net.l1, net.l2, net.l3]
        l.W .-= net.lr .* l.dW
        l.b .-= net.lr .* l.db
    end
    return err
end

"""Best action from critic (argmax Q(s,a) over all phases)"""
function critic_best_action(net::CriticNetwork, s::Vector{Float64})
    best_a = 1
    best_Q = -Inf
    for a in 1:N_PHASES
        a_oh = zeros(N_PHASES); a_oh[a] = 1.0
        Qval = critic_predict(net, s, a_oh)
        if Qval > best_Q
            best_Q = Qval
            best_a = a
        end
    end
    return best_a, best_Q
end

"""Soft target network update: θ_target ← τ·θ + (1-τ)·θ_target"""
function soft_update!(target::ActorNetwork, source::ActorNetwork, τ::Float64=TAU_SOFT)
    for (lt, ls) in [(target.l1,source.l1),(target.l2,source.l2),
                     (target.l3,source.l3),(target.l4,source.l4)]
        lt.W .= τ .* ls.W .+ (1-τ) .* lt.W
        lt.b .= τ .* ls.b .+ (1-τ) .* lt.b
    end
end

function soft_update!(target::CriticNetwork, source::CriticNetwork, τ::Float64=TAU_SOFT)
    for (lt, ls) in [(target.l1,source.l1),(target.l2,source.l2),(target.l3,source.l3)]
        lt.W .= τ .* ls.W .+ (1-τ) .* lt.W
        lt.b .= τ .* ls.b .+ (1-τ) .* lt.b
    end
end

end # module
