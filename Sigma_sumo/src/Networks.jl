"""
Networks.jl - Actor and Critic neural networks (identical architecture to Sigma_manual).
Actor:  STATE_DIM → 128 → 64 → 64 → N_PHASES (softmax)
Critic: STATE_DIM+N_PHASES → 128 → 64 → 1
"""
module Networks

using LinearAlgebra, Random, Serialization
using ..Constants

relu(x)      = max.(x, 0.0)
relu_grad(x) = Float64.(x .> 0)

function softmax(x::Vector{Float64})
    shifted = x .- maximum(x)
    ex = exp.(shifted)
    return ex ./ sum(ex)
end

mutable struct Dense
    W::Matrix{Float64}; b::Vector{Float64}
    dW::Matrix{Float64}; db::Vector{Float64}
end
Dense(in_dim::Int, out_dim::Int; scale=0.01) =
    Dense(randn(out_dim,in_dim).*scale, zeros(out_dim),
          zeros(out_dim,in_dim), zeros(out_dim))
fwd(l::Dense, x) = l.W*x .+ l.b

# ─── Actor ────────────────────────────────────────────────────────────────────
mutable struct ActorNetwork
    l1::Dense; l2::Dense; l3::Dense; l4::Dense
    lr::Float64
end

ActorNetwork(;lr=ACTOR_LR) = ActorNetwork(
    Dense(STATE_DIM, ACTOR_HIDDEN[1]),
    Dense(ACTOR_HIDDEN[1], ACTOR_HIDDEN[2]),
    Dense(ACTOR_HIDDEN[2], ACTOR_HIDDEN[3]),
    Dense(ACTOR_HIDDEN[3], N_PHASES), lr)

function copy_actor(a::ActorNetwork)
    c = ActorNetwork(lr=a.lr)
    for (d,s) in [(c.l1,a.l1),(c.l2,a.l2),(c.l3,a.l3),(c.l4,a.l4)]
        d.W .= s.W; d.b .= s.b
    end; return c
end

function actor_forward(net::ActorNetwork, s::Vector{Float64})
    z1=fwd(net.l1,s); h1=relu(z1)
    z2=fwd(net.l2,h1); h2=relu(z2)
    z3=fwd(net.l3,h2); h3=relu(z3)
    z4=fwd(net.l4,h3)
    pi=softmax(z4)
    return pi,(s=s,z1=z1,h1=h1,z2=z2,h2=h2,z3=z3,h3=h3,z4=z4)
end

actor_predict(net::ActorNetwork, s::Vector{Float64}) = (actor_forward(net,s))[1]

function actor_backward!(net::ActorNetwork, cache::NamedTuple,
                          pi::Vector{Float64}, a_oh::Vector{Float64},
                          extra_grad::Vector{Float64}=zeros(N_PHASES))
    dz4 = pi .- a_oh .+ extra_grad
    net.l4.dW .= dz4 * cache.h3'; net.l4.db .= dz4
    dh3 = net.l4.W'*dz4; dz3 = dh3 .* relu_grad(cache.z3)
    net.l3.dW .= dz3 * cache.h2'; net.l3.db .= dz3
    dh2 = net.l3.W'*dz3; dz2 = dh2 .* relu_grad(cache.z2)
    net.l2.dW .= dz2 * cache.h1'; net.l2.db .= dz2
    dh1 = net.l2.W'*dz2; dz1 = dh1 .* relu_grad(cache.z1)
    net.l1.dW .= dz1 * cache.s'; net.l1.db .= dz1
    for l in [net.l1,net.l2,net.l3,net.l4]
        l.W .-= net.lr.*l.dW; l.b .-= net.lr.*l.db
    end
end

function actor_pg_update!(net::ActorNetwork, s::Vector{Float64},
                           a_oh::Vector{Float64}, delta::Float64)
    pi, cache = actor_forward(net, s)
    grad = delta .* (a_oh .- pi)
    actor_backward!(net, cache, pi, a_oh, -grad)
end

# ─── Critic ───────────────────────────────────────────────────────────────────
mutable struct CriticNetwork
    l1::Dense; l2::Dense; l3::Dense
    lr::Float64
end

CriticNetwork(;lr=CRITIC_LR) = CriticNetwork(
    Dense(STATE_DIM+N_PHASES, CRITIC_HIDDEN[1]),
    Dense(CRITIC_HIDDEN[1], CRITIC_HIDDEN[2]),
    Dense(CRITIC_HIDDEN[2], 1), lr)

function copy_critic(c::CriticNetwork)
    cp = CriticNetwork(lr=c.lr)
    for (d,s) in [(cp.l1,c.l1),(cp.l2,c.l2),(cp.l3,c.l3)]
        d.W .= s.W; d.b .= s.b
    end; return cp
end

function critic_forward(net::CriticNetwork, s::Vector{Float64}, a::Vector{Float64})
    x=vcat(s,a); z1=fwd(net.l1,x); h1=relu(z1)
    z2=fwd(net.l2,h1); h2=relu(z2); z3=fwd(net.l3,h2)
    return z3[1],(x=x,z1=z1,h1=h1,z2=z2,h2=h2)
end

critic_predict(net::CriticNetwork, s, a) = (critic_forward(net,s,a))[1]

function critic_td_update!(net::CriticNetwork, s, a, target::Float64)
    Qval, cache = critic_forward(net,s,a)
    err = Qval - target
    dz3 = [err]
    net.l3.dW .= dz3*cache.h2'; net.l3.db .= dz3
    dh2 = net.l3.W'*dz3; dz2 = dh2.*relu_grad(cache.z2)
    net.l2.dW .= dz2*cache.h1'; net.l2.db .= dz2
    dh1 = net.l2.W'*dz2; dz1 = dh1.*relu_grad(cache.z1)
    net.l1.dW .= dz1*cache.x'; net.l1.db .= dz1
    for l in [net.l1,net.l2,net.l3]
        l.W .-= net.lr.*l.dW; l.b .-= net.lr.*l.db
    end
    return err
end

function critic_best_action(net::CriticNetwork, s::Vector{Float64})
    best_a=1; best_Q=-Inf
    for a in 1:N_PHASES
        a_oh=zeros(N_PHASES); a_oh[a]=1.0
        Q=critic_predict(net,s,a_oh)
        if Q>best_Q; best_Q=Q; best_a=a; end
    end
    return best_a, best_Q
end

function soft_update!(target::ActorNetwork, source::ActorNetwork, τ=TAU_SOFT)
    for (lt,ls) in [(target.l1,source.l1),(target.l2,source.l2),
                    (target.l3,source.l3),(target.l4,source.l4)]
        lt.W .= τ.*ls.W .+ (1-τ).*lt.W; lt.b .= τ.*ls.b .+ (1-τ).*lt.b
    end
end

function soft_update!(target::CriticNetwork, source::CriticNetwork, τ=TAU_SOFT)
    for (lt,ls) in [(target.l1,source.l1),(target.l2,source.l2),(target.l3,source.l3)]
        lt.W .= τ.*ls.W .+ (1-τ).*lt.W; lt.b .= τ.*ls.b .+ (1-τ).*lt.b
    end
end

"""Save / load model weights."""
function save_model(path::String, actor::ActorNetwork, critic::CriticNetwork)
    serialize(path, (actor=actor, critic=critic))
end

function load_model(path::String)
    d = deserialize(path)
    return d.actor, d.critic
end

end # module Networks
