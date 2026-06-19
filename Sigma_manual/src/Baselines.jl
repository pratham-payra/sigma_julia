"""
Baselines.jl - Fixed-Time, Actuated, and DQN baseline controllers (Appendix A.4-A.6).

All baselines use identical state representation and reward structure as SIGMA
for fair comparison.
"""
module Baselines

using Random, LinearAlgebra, Statistics, Printf
using ..Constants
using ..Networks
using ..Environment
using ..Augmentation

# ─────────────────────────────────────────────────────────────────────────────
# Fixed-Time Control (Appendix A.4)
# Cycles through all 8 phases in order, each for d_k = 30s (1 step)
# ─────────────────────────────────────────────────────────────────────────────
mutable struct FixedTimeController
    phase_durations::Vector{Int}   # steps per phase (all 1 by default)
    current_phase  ::Int
    steps_in_phase ::Int
end

function FixedTimeController(durations::Vector{Int} = ones(Int, N_PHASES))
    return FixedTimeController(durations, 1, 0)
end

function select_action(ctrl::FixedTimeController)::Int
    ctrl.steps_in_phase += 1
    if ctrl.steps_in_phase > ctrl.phase_durations[ctrl.current_phase]
        ctrl.current_phase = mod1(ctrl.current_phase + 1, N_PHASES)
        ctrl.steps_in_phase = 1
    end
    return ctrl.current_phase
end

function run_fixed_time_episode(
    env_state::IntersectionState;
    T_steps  ::Int = 120,
    rng      ::AbstractRNG = MersenneTwister(0),
    alpha    ::Vector{Float64} = ALPHA_VEC
)
    ctrl = FixedTimeController()
    metrics = init_metrics()
    prev_phase_oh = zeros(Float64, N_PHASES); prev_phase_oh[1] = 1.0

    for t in 1:T_steps
        a_idx = select_action(ctrl)
        a_oh  = zeros(Float64, N_PHASES); a_oh[a_idx] = 1.0

        old_l_in = copy(env_state.l_in)
        step_traffic!(env_state, a_idx, rng)

        record_metrics!(metrics, env_state, prev_phase_oh, a_oh, old_l_in)
        prev_phase_oh = a_oh
    end
    return metrics
end

# ─────────────────────────────────────────────────────────────────────────────
# Actuated Control (Appendix A.5)
# ─────────────────────────────────────────────────────────────────────────────
mutable struct ActuatedController
    G_min::Float64   # minimum green steps
    G_max::Float64   # maximum green steps
    G_gap::Float64   # gap threshold (queue drop fraction)
    current_phase::Int
    steps_green  ::Int
end

function ActuatedController(;G_min=1, G_max=6, G_gap=0.1)
    return ActuatedController(Float64(G_min), Float64(G_max), Float64(G_gap), 1, 0)
end

function select_action(ctrl::ActuatedController, l_in::Vector{Float64})::Int
    ctrl.steps_green += 1

    # Extend green if within G_min
    if ctrl.steps_green < ctrl.G_min
        return ctrl.current_phase
    end

    # Check gap condition: if max queue has dropped below gap threshold, switch
    max_q = maximum(l_in)
    gap_condition = max_q < ctrl.G_gap * Q_MAX

    if ctrl.steps_green >= ctrl.G_max || gap_condition
        # Select next phase: max net pressure
        best_p  = ctrl.current_phase
        best_mq = -Inf
        Mq = build_pressure_mask(l_in, zeros(4))
        for p in 1:N_PHASES
            served = sum(ATR[p, :] .* Mq)
            if served > best_mq
                best_mq = served
                best_p  = p
            end
        end
        if best_p != ctrl.current_phase
            ctrl.current_phase = best_p
            ctrl.steps_green   = 0
        end
    end
    return ctrl.current_phase
end

function run_actuated_episode(
    env_state::IntersectionState;
    T_steps  ::Int = 120,
    rng      ::AbstractRNG = MersenneTwister(0),
    alpha    ::Vector{Float64} = ALPHA_VEC
)
    ctrl = ActuatedController()
    metrics = init_metrics()
    prev_phase_oh = zeros(Float64, N_PHASES); prev_phase_oh[1] = 1.0

    for t in 1:T_steps
        a_idx = select_action(ctrl, env_state.l_in)
        a_oh  = zeros(Float64, N_PHASES); a_oh[a_idx] = 1.0

        old_l_in = copy(env_state.l_in)
        step_traffic!(env_state, a_idx, rng)

        record_metrics!(metrics, env_state, prev_phase_oh, a_oh, old_l_in)
        prev_phase_oh = a_oh
    end
    return metrics
end

# ─────────────────────────────────────────────────────────────────────────────
# Deep Q-Network Baseline (Appendix A.6)
# ─────────────────────────────────────────────────────────────────────────────
mutable struct DQNController
    Q_net   ::CriticNetwork
    Q_target::CriticNetwork
    replay  ::Vector{Tuple{Vector{Float64},Vector{Float64},Float64,Vector{Float64}}}
    epsilon ::Float64
    gamma   ::Float64
    step    ::Int
    target_freq::Int
end

function DQNController(;ε=DQN_EPS, γ=GAMMA, target_freq=100)
    net    = CriticNetwork(lr=DQN_LR)
    target = copy_critic(net)
    return DQNController(net, target, [], ε, γ, 0, target_freq)
end

function dqn_select_action(ctrl::DQNController, s::Vector{Float64}, rng::AbstractRNG)::Int
    if rand(rng) < ctrl.epsilon
        return rand(rng, 1:N_PHASES)
    else
        best_a, _ = critic_best_action(ctrl.Q_net, s)
        return best_a
    end
end

function dqn_update!(ctrl::DQNController, rng::AbstractRNG, batch_size::Int=DQN_BATCH)
    length(ctrl.replay) < batch_size && return

    idx   = randperm(rng, length(ctrl.replay))[1:batch_size]
    batch = ctrl.replay[idx]

    for (s, a, r, s_next) in batch
        _, Q_next = critic_best_action(ctrl.Q_target, s_next)
        y = r + ctrl.gamma * Q_next
        critic_td_update!(ctrl.Q_net, s, a, y)
    end

    ctrl.step += 1
    if ctrl.step % ctrl.target_freq == 0
        for (lt, ls) in [(ctrl.Q_target.l1, ctrl.Q_net.l1),
                         (ctrl.Q_target.l2, ctrl.Q_net.l2),
                         (ctrl.Q_target.l3, ctrl.Q_net.l3)]
            lt.W .= ls.W; lt.b .= ls.b
        end
    end
end

function run_dqn_episode(
    ctrl     ::DQNController,
    env_state::IntersectionState;
    T_steps  ::Int = 120,
    rng      ::AbstractRNG = MersenneTwister(0),
    alpha    ::Vector{Float64} = ALPHA_VEC
)
    metrics = init_metrics()
    prev_phase_oh = zeros(Float64, N_PHASES); prev_phase_oh[1] = 1.0

    for t in 1:T_steps
        prev_phase_16 = vec(prev_phase_oh' * ATR)
        s = vcat(prev_phase_16, env_state.Mq, env_state.Gamma, env_state.theta)

        a_idx = dqn_select_action(ctrl, s, rng)
        a_oh  = zeros(Float64, N_PHASES); a_oh[a_idx] = 1.0

        old_l_in = copy(env_state.l_in)
        step_traffic!(env_state, a_idx, rng)

        next_prev_16 = vec(a_oh' * ATR)
        s_next = vcat(next_prev_16, env_state.Mq, env_state.Gamma, env_state.theta)

        # Reward: same structure as SIGMA (fixed alpha weights)
        pi_a = copy(a_oh)
        r = compute_reward(pi_a, pi_a, env_state.Mq, env_state.Gamma, env_state.theta, alpha)

        push!(ctrl.replay, (s, a_oh, r, s_next))
        if length(ctrl.replay) > ONLINE_REPLAY
            popfirst!(ctrl.replay)
        end

        dqn_update!(ctrl, rng)

        record_metrics!(metrics, env_state, prev_phase_oh, a_oh, old_l_in)
        prev_phase_oh = a_oh
    end
    return metrics
end

# ─────────────────────────────────────────────────────────────────────────────
# Metrics helpers
# ─────────────────────────────────────────────────────────────────────────────
mutable struct EpisodeMetrics
    wait_times      ::Vector{Float64}
    emergency_waits ::Vector{Float64}
    max_wait_times  ::Vector{Float64}
    queue_lengths   ::Vector{Float64}
    phase_changes   ::Vector{Float64}
    throughputs     ::Vector{Float64}
end

function init_metrics()
    return EpisodeMetrics([], [], [], [], [], [])
end

function record_metrics!(m::EpisodeMetrics,
                          env ::IntersectionState,
                          prev_oh::Vector{Float64},
                          a_oh   ::Vector{Float64},
                          old_l_in::Vector{Float64})
    push!(m.wait_times,    mean(env.Gamma))
    push!(m.max_wait_times, maximum(env.Gamma))
    push!(m.queue_lengths, mean(env.l_in))
    push!(m.phase_changes,
          norm(vec(a_oh' * ATR) .- vec(prev_oh' * ATR))^2 / 8.0)
    if env.emergency_active
        push!(m.emergency_waits, maximum(env.Gamma))
    end
    released = sum(max.(old_l_in .- env.l_in, 0.0))
    push!(m.throughputs, released)
end

end # module
