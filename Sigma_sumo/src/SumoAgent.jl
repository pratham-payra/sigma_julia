"""
SumoAgent.jl - SIGMA online actor-critic agent running against live SUMO.

The agent:
  1. Reads the SIGMA state from SUMO via TraCI
  2. Selects action via trained actor (stochastic during training, greedy for eval)
  3. Executes the phase in SUMO (with yellow transition)
  4. Computes reward from live queue/waiting data
  5. Updates actor and critic via TD learning
  6. Collects evaluation metrics
"""
module SumoAgent

using Random, LinearAlgebra, Statistics, Printf
using ..Constants
using ..Networks
using ..UtilityFunctions
using ..SumoInterface

# ─── Replay buffer ────────────────────────────────────────────────────────────
mutable struct ReplayBuffer
    states     ::Vector{Vector{Float64}}
    actions    ::Vector{Vector{Float64}}
    rewards    ::Vector{Float64}
    next_states::Vector{Vector{Float64}}
    cap::Int; size::Int; ptr::Int
end

ReplayBuffer(cap=ONLINE_REPLAY) = ReplayBuffer(
    [zeros(STATE_DIM) for _ in 1:cap],
    [zeros(N_PHASES)  for _ in 1:cap],
    zeros(cap),
    [zeros(STATE_DIM) for _ in 1:cap],
    cap, 0, 1)

function push_replay!(buf::ReplayBuffer, s, a, r, sn)
    buf.states[buf.ptr]=s; buf.actions[buf.ptr]=a
    buf.rewards[buf.ptr]=r; buf.next_states[buf.ptr]=sn
    buf.ptr=mod1(buf.ptr+1,buf.cap); buf.size=min(buf.size+1,buf.cap)
end

function sample(buf::ReplayBuffer, n::Int, rng)
    idx = randperm(rng, buf.size)[1:min(n,buf.size)]
    return ([buf.states[i] for i in idx],
            [buf.actions[i] for i in idx],
            [buf.rewards[i] for i in idx],
            [buf.next_states[i] for i in idx])
end

# ─── Metrics ──────────────────────────────────────────────────────────────────
mutable struct EpisodeMetrics
    wait_times      ::Vector{Float64}
    emergency_waits ::Vector{Float64}
    max_wait_times  ::Vector{Float64}
    queue_lengths   ::Vector{Float64}
    phase_changes   ::Vector{Float64}
    throughputs     ::Vector{Float64}
    rewards         ::Vector{Float64}
end
EpisodeMetrics() = EpisodeMetrics(repeat([Float64[]],7)...)

# ─── SIGMA SUMO agent state ───────────────────────────────────────────────────
mutable struct SIGMAAgent
    actor         ::ActorNetwork
    critic        ::CriticNetwork
    actor_target  ::ActorNetwork
    critic_target ::CriticNetwork
    replay        ::ReplayBuffer
    alpha         ::Vector{Float64}
    update_count  ::Int
    rng           ::AbstractRNG
end

function SIGMAAgent(actor, critic; alpha=ALPHA_VEC, seed=0)
    return SIGMAAgent(actor, critic,
                      copy_actor(actor), copy_critic(critic),
                      ReplayBuffer(), alpha, 0, MersenneTwister(seed))
end

# ─── Single decision step ─────────────────────────────────────────────────────
"""
Run one RL decision step in SUMO:
  - Build state from SUMO
  - Select action (stochastic if train=true, greedy otherwise)
  - Execute phase in SUMO
  - Compute reward
  - Store transition and update networks
"""
function step!(agent::SIGMAAgent, env::SumoEnv;
               train::Bool=true, vehicle_ids::Vector{String}=String[])
    # Build current state
    s, l_in, l_out, Mq, Gamma = SumoInterface.build_state(env)

    # Detect emergency
    emg_active, theta_new = SumoInterface.detect_emergency(env, vehicle_ids)
    if emg_active
        env.theta = theta_new
        env.emergency_active = true
        s[end-15:end] .= theta_new  # update theta slice in state
    end

    # Action selection
    pi = actor_predict(agent.actor, s)
    if train
        cdf   = cumsum(pi)
        u     = rand(agent.rng)
        a_idx = something(findfirst(x->x>=u, cdf), N_PHASES)
    else
        a_idx = argmax(pi)
    end
    a_oh = zeros(N_PHASES); a_oh[a_idx] = 1.0

    # Execute in SUMO
    old_l_in = copy(l_in)
    SumoInterface.execute_phase!(env, a_idx)

    # Read next state
    s_next, l_in_next, _, Mq_next, Gamma_next = SumoInterface.build_state(env)
    env.prev_phase_oh = a_oh

    # Compute reward
    pi_next = actor_predict(agent.actor, s_next)
    r = compute_reward(pi, pi_next, Mq, Gamma, env.theta, agent.alpha)

    # Throughput proxy
    throughput = sum(max.(old_l_in .- l_in_next, 0.0))

    # Store in replay
    push_replay!(agent.replay, s, a_oh, r, s_next)

    # Network updates
    if train && agent.replay.size >= ONLINE_NBATCH
        _update_networks!(agent)
    end

    return (
        state=s, action=a_idx, reward=r,
        wait=mean(Gamma_next), max_wait=maximum(Gamma_next),
        queue=mean(l_in_next), throughput=throughput,
        emergency=env.emergency_active,
        phase_change=norm(vec(a_oh'*ATR).-vec(env.prev_phase_oh'*ATR))^2/8.0
    )
end

function _update_networks!(agent::SIGMAAgent)
    ss, as, rs, sns = sample(agent.replay, ONLINE_NBATCH, agent.rng)

    for i in 1:length(ss)
        a_next_idx, Q_next = critic_best_action(agent.critic_target, sns[i])
        a_next_oh = zeros(N_PHASES); a_next_oh[a_next_idx]=1.0
        y   = rs[i] + GAMMA*Q_next
        err = critic_td_update!(agent.critic, ss[i], as[i], y)

        Q_sa   = critic_predict(agent.critic, ss[i], as[i])
        Q_sn   = critic_predict(agent.critic, sns[i], a_next_oh)
        δ      = rs[i] + GAMMA*Q_sn - Q_sa
        actor_pg_update!(agent.actor, ss[i], as[i], δ)
    end

    agent.update_count += 1
    if agent.update_count % TARGET_UPDATE == 0
        soft_update!(agent.actor_target,  agent.actor,  TAU_SOFT)
        soft_update!(agent.critic_target, agent.critic, TAU_SOFT)
    end
end

# ─── Full episode runner ──────────────────────────────────────────────────────
"""
Run one complete SUMO episode (T_steps RL decisions).
Returns per-step metrics.
"""
function run_episode!(agent::SIGMAAgent, env::SumoEnv;
                       T_steps::Int=120, train::Bool=true,
                       vehicle_ids_fn::Function=()->String[])
    m = EpisodeMetrics()
    for t in 1:T_steps
        vids = vehicle_ids_fn()
        res  = step!(agent, env; train=train, vehicle_ids=vids)

        push!(m.wait_times,    res.wait)
        push!(m.max_wait_times,res.max_wait)
        push!(m.queue_lengths, res.queue)
        push!(m.phase_changes, res.phase_change)
        push!(m.throughputs,   res.throughput)
        push!(m.rewards,       res.reward)
        res.emergency && push!(m.emergency_waits, res.max_wait)
    end
    return m
end

# ─── Baselines (SUMO-aware) ───────────────────────────────────────────────────
"""Fixed-time baseline: cycle phases 1..8 each for one step."""
function run_fixed_time!(env::SumoEnv; T_steps=120, rng=MersenneTwister(0))
    m = EpisodeMetrics()
    phase = 1
    for t in 1:T_steps
        old_l_in, _ = SumoInterface.read_queues(env)
        SumoInterface.execute_phase!(env, phase)
        _, l_in_next, _, _, Gamma_next = SumoInterface.build_state(env)
        throughput = sum(max.(old_l_in .- l_in_next, 0.0))
        push!(m.wait_times,     mean(Gamma_next))
        push!(m.max_wait_times, maximum(Gamma_next))
        push!(m.queue_lengths,  mean(l_in_next))
        push!(m.throughputs,    throughput)
        push!(m.phase_changes,  0.0)
        push!(m.rewards,        0.0)
        phase = mod1(phase+1, N_PHASES)
    end
    return m
end

"""Actuated baseline: max-pressure phase selection."""
function run_actuated!(env::SumoEnv; T_steps=120, G_min=1, G_max=6, rng=MersenneTwister(0))
    m = EpisodeMetrics()
    current_phase=1; steps_green=0
    for t in 1:T_steps
        old_l_in, _ = SumoInterface.read_queues(env)
        _, l_in, l_out, Mq, _ = SumoInterface.build_state(env)

        steps_green += 1
        if steps_green >= G_min
            if steps_green >= G_max || maximum(l_in) < 0.1*Q_MAX
                best_p=current_phase; best_v=-Inf
                for p in 1:N_PHASES
                    v = sum(ATR[p,:].*Mq)
                    if v>best_v; best_v=v; best_p=p; end
                end
                if best_p != current_phase; current_phase=best_p; steps_green=0; end
            end
        end

        SumoInterface.execute_phase!(env, current_phase)
        _, l_in_next, _, _, Gamma_next = SumoInterface.build_state(env)
        throughput = sum(max.(old_l_in .- l_in_next, 0.0))
        push!(m.wait_times,     mean(Gamma_next))
        push!(m.max_wait_times, maximum(Gamma_next))
        push!(m.queue_lengths,  mean(l_in_next))
        push!(m.throughputs,    throughput)
        push!(m.phase_changes,  0.0)
        push!(m.rewards,        0.0)
    end
    return m
end

end # module SumoAgent
