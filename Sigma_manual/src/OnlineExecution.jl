"""
OnlineExecution.jl - Online actor-critic execution loop (Appendix A.3).

Closed-loop execution:
  1. Observe state (with optional LLM emergency injection)
  2. Sample action from actor π(·|s)
  3. Execute phase, observe reward r and next state s'
  4. Store in replay buffer D
  5. When |D| ≥ N_batch: update critic via TD loss, update actor via PG
  6. Periodically soft-update target networks
"""
module OnlineExecution

using Random, LinearAlgebra, Statistics, Printf
using ..Constants
using ..Networks
using ..Environment
using ..Augmentation

# ─────────────────────────────────────────────────────────────────────────────
# Replay buffer
# ─────────────────────────────────────────────────────────────────────────────
mutable struct ReplayBuffer
    states     ::Vector{Vector{Float64}}
    actions    ::Vector{Vector{Float64}}
    rewards    ::Vector{Float64}
    next_states::Vector{Vector{Float64}}
    capacity   ::Int
    size       ::Int
    ptr        ::Int
end

function ReplayBuffer(capacity::Int)
    return ReplayBuffer(
        Vector{Vector{Float64}}(undef, capacity),
        Vector{Vector{Float64}}(undef, capacity),
        zeros(Float64, capacity),
        Vector{Vector{Float64}}(undef, capacity),
        capacity, 0, 1
    )
end

function push_transition!(buf::ReplayBuffer,
                           s, a, r, s_next)
    buf.states[buf.ptr]      = s
    buf.actions[buf.ptr]     = a
    buf.rewards[buf.ptr]     = r
    buf.next_states[buf.ptr] = s_next
    buf.ptr  = mod1(buf.ptr + 1, buf.capacity)
    buf.size = min(buf.size + 1, buf.capacity)
end

function sample_batch(buf::ReplayBuffer, batch_size::Int, rng::AbstractRNG)
    idx = randperm(rng, buf.size)[1:min(batch_size, buf.size)]
    return (
        states      = [buf.states[i]      for i in idx],
        actions     = [buf.actions[i]     for i in idx],
        rewards     = [buf.rewards[i]     for i in idx],
        next_states = [buf.next_states[i] for i in idx]
    )
end

# ─────────────────────────────────────────────────────────────────────────────
# SIGMA online execution (single intersection, one episode)
# Returns named tuple of performance metrics
# ─────────────────────────────────────────────────────────────────────────────
function run_online_episode!(
    actor         ::ActorNetwork,
    critic        ::CriticNetwork,
    actor_target  ::ActorNetwork,
    critic_target ::CriticNetwork,
    replay        ::ReplayBuffer,
    env_state     ::IntersectionState;
    T_steps       ::Int = 120,
    alpha         ::Vector{Float64} = ALPHA_VEC,
    rng           ::AbstractRNG = MersenneTwister(0),
    update_step   ::Ref{Int} = Ref(0)
)
    prev_phase_oh = zeros(Float64, N_PHASES); prev_phase_oh[1] = 1.0
    total_reward  = 0.0
    metrics = (
        wait_times       = Float64[],
        emergency_waits  = Float64[],
        queue_lengths    = Float64[],
        phase_changes    = Float64[],
        throughputs      = Float64[]
    )

    for t in 1:T_steps
        prev_phase_16 = vec(prev_phase_oh' * ATR)
        s = vcat(prev_phase_16, env_state.Mq, env_state.Gamma, env_state.theta)

        # Sample action from actor
        pi = actor_predict(actor, s)
        # Stochastic sampling via inverse CDF
        cdf = cumsum(pi)
        u   = rand(rng)
        a_idx = findfirst(x -> x >= u, cdf)
        a_idx = isnothing(a_idx) ? N_PHASES : a_idx
        a_oh  = zeros(Float64, N_PHASES); a_oh[a_idx] = 1.0

        # Step environment
        old_l_in = copy(env_state.l_in)
        step_traffic!(env_state, a_idx, rng)

        # Next state
        next_prev_16 = vec(a_oh' * ATR)
        s_next = vcat(next_prev_16, env_state.Mq, env_state.Gamma, env_state.theta)

        # Compute reward
        pi_next = actor_predict(actor, s_next)
        r = compute_reward(pi, pi_next, env_state.Mq, env_state.Gamma, env_state.theta, alpha)

        push_transition!(replay, s, a_oh, r, s_next)
        total_reward += r

        # Track metrics
        push!(metrics.wait_times,      mean(env_state.Gamma))
        push!(metrics.queue_lengths,   mean(env_state.l_in))
        push!(metrics.phase_changes,
              norm(vec(a_oh' * ATR) .- vec(prev_phase_oh' * ATR))^2 / 8.0)
        if env_state.emergency_active
            push!(metrics.emergency_waits, maximum(env_state.Gamma))
        end
        # Throughput: vehicles released this step
        released = sum(max.(old_l_in .- env_state.l_in, 0.0))
        push!(metrics.throughputs, released)

        # Network updates
        if replay.size >= ONLINE_NBATCH
            batch = sample_batch(replay, ONLINE_NBATCH, rng)

            # Critic TD update
            for i in 1:length(batch.states)
                s_b  = batch.states[i]
                a_b  = batch.actions[i]
                r_b  = batch.rewards[i]
                s_nb = batch.next_states[i]

                a_next_idx, Q_next = critic_best_action(critic_target, s_nb)
                a_next_oh = zeros(Float64, N_PHASES); a_next_oh[a_next_idx] = 1.0
                y = r_b + GAMMA * Q_next
                critic_td_update!(critic, s_b, a_b, y)

                # Actor policy-gradient update
                Q_sa    = critic_predict(critic, s_b, a_b)
                Q_snext = critic_predict(critic, s_nb, a_next_oh)
                δ       = r_b + GAMMA * Q_snext - Q_sa
                actor_pg_update!(actor, s_b, a_b, δ)
            end

            update_step[] += 1

            # Soft-update target networks
            if update_step[] % TARGET_UPDATE == 0
                soft_update!(actor_target,  actor,  TAU_SOFT)
                soft_update!(critic_target, critic, TAU_SOFT)
            end
        end

        prev_phase_oh = a_oh
    end

    return total_reward, metrics
end

# ─────────────────────────────────────────────────────────────────────────────
# Full online training loop (multiple episodes)
# ─────────────────────────────────────────────────────────────────────────────
function train_online!(
    actor    ::ActorNetwork,
    critic   ::CriticNetwork;
    n_episodes::Int = 2000,
    T_steps  ::Int = 120,
    seed     ::Int = 0,
    verbose  ::Bool = true
)
    rng          = MersenneTwister(seed)
    actor_target  = copy_actor(actor)
    critic_target = copy_critic(critic)
    replay        = ReplayBuffer(ONLINE_REPLAY)
    update_step   = Ref(0)

    ep_rewards = Float64[]
    ep_awt     = Float64[]
    ep_aewt    = Float64[]
    ep_aql     = Float64[]
    ep_apc     = Float64[]
    ep_atp     = Float64[]

    for ep in 1:n_episodes
        env_state = IntersectionState()

        # Random emergency
        if rand(rng) < PE_EMERGENCY
            orig = rand(rng, 1:4)
            dest = rand(rng, setdiff(1:4, [orig]))
            env_state.theta = encode_emergency(orig, dest)
            env_state.emergency_active = true
        end

        r_total, metrics = run_online_episode!(
            actor, critic, actor_target, critic_target, replay, env_state;
            T_steps=T_steps, rng=rng, update_step=update_step
        )

        push!(ep_rewards, r_total)
        push!(ep_awt,  mean(metrics.wait_times))
        push!(ep_aewt, isempty(metrics.emergency_waits) ? 0.0 : mean(metrics.emergency_waits))
        push!(ep_aql,  mean(metrics.queue_lengths))
        push!(ep_apc,  mean(metrics.phase_changes))
        push!(ep_atp,  mean(metrics.throughputs))

        if verbose && (ep % 100 == 0 || ep == 1)
            @printf("[Online] Ep %4d/%d  R=%.2f  AWT=%.1f  AEWT=%.1f  AQL=%.1f  ATP=%.2f\n",
                    ep, n_episodes, r_total, ep_awt[end], ep_aewt[end], ep_aql[end], ep_atp[end])
        end
    end

    return (
        actor  = actor,
        critic = critic,
        rewards = ep_rewards,
        awt  = ep_awt,
        aewt = ep_aewt,
        aql  = ep_aql,
        apc  = ep_apc,
        atp  = ep_atp
    )
end

end # module
