"""
Evaluation.jl - Evaluation metrics and benchmark runner (Tables 2, 3, 5, 6).

Metrics (Table 2):
  AEWT  - Average Emergency Waiting Time
  AMWT  - Average Maximum Waiting Time per approach
  AWT   - Average Waiting Time (all vehicles)
  AQL   - Average Queue Length
  TPC   - Traffic Phase Change magnitude
  TC    - Transition Consistency
  ATP   - Average Throughput
"""
module Evaluation

using Random, Statistics, Printf, LinearAlgebra
using ..Constants
using ..Networks
using ..Environment
using ..Augmentation
using ..Baselines
using ..OnlineExecution
using ..Pretraining

# ─────────────────────────────────────────────────────────────────────────────
# Aggregate episode metrics into scalar KPIs (Table 2 formulas)
# ─────────────────────────────────────────────────────────────────────────────
struct KPIResult
    AWT  ::Float64   # average waiting time (s)
    AEWT ::Float64   # average emergency waiting time (s)
    AMWT ::Float64   # average max waiting time per approach (s)
    AQL  ::Float64   # average queue length (vehicles)
    APC  ::Float64   # average phase change magnitude
    TC   ::Float64   # transition consistency
    ATP  ::Float64   # average throughput (veh/step)
end

function aggregate_metrics(m::Baselines.EpisodeMetrics)::KPIResult
    awt  = isempty(m.wait_times)      ? 0.0 : mean(m.wait_times)
    aewt = isempty(m.emergency_waits) ? 0.0 : mean(m.emergency_waits)
    amwt = isempty(m.max_wait_times)  ? 0.0 : mean(m.max_wait_times)
    aql  = isempty(m.queue_lengths)   ? 0.0 : mean(m.queue_lengths)
    apc  = isempty(m.phase_changes)   ? 0.0 : mean(m.phase_changes)
    atp  = isempty(m.throughputs)     ? 0.0 : mean(m.throughputs)

    # Transition consistency: TC_t = |‖a_t · Q_p - a_{t+1}‖² - 2|²
    # Proxy: higher phase_change means lower consistency
    tc = max(0.0, 1.0 - apc)

    return KPIResult(awt, aewt, amwt, aql, apc, tc, atp)
end

# ─────────────────────────────────────────────────────────────────────────────
# Run SIGMA (online actor-critic) for N evaluation episodes
# ─────────────────────────────────────────────────────────────────────────────
function eval_sigma(
    actor ::ActorNetwork,
    critic::CriticNetwork;
    n_eval  ::Int = 1000,
    T_steps ::Int = 120,
    seed    ::Int = 999,
    verbose ::Bool = false
)
    rng = MersenneTwister(seed)
    results = KPIResult[]

    for ep in 1:n_eval
        env_state = IntersectionState()
        if rand(rng) < PE_EMERGENCY
            orig = rand(rng, 1:4)
            dest = rand(rng, setdiff(1:4, [orig]))
            env_state.theta = encode_emergency(orig, dest)
            env_state.emergency_active = true
        end

        m = _run_sigma_greedy(actor, env_state; T_steps=T_steps, rng=rng)
        push!(results, aggregate_metrics(m))
    end
    return results
end

"""Greedy (inference-only) rollout of trained SIGMA actor."""
function _run_sigma_greedy(
    actor    ::ActorNetwork,
    env_state::IntersectionState;
    T_steps  ::Int = 120,
    rng      ::AbstractRNG = MersenneTwister(0)
)
    metrics = Baselines.init_metrics()
    prev_phase_oh = zeros(Float64, N_PHASES); prev_phase_oh[1] = 1.0

    for t in 1:T_steps
        prev_phase_16 = vec(prev_phase_oh' * ATR)
        s = vcat(prev_phase_16, env_state.Mq, env_state.Gamma, env_state.theta)

        pi    = actor_predict(actor, s)
        a_idx = argmax(pi)   # greedy
        a_oh  = zeros(Float64, N_PHASES); a_oh[a_idx] = 1.0

        old_l_in = copy(env_state.l_in)
        step_traffic!(env_state, a_idx, rng)

        Baselines.record_metrics!(metrics, env_state, prev_phase_oh, a_oh, old_l_in)
        prev_phase_oh = a_oh
    end
    return metrics
end

# ─────────────────────────────────────────────────────────────────────────────
# Run a baseline controller for N episodes
# ─────────────────────────────────────────────────────────────────────────────
function eval_baseline(controller_fn::Function;
                        n_eval ::Int = 1000,
                        T_steps::Int = 120,
                        seed   ::Int = 999)
    rng = MersenneTwister(seed)
    results = KPIResult[]
    for ep in 1:n_eval
        env_state = IntersectionState()
        if rand(rng) < PE_EMERGENCY
            orig = rand(rng, 1:4)
            dest = rand(rng, setdiff(1:4, [orig]))
            env_state.theta = encode_emergency(orig, dest)
            env_state.emergency_active = true
        end
        m = controller_fn(env_state, T_steps, rng)
        push!(results, aggregate_metrics(m))
    end
    return results
end

# ─────────────────────────────────────────────────────────────────────────────
# Summary statistics
# ─────────────────────────────────────────────────────────────────────────────
function summary_stats(results::Vector{KPIResult})
    awtv  = [r.AWT  for r in results]
    aewtv = [r.AEWT for r in results]
    amwtv = [r.AMWT for r in results]
    aqlv  = [r.AQL  for r in results]
    apcv  = [r.APC  for r in results]
    tcv   = [r.TC   for r in results]
    atpv  = [r.ATP  for r in results]

    return (
        AWT  = (mean=mean(awtv),  std=std(awtv)),
        AEWT = (mean=mean(aewtv), std=std(aewtv)),
        AMWT = (mean=mean(amwtv), std=std(amwtv)),
        AQL  = (mean=mean(aqlv),  std=std(aqlv)),
        APC  = (mean=mean(apcv),  std=std(apcv)),
        TC   = (mean=mean(tcv),   std=std(tcv)),
        ATP  = (mean=mean(atpv),  std=std(atpv))
    )
end

# ─────────────────────────────────────────────────────────────────────────────
# Pretty-print comparison table (Table 3 format)
# ─────────────────────────────────────────────────────────────────────────────
function print_comparison_table(method_results::Dict{String, Vector{KPIResult}})
    println("\n" * "="^90)
    println("  Performance Comparison (mean ± std over $(length(first(values(method_results)))) episodes)")
    println("="^90)
    @printf("  %-20s %12s %12s %12s %8s %8s %8s %8s\n",
            "Method", "AMWT(s)↓", "AWT(s)↓", "AEWT(s)↓", "AQL↓", "APC↓", "TC↑", "ATP↑")
    println("-"^90)

    for (name, results) in sort(collect(method_results), by=x->x[1])
        s = summary_stats(results)
        @printf("  %-20s %5.1f±%4.1f %5.1f±%4.1f %5.1f±%4.1f %3.1f±%2.1f %5.2f %5.2f %5.2f\n",
                name,
                s.AMWT.mean, s.AMWT.std,
                s.AWT.mean,  s.AWT.std,
                s.AEWT.mean, s.AEWT.std,
                s.AQL.mean,  s.AQL.std,
                s.APC.mean,
                s.TC.mean,
                s.ATP.mean)
    end
    println("="^90)
end

# ─────────────────────────────────────────────────────────────────────────────
# Ablation study runner (Table 4/5): disable loss components progressively
# ─────────────────────────────────────────────────────────────────────────────
function run_ablation(
    DF::Vector{Augmentation.Transition};
    n_eval::Int = 200,
    T_steps::Int = 120,
    seed::Int = 777
)
    configs = [
        ("Case1_Full",              [LAMBDA_M, LAMBDA_S, LAMBDA_Q, LAMBDA_W, LAMBDA_E],
                                    [ALPHA_M,  ALPHA_S,  ALPHA_Q,  ALPHA_W,  ALPHA_E]),
        ("Case2_NoEmergency",       [LAMBDA_M, LAMBDA_S, LAMBDA_Q, LAMBDA_W, 0.0],
                                    [ALPHA_M,  ALPHA_S,  ALPHA_Q,  ALPHA_W,  0.0]),
        ("Case3_NoEmerg_NoMWT",     [LAMBDA_M, LAMBDA_S, LAMBDA_Q, 0.0, 0.0],
                                    [ALPHA_M,  ALPHA_S,  ALPHA_Q,  0.0, 0.0]),
        ("Case4_NoEmerg_NoMWT_NoQL",[LAMBDA_M, LAMBDA_S, 0.0, 0.0, 0.0],
                                    [ALPHA_M,  ALPHA_S,  0.0, 0.0, 0.0]),
        ("Case5_EqualWeight",       [1.0, 1.0, 1.0, 1.0, 1.0],
                                    [1.0, 1.0, 1.0, 1.0, 1.0]),
    ]

    all_results = Dict{String, Vector{KPIResult}}()
    for (name, λ, α) in configs
        println("\n--- Running ablation: $name ---")
        actor  = ActorNetwork()
        critic = CriticNetwork()
        Pretraining.pretrain_actor!(actor, DF; epochs=50, verbose=false)
        Pretraining.pretrain_critic!(critic, DF; max_iter=20, verbose=false, α=α)
        res = eval_sigma(actor, critic; n_eval=n_eval, T_steps=T_steps, seed=seed, verbose=false)
        all_results[name] = res
    end
    return all_results
end

end # module
