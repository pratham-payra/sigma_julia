"""
SigmaSumo.jl - Top-level module for SIGMA SUMO simulation.

Exposes:
  train_sigma_sumo()   - full pipeline: pretrain offline → online SUMO training
  evaluate_sumo()      - benchmark SIGMA vs Fixed-Time vs Actuated in SUMO
  run_mock_experiment()- run without SUMO (mock mode for testing/development)
"""
module SigmaSumo

using Printf, Random, LinearAlgebra, Statistics, Serialization

include("Constants.jl")
include("TraCI.jl")
include("SumoInterface.jl")
include("Networks.jl")
include("UtilityFunctions.jl")
include("Pretraining.jl")
include("SumoAgent.jl")
include("Evaluation.jl")

using .Constants
using .TraCI
using .SumoInterface
using .Networks
using .UtilityFunctions
using .Pretraining
using .SumoAgent
using .Evaluation

# ─── Full pipeline ────────────────────────────────────────────────────────────
"""
    train_sigma_sumo(; kwargs...) → (agent, results)

Full SIGMA SUMO training pipeline:
1. Generate synthetic offline data & pretrain actor+critic
2. Launch SUMO (or run in mock mode if SUMO not available)
3. Online actor-critic training against SUMO traffic
4. Periodic evaluation checkpoints
"""
function train_sigma_sumo(;
    sumo_cfg         ::String  = joinpath(@__DIR__,"..","sumo_nets","kolkata.sumocfg"),
    tl_id            ::String  = "J0",
    n_pretrain_eps   ::Int     = 200,
    T_pretrain       ::Int     = 120,
    actor_epochs     ::Int     = 80,
    fqi_iters        ::Int     = 20,
    n_online_eps     ::Int     = 1000,
    T_online         ::Int     = 120,
    eval_every       ::Int     = 100,
    n_eval_eps       ::Int     = 20,
    use_gui          ::Bool    = false,
    sumo_port        ::Int     = SUMO_PORT,
    seed             ::Int     = 42,
    verbose          ::Bool    = true,
    save_path        ::Union{String,Nothing} = nothing
)
    rng = MersenneTwister(seed)

    # ── Step 1: Offline pretraining ──────────────────────────────────────────
    verbose && println("\n[1/3] Generating synthetic dataset & pretraining...")
    DF, _ = Pretraining.generate_dataset(
        N_episodes=n_pretrain_eps, T_steps=T_pretrain, seed=seed, augment=true)
    verbose && @printf("      Augmented dataset: %d transitions\n", length(DF))

    actor  = ActorNetwork()
    critic = CriticNetwork()
    Pretraining.pretrain_actor!(actor, DF; epochs=actor_epochs, verbose=verbose)
    Pretraining.pretrain_critic!(critic, DF; max_iter=fqi_iters, verbose=verbose)
    verbose && println("      Pretraining complete.")

    # ── Step 2: Launch SUMO ──────────────────────────────────────────────────
    verbose && println("\n[2/3] Launching SUMO simulation...")
    config = SumoInterface.default_kolkata_config(tl_id=tl_id, sumo_cfg=sumo_cfg)
    env    = SumoEnv(config; use_gui=use_gui)
    connected = SumoInterface.launch!(env; port=sumo_port, seed=seed)
    if !connected
        verbose && println("      Running in MOCK mode (no real SUMO).")
    end

    # ── Step 3: Online training ──────────────────────────────────────────────
    verbose && println("\n[3/3] Online actor-critic training in SUMO...")
    agent = SIGMAAgent(actor, critic; alpha=ALPHA_VEC, seed=seed+1)

    ep_rewards = Float64[]
    eval_history = Dict{String,Vector{Evaluation.KPIResult}}()

    for ep in 1:n_online_eps
        # Reset environment (re-launch SUMO if needed)
        if ep > 1 && connected
            SumoInterface.close!(env)
            env = SumoEnv(config; use_gui=use_gui)
            SumoInterface.launch!(env; port=sumo_port, seed=seed+ep)
        end

        m = SumoAgent.run_episode!(agent, env;
            T_steps=T_online, train=true)

        push!(ep_rewards, isempty(m.rewards) ? 0.0 : mean(m.rewards))

        if verbose && (ep%50==0 || ep==1)
            @printf("[Online] Ep %4d/%d  R=%.3f  AWT=%.1f  AEWT=%.1f  ATP=%.2f\n",
                    ep, n_online_eps,
                    ep_rewards[end],
                    isempty(m.wait_times) ? 0.0 : mean(m.wait_times),
                    isempty(m.emergency_waits) ? 0.0 : mean(m.emergency_waits),
                    isempty(m.throughputs) ? 0.0 : mean(m.throughputs))
        end

        # Evaluation checkpoint
        if ep % eval_every == 0
            verbose && println("  [Eval checkpoint at ep $ep]")
            eval_res = _eval_all(agent, config; n_eps=n_eval_eps,
                                  T_steps=T_online, port=sumo_port+1,
                                  seed=seed+ep+10000, verbose=false)
            for (k,v) in eval_res
                haskey(eval_history,k) || (eval_history[k] = Evaluation.KPIResult[])
                append!(eval_history[k], v)
            end
            Evaluation.print_table(eval_res)
        end
    end

    SumoInterface.close!(env)

    if save_path !== nothing
        Networks.save_model(save_path, agent.actor, agent.critic)
        verbose && println("Model saved to: $save_path")
    end

    return agent, eval_history, ep_rewards
end

# ─── Evaluation runner ────────────────────────────────────────────────────────
function _eval_all(agent::SumoAgent.SIGMAAgent,
                    config::SumoInterface.IntersectionConfig;
                    n_eps=20, T_steps=120, port=SUMO_PORT+1, seed=999, verbose=false)
    results = Dict{String,Vector{Evaluation.KPIResult}}(
        "SIGMA"       => Evaluation.KPIResult[],
        "Fixed-Time"  => Evaluation.KPIResult[],
        "Actuated"    => Evaluation.KPIResult[],
    )
    rng = MersenneTwister(seed)

    for ep in 1:n_eps
        # SIGMA greedy
        env = SumoEnv(config)
        SumoInterface.launch!(env; port=port, seed=seed+ep)
        m_sigma = SumoAgent.run_episode!(agent, env; T_steps=T_steps, train=false)
        push!(results["SIGMA"], Evaluation.aggregate(m_sigma))
        SumoInterface.close!(env)

        # Fixed-Time
        env = SumoEnv(config)
        SumoInterface.launch!(env; port=port, seed=seed+ep)
        m_ft = SumoAgent.run_fixed_time!(env; T_steps=T_steps)
        push!(results["Fixed-Time"], Evaluation.aggregate(m_ft))
        SumoInterface.close!(env)

        # Actuated
        env = SumoEnv(config)
        SumoInterface.launch!(env; port=port, seed=seed+ep)
        m_act = SumoAgent.run_actuated!(env; T_steps=T_steps)
        push!(results["Actuated"], Evaluation.aggregate(m_act))
        SumoInterface.close!(env)
    end
    return results
end

"""
    evaluate_sumo(model_path; kwargs...)

Load a saved model and benchmark against all baselines in SUMO.
"""
function evaluate_sumo(model_path::String;
                        sumo_cfg  ::String = "kolkata.sumocfg",
                        tl_id     ::String = "J0",
                        n_eval    ::Int    = 50,
                        T_steps   ::Int    = 120,
                        port      ::Int    = SUMO_PORT,
                        seed      ::Int    = 999,
                        verbose   ::Bool   = true)
    actor, critic = Networks.load_model(model_path)
    agent  = SIGMAAgent(actor, critic)
    config = SumoInterface.default_kolkata_config(tl_id=tl_id, sumo_cfg=sumo_cfg)

    verbose && println("\n[Evaluating $n_eval episodes per method...]")
    results = _eval_all(agent, config; n_eps=n_eval, T_steps=T_steps,
                         port=port, seed=seed, verbose=verbose)
    Evaluation.print_table(results)
    return results
end

"""
    run_mock_experiment(; kwargs...)

Run the full experiment without a real SUMO installation.
SUMO is replaced by the mock traffic model in SumoInterface (offline Poisson data).
"""
function run_mock_experiment(;
    n_pretrain_eps::Int  = 100,
    T_steps       ::Int  = 120,
    actor_epochs  ::Int  = 40,
    fqi_iters     ::Int  = 15,
    n_online_eps  ::Int  = 200,
    n_eval        ::Int  = 50,
    seed          ::Int  = 42,
    verbose       ::Bool = true
)
    verbose && println("\n"*"="^60)
    verbose && println("  SIGMA SUMO — Mock Experiment (no SUMO required)")
    verbose && println("="^60)

    rng = MersenneTwister(seed)

    # Offline pretraining
    verbose && println("\n[1/3] Pretraining on synthetic data...")
    DF, _ = Pretraining.generate_dataset(
        N_episodes=n_pretrain_eps, T_steps=T_steps, seed=seed, augment=true)
    actor  = ActorNetwork()
    critic = CriticNetwork()
    Pretraining.pretrain_actor!(actor, DF; epochs=actor_epochs, verbose=verbose)
    Pretraining.pretrain_critic!(critic, DF; max_iter=fqi_iters, verbose=verbose)

    # Online training (mock SUMO)
    verbose && println("\n[2/3] Online training in mock SUMO...")
    agent  = SIGMAAgent(actor, critic; alpha=ALPHA_VEC, seed=seed+1)
    config = SumoInterface.default_kolkata_config()

    ep_rewards = Float64[]
    for ep in 1:n_online_eps
        env = SumoEnv(config)   # no launch → mock mode
        m   = SumoAgent.run_episode!(agent, env; T_steps=T_steps, train=true)
        push!(ep_rewards, isempty(m.rewards) ? 0.0 : mean(m.rewards))
        if verbose && (ep%50==0||ep==1)
            @printf("[Mock] Ep %4d/%d  R=%.3f  AWT=%.1f  ATP=%.2f\n",
                    ep, n_online_eps, ep_rewards[end],
                    isempty(m.wait_times) ? 0.0 : mean(m.wait_times),
                    isempty(m.throughputs) ? 0.0 : mean(m.throughputs))
        end
    end

    # Evaluation
    verbose && println("\n[3/3] Evaluation ($n_eval episodes)...")
    results = Dict{String,Vector{Evaluation.KPIResult}}(
        "SIGMA"=>[],  "Fixed-Time"=>[],  "Actuated"=>[]
    )

    eval_rng = MersenneTwister(seed+999)
    for ep in 1:n_eval
        env_s = SumoEnv(config)
        m_s   = SumoAgent.run_episode!(agent, env_s; T_steps=T_steps, train=false)
        push!(results["SIGMA"], Evaluation.aggregate(m_s))

        env_ft = SumoEnv(config)
        m_ft   = SumoAgent.run_fixed_time!(env_ft; T_steps=T_steps)
        push!(results["Fixed-Time"], Evaluation.aggregate(m_ft))

        env_ac = SumoEnv(config)
        m_ac   = SumoAgent.run_actuated!(env_ac; T_steps=T_steps)
        push!(results["Actuated"], Evaluation.aggregate(m_ac))
    end

    Evaluation.print_table(results)
    return results, agent, ep_rewards
end

end # module SigmaSumo
