"""
SigmaManual.jl - Top-level module for SIGMA manual synthetic simulation.

SIGMA: Signal Intelligence via Guided Multi-objective Actor-Critic for Traffic Control
Paper: "SIGMA: Signal Intelligence via Guided Multi-objective Actor-Critic" (2026)

This module wires together all sub-components and exposes the main entry points:
  - `run_experiment()` : full pipeline (data gen → pretrain → online → evaluate)
  - `run_ablation()`   : ablation study over loss components
  - `train_sigma()`    : returns trained actor+critic networks
"""
module SigmaManual

using Printf, Random, LinearAlgebra, Statistics, Serialization

include("Constants.jl")
include("Environment.jl")
include("Augmentation.jl")
include("Networks.jl")
include("DataGeneration.jl")
include("Pretraining.jl")
include("OnlineExecution.jl")
include("Baselines.jl")
include("Evaluation.jl")

using .Constants
using .Environment
using .Augmentation
using .Networks
using .DataGeneration
using .Pretraining
using .OnlineExecution
using .Baselines
using .Evaluation

# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────
"""
    train_sigma(; kwargs...) → (actor, critic)

Full SIGMA training pipeline:
1. Generate synthetic dataset D' with Poisson traffic + emergency injection
2. Apply C4 rotational augmentation → DF (4× size)
3. Pretrain actor network (cross-entropy + utility regularization)
4. Pretrain critic network (fitted Q-iteration with k-NN next-state approx)
5. Online actor-critic refinement against live simulated traffic
"""
function train_sigma(;
    n_episodes_data::Int  = 300,
    T_steps_data   ::Int  = 120,
    actor_epochs   ::Int  = 100,
    fqi_iters      ::Int  = 30,
    online_episodes::Int  = 1000,
    T_steps_online ::Int  = 120,
    seed           ::Int  = 42,
    verbose        ::Bool = true,
    save_path      ::Union{String,Nothing} = nothing
)
    rng = MersenneTwister(seed)

    # ── Step 1: Dataset generation ──────────────────────────────────────────
    verbose && println("\n[1/4] Generating synthetic dataset...")
    DF, D_prime = DataGeneration.generate_dataset(
        N_episodes = n_episodes_data,
        T_steps    = T_steps_data,
        seed       = seed,
        augment    = true
    )
    verbose && @printf("      D' size: %d  |  DF (augmented) size: %d\n",
                       length(D_prime), length(DF))

    # ── Step 2: Actor pretraining ────────────────────────────────────────────
    verbose && println("\n[2/4] Pretraining actor network...")
    actor = ActorNetwork()
    pretrain_actor!(actor, DF;
        epochs     = actor_epochs,
        batch_size = ACTOR_BATCH,
        rng        = rng,
        verbose    = verbose
    )

    # ── Step 3: Critic pretraining ───────────────────────────────────────────
    verbose && println("\n[3/4] Pretraining critic network (FQI)...")
    critic = CriticNetwork()
    pretrain_critic!(critic, DF;
        max_iter = fqi_iters,
        α        = ALPHA_VEC,
        rng      = rng,
        verbose  = verbose
    )

    # ── Step 4: Online refinement ────────────────────────────────────────────
    verbose && println("\n[4/4] Online actor-critic refinement...")
    result = train_online!(actor, critic;
        n_episodes = online_episodes,
        T_steps    = T_steps_online,
        seed       = seed + 1,
        verbose    = verbose
    )

    # ── Save ─────────────────────────────────────────────────────────────────
    if save_path !== nothing
        serialize(save_path, (actor=actor, critic=critic))
        verbose && println("Models saved to: $save_path")
    end

    return actor, critic, result
end

"""
    run_experiment(; kwargs...)

Full benchmark: train SIGMA, run all baselines, print comparison table.
"""
function run_experiment(;
    n_episodes_data::Int  = 200,
    T_steps        ::Int  = 120,
    actor_epochs   ::Int  = 80,
    fqi_iters      ::Int  = 20,
    online_train   ::Int  = 500,
    n_eval         ::Int  = 200,
    seed           ::Int  = 42,
    verbose        ::Bool = true
)
    verbose && println("\n" * "="^60)
    verbose && println("  SIGMA - Full Experiment")
    verbose && println("="^60)

    # Train SIGMA
    actor, critic, _ = train_sigma(
        n_episodes_data = n_episodes_data,
        T_steps_data    = T_steps,
        actor_epochs    = actor_epochs,
        fqi_iters       = fqi_iters,
        online_episodes = online_train,
        T_steps_online  = T_steps,
        seed            = seed,
        verbose         = verbose
    )

    verbose && println("\n[Evaluating all methods on $n_eval episodes...]")
    rng = MersenneTwister(seed + 100)

    # Evaluate SIGMA
    sigma_results = Evaluation.eval_sigma(actor, critic;
        n_eval=n_eval, T_steps=T_steps, seed=seed+100, verbose=false)

    # Fixed-Time baseline
    ft_results = Evaluation.eval_baseline(
        (env, Ts, r) -> Baselines.run_fixed_time_episode(env; T_steps=Ts, rng=r);
        n_eval=n_eval, T_steps=T_steps, seed=seed+100
    )

    # Actuated baseline
    act_results = Evaluation.eval_baseline(
        (env, Ts, r) -> Baselines.run_actuated_episode(env; T_steps=Ts, rng=r);
        n_eval=n_eval, T_steps=T_steps, seed=seed+100
    )

    # DQN baseline (fresh per eval run for fairness)
    dqn_ctrl = Baselines.DQNController()
    dqn_rng  = MersenneTwister(seed + 200)
    dqn_results = Evaluation.KPIResult[]

    # First train DQN for online_train episodes
    verbose && println("[Training DQN baseline...]")
    dqn_train_rng = MersenneTwister(seed + 50)
    for ep in 1:online_train
        env_state = IntersectionState()
        if rand(dqn_train_rng) < PE_EMERGENCY
            orig = rand(dqn_train_rng, 1:4)
            dest = rand(dqn_train_rng, setdiff(1:4, [orig]))
            env_state.theta = encode_emergency(orig, dest)
            env_state.emergency_active = true
        end
        Baselines.run_dqn_episode(dqn_ctrl, env_state; T_steps=T_steps, rng=dqn_train_rng)
    end

    # Then evaluate
    for ep in 1:n_eval
        env_state = IntersectionState()
        if rand(dqn_rng) < PE_EMERGENCY
            orig = rand(dqn_rng, 1:4)
            dest = rand(dqn_rng, setdiff(1:4, [orig]))
            env_state.theta = encode_emergency(orig, dest)
            env_state.emergency_active = true
        end
        m = Baselines.run_dqn_episode(dqn_ctrl, env_state; T_steps=T_steps, rng=dqn_rng)
        push!(dqn_results, Evaluation.aggregate_metrics(m))
    end

    # Print results table
    method_results = Dict(
        "Fixed-Time"  => ft_results,
        "Actuated"    => act_results,
        "DQN"         => dqn_results,
        "SIGMA (Ours)"=> sigma_results
    )
    Evaluation.print_comparison_table(method_results)

    return method_results, actor, critic
end

"""
    run_ablation(; kwargs...)

Ablation study matching Table 4/5 in the paper.
"""
function run_ablation(;
    n_episodes_data::Int = 150,
    T_steps        ::Int = 120,
    n_eval         ::Int = 100,
    seed           ::Int = 42,
    verbose        ::Bool = true
)
    verbose && println("\n[Generating dataset for ablation...]")
    DF, _ = DataGeneration.generate_dataset(
        N_episodes=n_episodes_data, T_steps=T_steps, seed=seed, augment=true)

    verbose && println("[Running ablation study...]")
    results = Evaluation.run_ablation(DF; n_eval=n_eval, T_steps=T_steps, seed=seed)
    Evaluation.print_comparison_table(results)
    return results
end

end # module SigmaManual
