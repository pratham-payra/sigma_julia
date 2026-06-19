"""
main.jl - Entry point for SIGMA manual synthetic simulation.

Usage:
  julia main.jl                    # full experiment (default settings)
  julia main.jl --quick            # quick run (fewer episodes for testing)
  julia main.jl --ablation         # ablation study only
  julia main.jl --train-only       # train and save models only

The script adds the src/ directory to LOAD_PATH and instantiates the project.
"""

using Pkg
Pkg.activate(@__DIR__)
try
    Pkg.instantiate()
catch e
    @warn "Could not instantiate project (may already be set up): $e"
end

push!(LOAD_PATH, joinpath(@__DIR__, "src"))
include(joinpath(@__DIR__, "src", "SigmaManual.jl"))
using .SigmaManual
using Printf, Random

# ─────────────────────────────────────────────────────────────────────────────
# Parse command-line arguments
# ─────────────────────────────────────────────────────────────────────────────
quick    = "--quick"    in ARGS
ablation = "--ablation" in ARGS
trainonly= "--train-only" in ARGS

if quick
    println("Running in QUICK mode (reduced episode counts)...")
    N_DATA   = 50
    T_STEPS  = 60
    A_EPOCHS = 20
    FQI      = 10
    N_ONLINE = 100
    N_EVAL   = 50
else
    N_DATA   = 200
    T_STEPS  = 120
    A_EPOCHS = 80
    FQI      = 20
    N_ONLINE = 500
    N_EVAL   = 200
end

if ablation
    println("\n" * "="^60)
    println("  SIGMA Ablation Study")
    println("="^60)
    run_ablation(
        n_episodes_data = N_DATA,
        T_steps         = T_STEPS,
        n_eval          = N_EVAL ÷ 2,
        seed            = 42,
        verbose         = true
    )
elseif trainonly
    println("\nTraining SIGMA only...")
    actor, critic, result = train_sigma(
        n_episodes_data = N_DATA,
        T_steps_data    = T_STEPS,
        actor_epochs    = A_EPOCHS,
        fqi_iters       = FQI,
        online_episodes = N_ONLINE,
        T_steps_online  = T_STEPS,
        seed            = 42,
        verbose         = true,
        save_path       = joinpath(@__DIR__, "sigma_model.jls")
    )
    println("\nTraining complete. Model saved to sigma_model.jls")
else
    println("\nRunning full SIGMA experiment...")
    method_results, actor, critic = run_experiment(
        n_episodes_data = N_DATA,
        T_steps         = T_STEPS,
        actor_epochs    = A_EPOCHS,
        fqi_iters       = FQI,
        online_train    = N_ONLINE,
        n_eval          = N_EVAL,
        seed            = 42,
        verbose         = true
    )

    println("\nExperiment complete!")
    println("Results stored in method_results Dict with keys: ",
            join(keys(method_results), ", "))
end
