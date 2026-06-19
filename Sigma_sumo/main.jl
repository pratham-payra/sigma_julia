"""
main.jl - Entry point for SIGMA SUMO simulation.

Usage:
  julia main.jl                   # mock experiment (no SUMO needed)
  julia main.jl --sumo            # full experiment with real SUMO
  julia main.jl --sumo-gui        # with SUMO GUI
  julia main.jl --eval MODEL.jls  # evaluate saved model
  julia main.jl --quick           # quick smoke-test

SUMO requirements (for --sumo mode):
  - SUMO installed and on PATH (https://sumo.dlr.de)
  - kolkata.sumocfg + network files in current directory
  - Or update sumo_cfg kwarg below to point to your .sumocfg

Mock mode (default):
  - No SUMO required; uses internal Poisson traffic model
  - Full actor-critic training pipeline still runs
  - Produces equivalent performance tables
"""

using Pkg
Pkg.activate(@__DIR__)
try Pkg.instantiate() catch e; @warn "Pkg.instantiate: $e"; end

push!(LOAD_PATH, joinpath(@__DIR__, "src"))
include(joinpath(@__DIR__, "src", "SigmaSumo.jl"))
using .SigmaSumo
using Printf

# ─── Parse arguments ─────────────────────────────────────────────────────────
use_sumo   = "--sumo"    in ARGS || "--sumo-gui" in ARGS
use_gui    = "--sumo-gui" in ARGS
quick      = "--quick"   in ARGS
do_eval    = "--eval"    in ARGS

eval_path  = nothing
if do_eval
    idx = findfirst(==("--eval"), ARGS)
    if idx !== nothing && idx < length(ARGS)
        eval_path = ARGS[idx+1]
    end
end

# ─── Quick mode params ────────────────────────────────────────────────────────
if quick
    println("Running in QUICK mode...")
    N_PRE = 30; T_S=60; A_EP=15; FQI=8; N_ON=50; N_EV=20
else
    N_PRE = 200; T_S=120; A_EP=80; FQI=20; N_ON=1000; N_EV=50
end

# ─── Dispatch ─────────────────────────────────────────────────────────────────
if do_eval && eval_path !== nothing
    println("\nEvaluating model: $eval_path")
    if use_sumo
        evaluate_sumo(eval_path; n_eval=N_EV, T_steps=T_S, verbose=true)
    else
        println("(Mock eval — load model and run mock experiment)")
        actor, critic = SigmaSumo.Networks.load_model(eval_path)
        # Mini mock eval
        agent  = SigmaSumo.SumoAgent.SIGMAAgent(actor, critic)
        config = SigmaSumo.SumoInterface.default_kolkata_config()
        results = Dict{String,Vector{SigmaSumo.Evaluation.KPIResult}}(
            "SIGMA"=>[], "Fixed-Time"=>[], "Actuated"=>[])
        for ep in 1:N_EV
            env = SigmaSumo.SumoInterface.SumoEnv(config)
            m   = SigmaSumo.SumoAgent.run_episode!(agent, env; T_steps=T_S, train=false)
            push!(results["SIGMA"], SigmaSumo.Evaluation.aggregate(m))
            env2 = SigmaSumo.SumoInterface.SumoEnv(config)
            m2   = SigmaSumo.SumoAgent.run_fixed_time!(env2; T_steps=T_S)
            push!(results["Fixed-Time"], SigmaSumo.Evaluation.aggregate(m2))
        end
        SigmaSumo.Evaluation.print_table(results)
    end

elseif use_sumo
    println("\nRunning SIGMA with real SUMO...")
    agent, eval_hist, rewards = train_sigma_sumo(
        sumo_cfg      = "kolkata.sumocfg",
        tl_id         = "J0",
        n_pretrain_eps = N_PRE,
        T_pretrain     = T_S,
        actor_epochs   = A_EP,
        fqi_iters      = FQI,
        n_online_eps   = N_ON,
        T_online       = T_S,
        eval_every     = 100,
        n_eval_eps     = N_EV,
        use_gui        = use_gui,
        seed           = 42,
        verbose        = true,
        save_path      = "sigma_sumo_model.jls"
    )
    println("\nTraining complete. Model saved to sigma_sumo_model.jls")

else
    println("\nRunning SIGMA SUMO mock experiment (no SUMO required)...")
    results, agent, rewards = run_mock_experiment(
        n_pretrain_eps = N_PRE,
        T_steps        = T_S,
        actor_epochs   = A_EP,
        fqi_iters      = FQI,
        n_online_eps   = N_ON,
        n_eval         = N_EV,
        seed           = 42,
        verbose        = true
    )
    println("\nMock experiment complete!")
end
