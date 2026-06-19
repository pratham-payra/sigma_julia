"""
Evaluation.jl - SUMO evaluation metrics and benchmark runner.

Evaluation metrics (Table 2):
  AEWT - Average Emergency Waiting Time (s)
  AMWT - Average Maximum Waiting Time   (s)
  AWT  - Average Waiting Time           (s)
  AQL  - Average Queue Length
  APC  - Average Phase Change magnitude
  TC   - Transition Consistency
  ATP  - Average Throughput (veh/step)
"""
module Evaluation

using Statistics, Printf, LinearAlgebra
using ..Constants
using ..Networks
using ..SumoAgent
using ..SumoInterface

struct KPIResult
    AWT ::Float64; AEWT::Float64; AMWT::Float64
    AQL ::Float64; APC ::Float64; TC  ::Float64; ATP::Float64
end

function aggregate(m::SumoAgent.EpisodeMetrics)
    awt  = isempty(m.wait_times)      ? 0.0 : mean(m.wait_times)
    aewt = isempty(m.emergency_waits) ? 0.0 : mean(m.emergency_waits)
    amwt = isempty(m.max_wait_times)  ? 0.0 : mean(m.max_wait_times)
    aql  = isempty(m.queue_lengths)   ? 0.0 : mean(m.queue_lengths)
    apc  = isempty(m.phase_changes)   ? 0.0 : mean(m.phase_changes)
    atp  = isempty(m.throughputs)     ? 0.0 : mean(m.throughputs)
    tc   = max(0.0, 1.0 - apc)
    return KPIResult(awt, aewt, amwt, aql, apc, tc, atp)
end

function summary_stats(results::Vector{KPIResult})
    f(x) = (mean=mean(x), std=std(x))
    return (AWT =f([r.AWT  for r in results]),
            AEWT=f([r.AEWT for r in results]),
            AMWT=f([r.AMWT for r in results]),
            AQL =f([r.AQL  for r in results]),
            APC =f([r.APC  for r in results]),
            TC  =f([r.TC   for r in results]),
            ATP =f([r.ATP  for r in results]))
end

function print_table(method_results::Dict{String,Vector{KPIResult}})
    println("\n"*"="^92)
    println("  SIGMA SUMO Benchmark — mean±std over $(length(first(values(method_results)))) episodes")
    println("="^92)
    @printf("  %-22s %12s %12s %12s %8s %8s %8s %8s\n",
            "Method","AMWT(s)↓","AWT(s)↓","AEWT(s)↓","AQL↓","APC↓","TC↑","ATP↑")
    println("-"^92)
    for (name,res) in sort(collect(method_results),by=x->x[1])
        s = summary_stats(res)
        @printf("  %-22s %5.1f±%4.1f %5.1f±%4.1f %5.1f±%4.1f %3.1f±%2.1f %5.3f %5.3f %5.2f\n",
                name,
                s.AMWT.mean,s.AMWT.std, s.AWT.mean,s.AWT.std,
                s.AEWT.mean,s.AEWT.std, s.AQL.mean,s.AQL.std,
                s.APC.mean, s.TC.mean,  s.ATP.mean)
    end
    println("="^92)
end

end # module Evaluation
