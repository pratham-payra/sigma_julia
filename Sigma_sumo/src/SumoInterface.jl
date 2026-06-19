"""
SumoInterface.jl - SUMO environment wrapper for SIGMA.

Manages:
  - SUMO process lifecycle (launch, close)
  - Four-way intersection state extraction via TraCI
  - Phase-to-TraCI-state string mapping
  - SIGMA state vector construction from live SUMO data
  - Emergency vehicle detection (by vehicle type)
  - Step execution with yellow-phase transition handling

The SIGMA state vector s = [p_{t-1}·ATR (16), Mq (16), Γ (4), θ (16)] = 52-dim
is built from SUMO lane data at each decision step.
"""
module SumoInterface

using Random, LinearAlgebra, Printf
using ..Constants
using ..TraCI

# ─── SUMO intersection description ───────────────────────────────────────────
struct IntersectionConfig
    tl_id       ::String                  # SUMO traffic light ID
    # Lanes per approach: [E_lanes, N_lanes, W_lanes, S_lanes]
    # Each is a list of SUMO lane IDs for that approach
    incoming_lanes::Vector{Vector{String}}  # 4 × n_lanes_per_approach
    outgoing_lanes::Vector{Vector{String}}
    sumo_cfg    ::String                  # path to .sumocfg file
    n_phases    ::Int                     # number of SUMO signal phases
end

"""Default Kolkata-style four-way intersection config (placeholder IDs).
Real IDs come from the .sumocfg / .net.xml files."""
function default_kolkata_config(;
    tl_id      = "J0",
    sumo_cfg   = "kolkata.sumocfg",
    n_phases   = 8
)
    # 3 lanes per approach (left-turn, straight, right-turn)
    incoming_lanes = [
        ["E_in_0","E_in_1","E_in_2"],  # from East
        ["N_in_0","N_in_1","N_in_2"],  # from North
        ["W_in_0","W_in_1","W_in_2"],  # from West
        ["S_in_0","S_in_1","S_in_2"],  # from South
    ]
    outgoing_lanes = [
        ["E_out_0","E_out_1","E_out_2"],
        ["N_out_0","N_out_1","N_out_2"],
        ["W_out_0","W_out_1","W_out_2"],
        ["S_out_0","S_out_1","S_out_2"],
    ]
    return IntersectionConfig(tl_id, incoming_lanes, outgoing_lanes, sumo_cfg, n_phases)
end

# ─── Phase → TraCI RYG string mapping ────────────────────────────────────────
# 12-signal string: [E_left, E_str, E_right,  N_left, N_str, N_right,
#                    W_left, W_str, W_right,  S_left, S_str, S_right]
# G = green, r = red, y = yellow (for transitions)
const PHASE_TO_RYG = Dict{Int,String}(
    1 => "GGGrrrrrrrrr",   # AE1: all E movements green
    2 => "rGGrrrrrrGGr",   # AE2-W: E straight+left, W straight+left
    3 => "GrrrrrrrrrrG",   # AE3-W: E right+U, W right+U  (simplified)
    4 => "rrrrrrGGGrrr",   # AW1: all W movements green
    5 => "rrrGGGrrrrrr",   # AN1: all N movements green
    6 => "rrrrrGrrrGrr",   # AN2-S: N straight+left, S straight+left
    7 => "rrrGrrrrrrrG",   # AN3-S: N right+U, S right+U
    8 => "rrrrrrrrrGGG",   # AS1: all S movements green
)

const YELLOW_RYG = "yyyyyyyyyyyy"  # full yellow transition

# ─── SUMO process management ─────────────────────────────────────────────────
mutable struct SumoEnv
    config      ::IntersectionConfig
    conn        ::Union{TraCI.TraCIConnection, Nothing}
    process     ::Union{Base.Process, Nothing}
    use_gui     ::Bool
    step_length ::Float64   # SUMO sim step (seconds)
    delta_t     ::Int       # SUMO steps per RL decision
    sim_time    ::Float64
    current_phase::Int
    emergency_active::Bool
    emergency_origin::Int
    emergency_dest  ::Int
    theta       ::Vector{Float64}   # 16-dim emergency priority
    prev_phase_oh::Vector{Float64}  # 8-dim one-hot
end

function SumoEnv(config::IntersectionConfig;
                  use_gui     ::Bool  = false,
                  step_length ::Float64 = SUMO_STEP_LENGTH,
                  delta_t     ::Int   = SUMO_DELTA_T)
    return SumoEnv(config, nothing, nothing, use_gui,
                   step_length, delta_t, 0.0, 1,
                   false, 0, 0,
                   zeros(Float64, 16),
                   let v = zeros(Float64, N_PHASES); v[1]=1.0; v end)
end

"""Launch SUMO process and connect via TraCI."""
function launch!(env::SumoEnv;
                  port::Int   = SUMO_PORT,
                  seed::Int   = 42,
                  extra_args  ::Vector{String} = String[])
    binary = env.use_gui ? "sumo-gui" : "sumo"
    args = [
        binary,
        "-c", env.config.sumo_cfg,
        "--remote-port", string(port),
        "--step-length", string(env.step_length),
        "--seed", string(seed),
        "--no-warnings", "true",
        "--quit-on-end", "true",
    ]
    append!(args, extra_args)

    try
        env.process = run(Cmd(args); wait=false)
    catch e
        @warn "Could not launch SUMO binary '$binary': $e"
        @warn "Proceeding in mock mode (no real SUMO)."
        env.conn = nothing
        return false
    end

    sleep(2.0)  # wait for SUMO to start listening

    try
        env.conn = TraCI.connect(SUMO_HOST, port)
        @printf("[SUMO] Connected to %s:%d\n", SUMO_HOST, port)
        return true
    catch e
        @warn "TraCI connection failed: $e. Running in mock mode."
        env.conn = nothing
        return false
    end
end

"""Close SUMO and TraCI connection."""
function close!(env::SumoEnv)
    if env.conn !== nothing
        try TraCI.close_connection(env.conn) catch; end
        env.conn = nothing
    end
    if env.process !== nothing
        try kill(env.process) catch; end
        env.process = nothing
    end
end

# ─── State extraction from SUMO ──────────────────────────────────────────────
"""
Read per-approach queue lengths from SUMO lane halt counts.
Returns l_in (4-dim) and l_out (4-dim).
"""
function read_queues(env::SumoEnv)
    l_in  = zeros(Float64, 4)
    l_out = zeros(Float64, 4)

    if env.conn === nothing
        # Mock: random small queues for offline testing
        l_in  = max.(rand(4) .* 10.0, 0.0)
        l_out = max.(rand(4) .* 5.0, 0.0)
        return l_in, l_out
    end

    for i in 1:4
        for lane in env.config.incoming_lanes[i]
            l_in[i] += TraCI.get_lane_halt_number(env.conn, lane)
        end
        for lane in env.config.outgoing_lanes[i]
            l_out[i] += TraCI.get_lane_halt_number(env.conn, lane)
        end
        l_in[i]  = min(l_in[i],  Float64(Q_MAX))
        l_out[i] = min(l_out[i], Float64(Q_MAX))
    end
    return l_in, l_out
end

"""Read per-approach maximum waiting times from SUMO lanes (seconds)."""
function read_max_waiting_times(env::SumoEnv)
    Gamma = zeros(Float64, 4)
    if env.conn === nothing
        return rand(4) .* 120.0
    end
    for i in 1:4
        max_wt = 0.0
        for lane in env.config.incoming_lanes[i]
            wt = TraCI.get_lane_waiting_time(env.conn, lane)
            max_wt = max(max_wt, wt)
        end
        Gamma[i] = max_wt
    end
    return Gamma
end

"""Build net pressure mask from queue lengths."""
function build_pressure_mask(l_in::Vector{Float64}, l_out::Vector{Float64})
    Mq_in = Float64[]
    for i in 1:4; append!(Mq_in, fill(l_in[i], 4)); end
    Mq_out = Float64[]
    for i in 1:4; append!(Mq_out, l_out); end
    Mprime = Mq_in .- Mq_out
    return Mprime .- minimum(Mprime)
end

"""Build full 52-dim SIGMA state vector from live SUMO data."""
function build_state(env::SumoEnv)
    l_in, l_out = read_queues(env)
    Gamma       = read_max_waiting_times(env)
    Mq          = build_pressure_mask(l_in, l_out)
    prev_16     = vec(env.prev_phase_oh' * ATR)
    return vcat(prev_16, Mq, Gamma, env.theta), l_in, l_out, Mq, Gamma
end

# ─── Emergency vehicle detection ─────────────────────────────────────────────
const EMERGENCY_VTYPES = Set(["emergency","ambulance","fire","police",
                               "ambulance_car","fire_truck","police_car"])

"""
Scan SUMO for emergency vehicles; if found, build priority vector θ.
Returns (active::Bool, theta::Vector{Float64}).
"""
function detect_emergency(env::SumoEnv, vehicle_ids::Vector{String})
    if env.conn === nothing
        return false, zeros(Float64, 16)
    end
    for vid in vehicle_ids
        vtype = TraCI.get_vehicle_type(env.conn, vid)
        if lowercase(vtype) in EMERGENCY_VTYPES
            # Determine origin approach by vehicle position (simplified: use route ID heuristic)
            origin = _guess_approach_from_id(vid)
            dest   = mod1(origin + 2, 4)  # opposite direction as default
            theta  = encode_emergency(origin, dest)
            return true, theta
        end
    end
    return false, zeros(Float64, 16)
end

function _guess_approach_from_id(vid::String)
    vid_l = lowercase(vid)
    contains(vid_l, "_e") && return 1
    contains(vid_l, "_n") && return 2
    contains(vid_l, "_w") && return 3
    contains(vid_l, "_s") && return 4
    return rand(1:4)
end

"""Rule-based emergency priority vector encoder."""
function encode_emergency(origin::Int, dest::Int)
    theta = zeros(Float64, 16)
    if 1 <= origin <= 4
        if 1 <= dest <= 4 && dest != origin
            theta[(origin-1)*4 + dest] = 1.0
        else
            for j in 1:4; theta[(origin-1)*4 + j] = 1.0; end
        end
    end
    return theta
end

# ─── Phase execution in SUMO ─────────────────────────────────────────────────
"""
Execute a SIGMA phase index in SUMO:
1. If changing phase, insert yellow transition for YELLOW_DURATION seconds.
2. Set new green phase.
3. Advance SUMO by delta_t steps.
"""
function execute_phase!(env::SumoEnv, phase_idx::Int)
    if env.conn === nothing
        # Mock: just advance sim time
        env.sim_time += env.delta_t * env.step_length
        env.current_phase = phase_idx
        return
    end

    # Insert yellow if phase changes
    if phase_idx != env.current_phase
        TraCI.set_tl_state!(env.conn, env.config.tl_id, YELLOW_RYG)
        for _ in 1:YELLOW_DURATION
            TraCI.simulation_step!(env.conn, env.sim_time + env.step_length)
            env.sim_time += env.step_length
        end
    end

    # Set new green phase
    if haskey(PHASE_TO_RYG, phase_idx)
        TraCI.set_tl_state!(env.conn, env.config.tl_id, PHASE_TO_RYG[phase_idx])
    else
        TraCI.set_tl_phase!(env.conn, env.config.tl_id, phase_idx - 1)  # 0-indexed
    end

    # Advance simulation by delta_t steps
    for _ in 1:env.delta_t
        TraCI.simulation_step!(env.conn, env.sim_time + env.step_length)
        env.sim_time += env.step_length
    end

    env.current_phase = phase_idx
end

end # module SumoInterface
