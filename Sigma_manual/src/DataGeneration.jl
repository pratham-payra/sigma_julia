"""
DataGeneration.jl - Synthetic offline dataset generation (Appendix A.7, A.8).

Generates trajectories using non-homogeneous Poisson arrivals/departures,
constructs pressure masks, injects emergency events, encodes priority vectors,
evaluates all actions via reward, and stores the best action per step.
"""
module DataGeneration

using Random, Distributions, LinearAlgebra, Statistics
using ..Constants
using ..Environment
using ..Augmentation

# ─────────────────────────────────────────────────────────────────────────────
# Emergency scenario library (40 scenarios, 5 categories)
# ─────────────────────────────────────────────────────────────────────────────
struct EmergencyScenario
    vehicle_type::String
    origin      ::Int   # 1=E,2=N,3=W,4=S
    destination ::Int
    description ::String
end

function build_scenario_library()
    types = ["Ambulance", "Fire truck", "Police car", "Rescue vehicle", "Hazmat truck"]
    scenarios = EmergencyScenario[]
    for (i, vtype) in enumerate(types)
        for orig in 1:4, dest in 1:4
            orig == dest && continue
            dir_names = ["East", "North", "West", "South"]
            desc = "$vtype approaching from $(dir_names[orig]) and traveling toward $(dir_names[dest])"
            push!(scenarios, EmergencyScenario(vtype, orig, dest, desc))
        end
    end
    # truncate or pad to exactly 40
    while length(scenarios) < 40
        push!(scenarios, scenarios[mod1(length(scenarios)+1, length(scenarios))])
    end
    return scenarios[1:40]
end

const SCENARIO_LIBRARY = build_scenario_library()

# ─────────────────────────────────────────────────────────────────────────────
# Generate one episode of synthetic data
# ─────────────────────────────────────────────────────────────────────────────
function generate_episode(
    T       ::Int,        # number of steps per episode
    rng     ::AbstractRNG,
    alpha   ::Vector{Float64} = ALPHA_VEC
)
    state = IntersectionState()
    transitions = Augmentation.Transition[]

    # Sample emergency for this episode
    has_emergency = rand(rng) < PE_EMERGENCY
    emergency = has_emergency ? rand(rng, SCENARIO_LIBRARY) : nothing
    theta = has_emergency ? encode_emergency(emergency.origin, emergency.destination) :
                            zeros(Float64, 16)
    state.theta = theta
    state.emergency_active = has_emergency

    prev_phase_oh = zeros(Float64, N_PHASES)
    prev_phase_oh[1] = 1.0   # start with phase 1

    for t in 1:T
        prev_phase_16 = vec(prev_phase_oh' * ATR)
        Mq   = build_pressure_mask(state.l_in, state.l_out)
        Gamma = copy(state.Gamma)

        # Build state vector
        s = vcat(prev_phase_16, Mq, Gamma, theta)

        # Random alpha for diversity (paper: αi ~ Uniform, sum=5)
        α_rand = rand(rng, 5)
        α_rand = α_rand ./ sum(α_rand) .* 5.0

        # Evaluate all actions
        r_vec = zeros(Float64, N_PHASES)
        for a in 1:N_PHASES
            pi_a  = zeros(Float64, N_PHASES); pi_a[a] = 1.0
            u = compute_utilities(pi_a, pi_a, Mq, Gamma, theta)
            r_vec[a] = α_rand[1]*u.M + α_rand[2]*u.S + α_rand[3]*u.Q + α_rand[4]*u.W + α_rand[5]*u.E
        end

        # Best action
        a_star = argmax(r_vec)
        a_oh   = zeros(Float64, N_PHASES); a_oh[a_star] = 1.0
        r_star = r_vec[a_star]

        # Per-utility reward vector for critic pretraining
        pi_a  = copy(a_oh)
        u = compute_utilities(pi_a, pi_a, Mq, Gamma, theta)
        rv5 = [u.M, u.S, u.Q, u.W, u.E]

        # Store transition
        push!(transitions, Augmentation.Transition(
            s, a_oh, r_star, rv5,
            prev_phase_16, Mq, copy(state.l_in), Gamma, theta
        ))

        # Step environment
        step_traffic!(state, a_star, rng)

        prev_phase_oh = a_oh
    end

    return transitions
end

"""
Generate full offline dataset D' by running N_episodes episodes of length T_steps.
Then apply C4 augmentation to produce DF (4× size).
"""
function generate_dataset(;
    N_episodes ::Int = 500,
    T_steps    ::Int = 120,   # 120 steps × 30s = 3600s per episode
    seed       ::Int = 42,
    augment    ::Bool = true
)
    rng = MersenneTwister(seed)
    D_prime = Augmentation.Transition[]

    for ep in 1:N_episodes
        ep_transitions = generate_episode(T_steps, rng)
        append!(D_prime, ep_transitions)
    end

    if augment
        DF = Augmentation.augment_dataset(D_prime)
    else
        DF = D_prime
    end

    return DF, D_prime
end

end # module
