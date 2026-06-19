"""
UtilityFunctions.jl - Five utility objectives + reward (shared between manual and SUMO).
"""
module UtilityFunctions

using LinearAlgebra
using ..Constants

# U_M: Markovian consistency
function U_M(pi_t::Vector{Float64}, pi_t1::Vector{Float64}, Q=Q_TRANSITION)
    return -norm(pi_t1 .- Q'*pi_t)^2
end

# U_S: Action smoothness (movement-space L2)
function U_S(pi_t::Vector{Float64}, pi_t1::Vector{Float64})
    return -norm(ATR'*pi_t1 .- ATR'*pi_t)^2
end

# U_Q: Queue pressure reduction
function U_Q(pi_t1::Vector{Float64}, Mq::Vector{Float64})
    served = (ATR'*pi_t1)'*Mq
    return served / (max(maximum(Mq),1e-6) + 1.0)
end

# U_W: Waiting time fairness
function U_W(pi_t1::Vector{Float64}, Gamma::Vector{Float64})
    dir_served = zeros(4)
    mv = ATR'*pi_t1
    for i in 1:4
        for j in 1:4; dir_served[i] += mv[(i-1)*4+j]; end
    end
    return -norm(dir_served .- Gamma ./ (max(maximum(Gamma),1e-6)))^2
end

# U_E: Emergency alignment
function U_E(pi_t::Vector{Float64}, theta::Vector{Float64})
    all(theta .== 0) && return 0.0
    return -norm(ATR'*pi_t .- theta)^2
end

function compute_utilities(pi_t, pi_t1, Mq, Gamma, theta, Q=Q_TRANSITION)
    return (M=U_M(pi_t,pi_t1,Q), S=U_S(pi_t,pi_t1),
            Q=U_Q(pi_t1,Mq),    W=U_W(pi_t1,Gamma), E=U_E(pi_t,theta))
end

function compute_reward(pi_t, pi_t1, Mq, Gamma, theta,
                         alpha=ALPHA_VEC, Q=Q_TRANSITION)
    u = compute_utilities(pi_t, pi_t1, Mq, Gamma, theta, Q)
    return alpha[1]*u.M + alpha[2]*u.S + alpha[3]*u.Q + alpha[4]*u.W + alpha[5]*u.E
end

end # module UtilityFunctions
