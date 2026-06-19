"""
Pretraining.jl - Offline actor/critic pretraining on synthetic data before SUMO.

Generates synthetic Poisson traffic data (identical to Sigma_manual pipeline),
applies C4 augmentation, then pretrain actor (CE + utility loss) and critic (FQI).
The resulting pretrained weights are used as the initial policy for SUMO online training.
"""
module Pretraining

using Random, LinearAlgebra, Statistics, Printf, Distributions
using ..Constants
using ..Networks
using ..UtilityFunctions

# ─── Synthetic data ───────────────────────────────────────────────────────────
struct Transition
    state        ::Vector{Float64}  # 52-dim
    action_oh    ::Vector{Float64}  # 8-dim
    reward       ::Float64
    rewards_vec  ::Vector{Float64}  # 5-dim
    prev_phase_16::Vector{Float64}  # 16-dim
    Mq           ::Vector{Float64}  # 16-dim
    l_in         ::Vector{Float64}  # 4-dim
    gamma        ::Vector{Float64}  # 4-dim
    theta        ::Vector{Float64}  # 16-dim
end

function arrival_rate(t)
    ν = NU0
    for r in 1:R_HARM; ν += KR[r]*sin(WR_BASE[r]*t+PHI_R[r]); end
    return max(ν, 0.0)
end

function departure_rate(t)
    μ = MU0
    for r in 1:R_HARM; μ += KAPPA_R[r]*sin(WR_BASE[r]*t+PSI_R[r]); end
    return max(μ, 0.0)
end

function build_pressure_mask(l_in, l_out)
    Mq_in = vcat([fill(l_in[i],4) for i in 1:4]...)
    Mq_out = vcat([l_out for _ in 1:4]...)
    mp = Mq_in .- Mq_out
    return mp .- minimum(mp)
end

function encode_emergency(origin::Int, dest::Int)
    θ = zeros(16)
    if origin>0 && dest>0; θ[(origin-1)*4+dest] = 1.0
    elseif origin>0; for j in 1:4; θ[(origin-1)*4+j]=1.0; end; end
    return θ
end

# ─── Rotation operators ───────────────────────────────────────────────────────
rotate4(v,k)   = k==0 ? copy(v) : (RHO_PRIME^mod(k,4))*v
rotate16(v,k)  = k==0 ? copy(v) : (RHO16^mod(k,4))*v
rotate_a(v,k)  = k==0 ? copy(v) : (RHO_A^mod(k,4))*v

function augment_transition(tr::Transition)
    aug = Transition[]
    for k in 0:3
        ra  = rotate_a(tr.action_oh, k)
        rp  = rotate16(tr.prev_phase_16, k)
        rl  = rotate4(tr.l_in, k)
        rg  = rotate4(tr.gamma, k)
        rt  = rotate16(tr.theta, k)
        rMq = rotate16(tr.Mq, k); rMq .-= minimum(rMq)
        rs  = vcat(rp, rMq, rg, rt)
        push!(aug, Transition(rs, ra, tr.reward, tr.rewards_vec, rp, rMq, rl, rg, rt))
    end
    return aug
end

# ─── Dataset generation ───────────────────────────────────────────────────────
function generate_dataset(;N_episodes=300, T_steps=120, seed=42, augment=true)
    rng = MersenneTwister(seed)
    D = Transition[]
    for ep in 1:N_episodes
        l_in  = zeros(4); l_out = zeros(4); Gamma = zeros(4)
        prev_oh = zeros(N_PHASES); prev_oh[1]=1.0
        t = 0.0

        has_emg = rand(rng) < PE_EMERGENCY
        origin = has_emg ? rand(rng,1:4) : 0
        dest   = has_emg ? rand(rng, setdiff(1:4,[origin])) : 0
        theta  = has_emg ? encode_emergency(origin,dest) : zeros(16)

        for _ in 1:T_steps
            ν = arrival_rate(t); μ = departure_rate(t)
            for i in 1:4
                l_in[i] = min(l_in[i]+rand(rng,Poisson(ν*DT)), Float64(Q_MAX))
            end

            Mq    = build_pressure_mask(l_in, l_out)
            Gamma = l_in .* DT
            prev16 = vec(prev_oh'*ATR)
            s  = vcat(prev16, Mq, Gamma, theta)

            α_rand = rand(rng,5); α_rand ./= sum(α_rand); α_rand .*= 5.0

            rvec = zeros(N_PHASES)
            for a in 1:N_PHASES
                pi_a = zeros(N_PHASES); pi_a[a]=1.0
                u = compute_utilities(pi_a,pi_a,Mq,Gamma,theta)
                rvec[a] = α_rand[1]*u.M+α_rand[2]*u.S+α_rand[3]*u.Q+α_rand[4]*u.W+α_rand[5]*u.E
            end
            a_star = argmax(rvec)
            a_oh   = zeros(N_PHASES); a_oh[a_star]=1.0
            pi_a   = copy(a_oh)
            u      = compute_utilities(pi_a,pi_a,Mq,Gamma,theta)
            rv5    = [u.M,u.S,u.Q,u.W,u.E]

            push!(D, Transition(s,a_oh,rvec[a_star],rv5,prev16,Mq,copy(l_in),Gamma,theta))

            # Update queues (simplified: served on phase)
            for j in 1:4
                if any(ATR[a_star,(i-1)*4+j]>0 for i in 1:4)
                    dep = rand(rng, Poisson(μ*DT))
                    l_out[j] = min(l_out[j]+dep, Float64(Q_MAX))
                end
            end
            t += DT; prev_oh = a_oh
        end
    end
    if augment
        DF = Transition[]
        for tr in D; append!(DF, augment_transition(tr)); end
        return DF, D
    end
    return D, D
end

# ─── Actor pretraining ────────────────────────────────────────────────────────
function utility_grad(pi, tr::Transition, λ=LAMBDA_VEC; eps=1e-4)
    grad = zeros(N_PHASES)
    for k in 1:N_PHASES
        pip = copy(pi); pip[k]+=eps; pip = max.(pip,1e-9); pip ./= sum(pip)
        pim = copy(pi); pim[k]-=eps; pim = max.(pim,1e-9); pim ./= sum(pim)
        u_p = compute_utilities(pip,pip,tr.Mq,tr.gamma,tr.theta)
        u_m = compute_utilities(pim,pim,tr.Mq,tr.gamma,tr.theta)
        lp = λ[1]*(-u_p.M)+λ[2]*(-u_p.S)+λ[3]*(-u_p.Q)+λ[4]*(-u_p.W)+λ[5]*(-u_p.E)
        lm = λ[1]*(-u_m.M)+λ[2]*(-u_m.S)+λ[3]*(-u_m.Q)+λ[4]*(-u_m.W)+λ[5]*(-u_m.E)
        grad[k] = (lp-lm)/(2*eps)
    end
    return grad
end

function pretrain_actor!(actor::ActorNetwork, DF::Vector{Transition};
                          epochs=ACTOR_EPOCHS, batch_size=ACTOR_BATCH,
                          rng=MersenneTwister(0), verbose=true)
    N = length(DF)
    for epoch in 1:epochs
        idx = randperm(rng, N)
        total_loss = 0.0
        for start in 1:batch_size:N
            bidx = idx[start:min(start+batch_size-1,N)]
            for i in bidx
                tr = DF[i]
                pi, cache = actor_forward(actor, tr.state)
                ce_grad   = pi .- tr.action_oh
                u_grad    = utility_grad(pi, tr)
                combined  = ce_grad .+ u_grad
                actor_backward!(actor, cache, pi, tr.action_oh, combined.-(pi.-tr.action_oh))
                total_loss -= sum(tr.action_oh .* log.(max.(pi,1e-9)))
            end
        end
        if verbose && (epoch%20==0 || epoch==1)
            @printf("[Actor] Epoch %3d/%d  CE-loss=%.4f\n", epoch, epochs, total_loss/N)
        end
    end
end

# ─── Critic pretraining (FQI) ─────────────────────────────────────────────────
function pretrain_critic!(critic::CriticNetwork, DF::Vector{Transition};
                           max_iter=FQI_MAX_ITER, γ=GAMMA, α=ALPHA_VEC,
                           rng=MersenneTwister(0), verbose=true)
    N = length(DF)
    states  = hcat([tr.state for tr in DF]...)
    actions = [argmax(tr.action_oh) for tr in DF]

    # k-NN next states
    next_states = Vector{Vector{Float64}}(undef, N)
    for i in 1:N
        ai = actions[i]
        same = findall(j->actions[j]==ai && j!=i, 1:N)
        if isempty(same); next_states[i]=DF[mod1(i+1,N)].state; continue; end
        dists = [norm(states[:,i].-states[:,j]) for j in same]
        kk = min(KNN_K, length(dists))
        nn  = same[sortperm(dists)[1:kk]]
        avg = zeros(STATE_DIM)
        for j in nn; avg .+= DF[mod1(j+1,N)].state; end
        next_states[i] = avg./kk
    end

    Q_vals = [sum(α.*tr.rewards_vec) for tr in DF]
    ctarget = copy_critic(critic)

    for iter in 1:max_iter
        Q_prev = copy(Q_vals)
        for i in 1:N
            _, Qn = critic_best_action(ctarget, next_states[i])
            Q_vals[i] = sum(α.*DF[i].rewards_vec) + γ*Qn
        end
        for i in randperm(rng, N)
            critic_td_update!(critic, DF[i].state, DF[i].action_oh, Q_vals[i])
        end
        Δ = norm(Q_vals.-Q_prev)
        verbose && (iter%10==0||iter==1) && @printf("[Critic] Iter %3d/%d  ΔQ=%.6f\n",iter,max_iter,Δ)
        soft_update!(ctarget, critic, 0.1)
        Δ < FQI_EPS && (verbose && println("[Critic] Converged at iter $iter"); break)
    end
end

end # module Pretraining
