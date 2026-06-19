"""
Pretraining.jl - Offline actor and critic pretraining (Appendix A.1, A.2).

Actor pretraining:
  Loss = L_en + λ⊤·L_R
  L_en = -mean(a⊤ log π(s))    (cross-entropy)
  L_R  = [L_M, L_S, L_Q, L_W, L_E]  (utility losses)

Critic pretraining via fitted Q-iteration (FQI):
  Next states approximated by k-NN averaging over same-action transitions.
  Q values iterated via Bellman until convergence.
"""
module Pretraining

using Random, LinearAlgebra, Statistics, Printf
using ..Constants
using ..Networks
using ..Environment
using ..Augmentation

# ─────────────────────────────────────────────────────────────────────────────
# Utility loss computation for a batch (mean over transitions)
# ─────────────────────────────────────────────────────────────────────────────
function batch_utility_losses(
    actor  ::ActorNetwork,
    batch  ::Vector{Augmentation.Transition}
)
    lM = lS = lQ = lW = lE = 0.0
    n = length(batch)

    for tr in batch
        pi_t, _ = actor_forward(actor, tr.state)

        # Extract Mq, Gamma, theta from state vector
        Mq    = tr.Mq
        Gamma = tr.gamma
        theta = tr.theta

        # For utility computation we need π(t) and π(t+1).
        # During pretraining we use π(t)=π(t+1)=current actor output (self-consistency).
        u = compute_utilities(pi_t, pi_t, Mq, Gamma, theta)
        lM += -u.M
        lS += -u.S
        lQ += -u.Q
        lW += -u.W
        lE += -u.E
    end
    return lM/n, lS/n, lQ/n, lW/n, lE/n
end

# ─────────────────────────────────────────────────────────────────────────────
# Gradient of utility losses w.r.t. actor output z4 (logit-space gradient)
# We compute a numerical approximation of the utility gradient through softmax.
# ─────────────────────────────────────────────────────────────────────────────
function utility_grad_logits(
    pi    ::Vector{Float64},
    tr    ::Augmentation.Transition,
    λ     ::Vector{Float64} = LAMBDA_VEC;
    eps   ::Float64 = 1e-4
)
    Mq    = tr.Mq
    Gamma = tr.gamma
    theta = tr.theta

    # Compute gradient of total utility w.r.t. softmax distribution pi
    # via finite differences on each logit dimension
    grad = zeros(Float64, N_PHASES)
    for k in 1:N_PHASES
        pi_p = copy(pi); pi_p[k] += eps; pi_p ./= sum(pi_p)
        pi_m = copy(pi); pi_m[k] -= eps; pi_m = max.(pi_m, 1e-9); pi_m ./= sum(pi_m)

        u_p = compute_utilities(pi_p, pi_p, Mq, Gamma, theta)
        u_m = compute_utilities(pi_m, pi_m, Mq, Gamma, theta)

        total_p = λ[1]*(-u_p.M) + λ[2]*(-u_p.S) + λ[3]*(-u_p.Q) + λ[4]*(-u_p.W) + λ[5]*(-u_p.E)
        total_m = λ[1]*(-u_m.M) + λ[2]*(-u_m.S) + λ[3]*(-u_m.Q) + λ[4]*(-u_m.W) + λ[5]*(-u_m.E)

        grad[k] = (total_p - total_m) / (2*eps)
    end
    return grad
end

# ─────────────────────────────────────────────────────────────────────────────
# Actor pretraining loop
# ─────────────────────────────────────────────────────────────────────────────
function pretrain_actor!(
    actor  ::ActorNetwork,
    DF     ::Vector{Augmentation.Transition};
    epochs ::Int = ACTOR_EPOCHS,
    batch_size::Int = ACTOR_BATCH,
    rng    ::AbstractRNG = MersenneTwister(0),
    verbose::Bool = true
)
    N = length(DF)
    for epoch in 1:epochs
        indices = randperm(rng, N)
        total_loss = 0.0
        n_batches  = 0

        for start in 1:batch_size:N
            batch_idx = indices[start : min(start + batch_size - 1, N)]
            batch = DF[batch_idx]

            for tr in batch
                pi, cache = actor_forward(actor, tr.state)

                # Cross-entropy gradient w.r.t. logits: dCE/dz4 = π - a
                ce_grad = pi .- tr.action_oh

                # Utility regularisation gradient
                u_grad = utility_grad_logits(pi, tr, LAMBDA_VEC)

                combined_grad = ce_grad .+ u_grad
                actor_backward!(actor, cache, pi, tr.action_oh, combined_grad .- (pi .- tr.action_oh))

                # Track CE loss
                log_prob = sum(tr.action_oh .* log.(max.(pi, 1e-9)))
                total_loss -= log_prob
            end
            n_batches += 1
        end

        if verbose && (epoch % 20 == 0 || epoch == 1)
            avg_loss = total_loss / N
            @printf("[Actor Pretraining] Epoch %3d/%d  CE-loss = %.4f\n", epoch, epochs, avg_loss)
        end
    end
    return actor
end

# ─────────────────────────────────────────────────────────────────────────────
# k-NN next-state approximation for critic pretraining
# For each transition (s_i, a_i), find k nearest neighbours in D that share
# the same action, and average their successor states (= next element in sequence).
# ─────────────────────────────────────────────────────────────────────────────
function build_knn_next_states(
    DF ::Vector{Augmentation.Transition};
    k  ::Int = KNN_K
)
    N = length(DF)
    states = hcat([tr.state for tr in DF]...)  # STATE_DIM × N
    actions = [argmax(tr.action_oh) for tr in DF]

    next_states = Vector{Vector{Float64}}(undef, N)

    for i in 1:N
        ai = actions[i]
        # Find transitions with same action
        same_action_idx = findall(j -> actions[j] == ai && j != i, 1:N)
        if isempty(same_action_idx)
            next_states[i] = DF[mod1(i+1, N)].state
            continue
        end

        # Compute L2 distances
        dists = [norm(states[:, i] .- states[:, j]) for j in same_action_idx]
        kk    = min(k, length(dists))
        sorted = sortperm(dists)[1:kk]
        nn_idx = same_action_idx[sorted]

        # Average successor states (circular wrap)
        avg = zeros(Float64, STATE_DIM)
        for j in nn_idx
            avg .+= DF[mod1(j+1, N)].state
        end
        next_states[i] = avg ./ kk
    end
    return next_states
end

# ─────────────────────────────────────────────────────────────────────────────
# Fitted Q-Iteration (FQI) for critic pretraining
# ─────────────────────────────────────────────────────────────────────────────
function pretrain_critic!(
    critic ::CriticNetwork,
    DF     ::Vector{Augmentation.Transition};
    next_states::Union{Vector{Vector{Float64}}, Nothing} = nothing,
    max_iter::Int = FQI_MAX_ITER,
    γ       ::Float64 = GAMMA,
    ε_conv  ::Float64 = FQI_EPS,
    α       ::Vector{Float64} = ALPHA_VEC,
    rng     ::AbstractRNG = MersenneTwister(0),
    verbose ::Bool = true
)
    N = length(DF)

    # Compute next states via k-NN if not provided
    if next_states === nothing
        verbose && println("[Critic Pretraining] Computing k-NN next states...")
        next_states = build_knn_next_states(DF)
    end

    # Initial Q values: Q^(0) = r_i
    Q_vals = [sum(α .* tr.rewards_vec) for tr in DF]

    critic_target = copy_critic(critic)

    for iter in 1:max_iter
        Q_prev = copy(Q_vals)

        # Update Q values via Bellman
        for i in 1:N
            s_next = next_states[i]
            _, best_Q_next = critic_best_action(critic_target, s_next)
            Q_vals[i] = sum(α .* DF[i].rewards_vec) + γ * best_Q_next
        end

        # Train critic network on (s_i, a_i, Q_i) dataset
        indices = randperm(rng, N)
        for idx in indices
            tr = DF[idx]
            critic_td_update!(critic, tr.state, tr.action_oh, Q_vals[idx])
        end

        # Check convergence
        delta = norm(Q_vals .- Q_prev)
        if verbose && (iter % 10 == 0 || iter == 1)
            @printf("[Critic Pretraining] Iter %3d/%d  ΔQ = %.6f\n", iter, max_iter, delta)
        end

        # Soft-update target critic
        soft_update!(critic_target, critic, 0.1)

        if delta < ε_conv
            verbose && println("[Critic Pretraining] Converged at iter $iter")
            break
        end
    end
    return critic
end

end # module
