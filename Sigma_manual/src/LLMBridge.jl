"""
LLMBridge.jl
Julia client for the Python TF bridge server (tf_bridge/server.py).

Provides:
  - encode_emergency(instruction) → Vector{Float64} (16-dim θ)
  - actor_forward(state)          → Vector{Float64} (8-dim π)
  - critic_forward(state, action) → Float64
  - push_transition!(s,a,r,s')   → Int (buffer size)
  - train_step!(;batch_size)      → NamedTuple(actor_loss, critic_loss)
  - save_model(path)
  - load_model(path)
  - ping()                        → Bool

Start the server first:
  python tf_bridge/server.py [--use-llm]

Then in Julia:
  using .LLMBridge
  bridge = TFBridge()
  θ = encode_emergency(bridge, "Ambulance from North toward East")
"""
module LLMBridge

using Sockets, JSON3

export TFBridge, encode_emergency, actor_forward, critic_forward,
       push_transition!, train_step!, save_model, load_model, ping,
       pretrain_actor_batch!, pretrain_critic_batch!

# ─────────────────────────────────────────────────────────────────────────────
mutable struct TFBridge
    host::String
    port::Int
    sock::Union{TCPSocket, Nothing}
end

function TFBridge(; host="127.0.0.1", port=9876, autoconnect=true)
    b = TFBridge(host, port, nothing)
    autoconnect && connect!(b)
    return b
end

function connect!(b::TFBridge)
    b.sock = connect(b.host, b.port)
end

function _send(b::TFBridge, cmd::String, args::Dict)
    msg = JSON3.write(Dict("cmd" => cmd, "args" => args)) * "\n"
    write(b.sock, msg)
    resp_line = readline(b.sock)
    resp = JSON3.read(resp_line)
    if !resp.ok
        error("[TFBridge] Server error: $(resp.error)")
    end
    return resp.result
end

# ─── API ──────────────────────────────────────────────────────────────────────

"""Encode a natural language emergency instruction into a 16-dim θ vector."""
function encode_emergency(b::TFBridge, instruction::String)::Vector{Float64}
    result = _send(b, "encode_emergency", Dict("instruction" => instruction))
    return Float64.(result)
end

"""Run actor forward pass; returns 8-dim softmax distribution."""
function actor_forward(b::TFBridge, state::Vector{Float64})::Vector{Float64}
    result = _send(b, "actor_forward", Dict("state" => state))
    return Float64.(result)
end

"""Run critic forward pass; returns scalar Q-value."""
function critic_forward(b::TFBridge, state::Vector{Float64},
                         action::Vector{Float64})::Float64
    result = _send(b, "critic_forward",
                   Dict("state" => state, "action" => action))
    return Float64(result)
end

"""Push a (s, a, r, s') transition to the Python replay buffer."""
function push_transition!(b::TFBridge, s, a, r, s_next)::Int
    result = _send(b, "push_transition",
                   Dict("state"      => collect(Float64, s),
                        "action"     => collect(Float64, a),
                        "reward"     => Float64(r),
                        "next_state" => collect(Float64, s_next)))
    return Int(result)
end

"""Run one actor+critic TD update on a sampled batch."""
function train_step!(b::TFBridge; batch_size::Int=64)
    result = _send(b, "train_step", Dict("batch_size" => batch_size))
    return (actor_loss=Float64(result.actor_loss),
            critic_loss=Float64(result.critic_loss))
end

"""Send a batch for actor pretraining (CE + utility loss)."""
function pretrain_actor_batch!(b::TFBridge,
                                states, actions, Mqs, gammas, thetas)::Float64
    result = _send(b, "pretrain_actor_batch",
                   Dict("states"  => [collect(Float64, s) for s in states],
                        "actions" => [collect(Float64, a) for a in actions],
                        "Mqs"     => [collect(Float64, m) for m in Mqs],
                        "gammas"  => [collect(Float64, g) for g in gammas],
                        "thetas"  => [collect(Float64, t) for t in thetas]))
    return Float64(result)
end

"""Send a batch for critic pretraining (regression on Q targets)."""
function pretrain_critic_batch!(b::TFBridge,
                                 states, actions, q_targets)::Float64
    result = _send(b, "pretrain_critic_batch",
                   Dict("states"    => [collect(Float64, s) for s in states],
                        "actions"   => [collect(Float64, a) for a in actions],
                        "q_targets" => collect(Float64, q_targets)))
    return Float64(result)
end

"""Save actor+critic weights to disk."""
function save_model(b::TFBridge, path::String)
    _send(b, "save_model", Dict("path" => path))
end

"""Load actor+critic weights from disk."""
function load_model(b::TFBridge, path::String)
    _send(b, "load_model", Dict("path" => path))
end

"""Check connectivity — returns true if server responds."""
function ping(b::TFBridge)::Bool
    try
        result = _send(b, "ping", Dict())
        return result == "pong"
    catch
        return false
    end
end

end # module LLMBridge
