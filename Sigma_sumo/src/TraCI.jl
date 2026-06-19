"""
TraCI.jl - Julia TraCI (Traffic Control Interface) client for SUMO.

Implements the minimal TraCI binary protocol needed by SIGMA:
  - TCP connection management
  - Simulation control (step, close)
  - Lane/edge queue length retrieval
  - Vehicle waiting-time queries
  - Traffic-light phase setting
  - Emergency vehicle detection

Protocol reference: https://sumo.dlr.de/docs/TraCI/Protocol.html

Message format:
  [total_length:4B][cmd_length:1B][cmd_id:1B][payload...]
  Response: [total_length:4B][status_length:1B][0x00][result_type:1B][cmd_id:1B][ok:1B]...
"""
module TraCI

using Sockets, Printf

# ─── TraCI command IDs ────────────────────────────────────────────────────────
const CMD_SIMSTEP        = 0x02
const CMD_CLOSE          = 0x7F
const CMD_GET_LANE_VAR   = 0xa3
const CMD_GET_TL_VAR     = 0xa2
const CMD_SET_TL_VAR     = 0xc2
const CMD_GET_VEH_VAR    = 0xa4
const CMD_GET_EDGE_VAR   = 0xaa

const VAR_LAST_STEP_VEHICLE_NUMBER = 0x10
const VAR_LAST_STEP_HALT_NUMBER    = 0x12
const VAR_LAST_STEP_MEAN_SPEED     = 0x11
const VAR_WAITING_TIME             = 0x7a
const VAR_PHASE_DURATION           = 0x29
const VAR_RED_YELLOW_GREEN_STATE   = 0x20
const VAR_PHASE                    = 0x28
const VAR_CONTROLLED_LANES         = 0x26
const VAR_CURRENT_PHASE            = 0x28
const VAR_SPEED                    = 0x40
const VAR_VTYPE                    = 0x4f

# ─── Connection struct ────────────────────────────────────────────────────────
mutable struct TraCIConnection
    sock    ::TCPSocket
    host    ::String
    port    ::Int
    connected::Bool
    sim_time ::Float64
end

function connect(host::String, port::Int; timeout::Float64=30.0)
    t0 = time()
    sock = nothing
    while time() - t0 < timeout
        try
            sock = Sockets.connect(host, port)
            break
        catch
            sleep(0.5)
        end
    end
    if sock === nothing
        error("Could not connect to SUMO TraCI at $host:$port within $(timeout)s")
    end
    conn = TraCIConnection(sock, host, port, true, 0.0)
    _read_version(conn)
    return conn
end

function close_connection(conn::TraCIConnection)
    if conn.connected
        _send_cmd(conn, CMD_CLOSE, UInt8[])
        close(conn.sock)
        conn.connected = false
    end
end

# ─── Low-level I/O ────────────────────────────────────────────────────────────
function _write_int32(buf::IOBuffer, v::Int32)
    write(buf, hton(v))
end

function _write_uint8(buf::IOBuffer, v::UInt8)
    write(buf, v)
end

function _write_string(buf::IOBuffer, s::String)
    b = Vector{UInt8}(s)
    write(buf, hton(Int32(length(b))))
    write(buf, b)
end

function _write_double(buf::IOBuffer, v::Float64)
    write(buf, hton(reinterpret(UInt64, v)))
end

function _send_cmd(conn::TraCIConnection, cmd_id::UInt8, payload::Vector{UInt8})
    buf = IOBuffer()
    cmd_len = 1 + length(payload)   # cmd_id byte + payload
    if cmd_len <= 255
        write(buf, UInt8(cmd_len))
        write(buf, UInt8(cmd_id))
    else
        write(buf, UInt8(0))
        _write_int32(buf, Int32(cmd_len + 4))
        write(buf, UInt8(cmd_id))
    end
    write(buf, payload)
    msg = take!(buf)

    # Prepend total message length (4 bytes)
    total = IOBuffer()
    _write_int32(total, Int32(length(msg) + 4))
    write(total, msg)
    write(conn.sock, take!(total))
end

function _read_bytes(conn::TraCIConnection, n::Int)
    buf = Vector{UInt8}(undef, n)
    offset = 1
    while offset <= n
        got = readbytes!(conn.sock, @view(buf[offset:end]), n - offset + 1)
        got == 0 && error("TraCI connection closed unexpectedly")
        offset += got
    end
    return buf
end

function _read_response(conn::TraCIConnection)
    # Read total length
    len_bytes = _read_bytes(conn, 4)
    total_len = ntoh(reinterpret(Int32, len_bytes)[1])
    remaining = total_len - 4
    remaining <= 0 && return nothing, nothing

    data = _read_bytes(conn, remaining)
    return data, total_len
end

function _read_version(conn::TraCIConnection)
    data, _ = _read_response(conn)
    data === nothing && return
    # Parse version response (cmd 0x00) — just consume it
end

# ─── Simulation step ─────────────────────────────────────────────────────────
function simulation_step!(conn::TraCIConnection, target_time::Float64=0.0)
    buf = IOBuffer()
    _write_double(buf, target_time)
    _send_cmd(conn, UInt8(CMD_SIMSTEP), take!(buf))
    _read_response(conn)
    conn.sim_time = target_time > 0 ? target_time : conn.sim_time + 1.0
end

# ─── Lane variable getters ────────────────────────────────────────────────────
function _build_var_request(var_id::UInt8, object_id::String)
    buf = IOBuffer()
    write(buf, var_id)
    _write_string(buf, object_id)
    return take!(buf)
end

function _parse_int_response(data::Vector{UInt8})
    # Response layout: [cmd_len][0xb3 response cmd][var_id][type=0x09][int32]
    pos = 1
    while pos <= length(data) - 8
        if data[pos+1] == 0x09 || data[pos] == 0x09   # TYPE_INTEGER
            v = ntoh(reinterpret(Int32, data[pos+1:pos+4])[1])
            return Int(v)
        end
        pos += 1
    end
    # Fallback: scan for int32 pattern
    for i in max(1,length(data)-7):length(data)-3
        v = ntoh(reinterpret(Int32, data[i:i+3])[1])
        if 0 <= v <= 10000
            return Int(v)
        end
    end
    return 0
end

function _parse_double_response(data::Vector{UInt8})
    for i in max(1,length(data)-11):length(data)-7
        v = ntoh(reinterpret(UInt64, data[i:i+7])[1])
        f = reinterpret(Float64, v)
        if isfinite(f) && 0.0 <= f <= 1e6
            return f
        end
    end
    return 0.0
end

function get_lane_halt_number(conn::TraCIConnection, lane_id::String)::Int
    payload = _build_var_request(UInt8(VAR_LAST_STEP_HALT_NUMBER), lane_id)
    _send_cmd(conn, UInt8(CMD_GET_LANE_VAR), payload)
    data, _ = _read_response(conn)
    data === nothing && return 0
    return _parse_int_response(data)
end

function get_lane_vehicle_count(conn::TraCIConnection, lane_id::String)::Int
    payload = _build_var_request(UInt8(VAR_LAST_STEP_VEHICLE_NUMBER), lane_id)
    _send_cmd(conn, UInt8(CMD_GET_LANE_VAR), payload)
    data, _ = _read_response(conn)
    data === nothing && return 0
    return _parse_int_response(data)
end

function get_lane_waiting_time(conn::TraCIConnection, lane_id::String)::Float64
    payload = _build_var_request(UInt8(VAR_WAITING_TIME), lane_id)
    _send_cmd(conn, UInt8(CMD_GET_LANE_VAR), payload)
    data, _ = _read_response(conn)
    data === nothing && return 0.0
    return _parse_double_response(data)
end

# ─── Traffic light control ────────────────────────────────────────────────────
function get_tl_phase(conn::TraCIConnection, tl_id::String)::Int
    payload = _build_var_request(UInt8(VAR_CURRENT_PHASE), tl_id)
    _send_cmd(conn, UInt8(CMD_GET_TL_VAR), payload)
    data, _ = _read_response(conn)
    data === nothing && return 0
    return _parse_int_response(data)
end

function get_tl_state(conn::TraCIConnection, tl_id::String)::String
    payload = _build_var_request(UInt8(VAR_RED_YELLOW_GREEN_STATE), tl_id)
    _send_cmd(conn, UInt8(CMD_GET_TL_VAR), payload)
    data, _ = _read_response(conn)
    data === nothing && return "rrrrrrrr"
    # Parse string response
    try
        # Find string type marker 0x0c
        for i in 1:length(data)-4
            if data[i] == 0x0c
                slen = Int(ntoh(reinterpret(Int32, data[i+1:i+4])[1]))
                if slen > 0 && i+4+slen <= length(data)
                    return String(data[i+5:i+4+slen])
                end
            end
        end
    catch; end
    return "rrrrrrrr"
end

function set_tl_phase!(conn::TraCIConnection, tl_id::String, phase_idx::Int)
    buf = IOBuffer()
    write(buf, UInt8(VAR_PHASE))
    _write_string(buf, tl_id)
    write(buf, UInt8(0x09))   # TYPE_INTEGER
    _write_int32(buf, Int32(phase_idx))
    _send_cmd(conn, UInt8(CMD_SET_TL_VAR), take!(buf))
    _read_response(conn)
end

function set_tl_state!(conn::TraCIConnection, tl_id::String, state::String)
    buf = IOBuffer()
    write(buf, UInt8(VAR_RED_YELLOW_GREEN_STATE))
    _write_string(buf, tl_id)
    write(buf, UInt8(0x0c))   # TYPE_STRING
    _write_string(buf, state)
    _send_cmd(conn, UInt8(CMD_SET_TL_VAR), take!(buf))
    _read_response(conn)
end

# ─── Vehicle queries ──────────────────────────────────────────────────────────
function get_vehicle_type(conn::TraCIConnection, veh_id::String)::String
    payload = _build_var_request(UInt8(VAR_VTYPE), veh_id)
    _send_cmd(conn, UInt8(CMD_GET_VEH_VAR), payload)
    data, _ = _read_response(conn)
    data === nothing && return ""
    try
        for i in 1:length(data)-4
            if data[i] == 0x0c
                slen = Int(ntoh(reinterpret(Int32, data[i+1:i+4])[1]))
                if slen > 0 && i+4+slen <= length(data)
                    return String(data[i+5:i+4+slen])
                end
            end
        end
    catch; end
    return ""
end

function get_vehicle_waiting_time(conn::TraCIConnection, veh_id::String)::Float64
    payload = _build_var_request(UInt8(VAR_WAITING_TIME), veh_id)
    _send_cmd(conn, UInt8(CMD_GET_VEH_VAR), payload)
    data, _ = _read_response(conn)
    data === nothing && return 0.0
    return _parse_double_response(data)
end

end # module TraCI
