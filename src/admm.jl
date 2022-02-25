# TODO: Bad idea to store AAᵀ unless m ≪ n²
struct ProblemData{T}
    c::Vector{T}
    A::Matrix{T}
    AAT::Matrix{T}
    b::Vector{T}
    m::Int
    n::Int

    function ProblemData(c::Vector{T}, A::Matrix{T}, b::Vector{T}) where {T <: Real}
        m, n = size(A)
        (m != length(b) || n != length(c)) && error(DimensionMismatch("Invalid dimensions"))
        return new{T}(c, A, A*A', b, m, n)
    end
end


mutable struct SDP{T}
    data::ProblemData{T}
    obj_val::T
    xk::Vector{T}       # primal var
    zk::Vector{T}       # primal var
    uk::Vector{T}       # dual var
    rp::Vector{T}       # primal residual
    rd::Vector{T}       # dual residual
    ρ::T                # ADMM param 
    α::T                # over-relaxation param
    cache
    function SDP(c::Vector{T}, A::Matrix{T}, b::Vector{T}; ρ=1.0, α=1.5) where {T <: Real}
        m, n = size(A)
        data = ProblemData(c, A, b)
        return new{T}(
            data,
            zero(T),
            zeros(T, size(c)),
            zeros(T, size(c)),
            zeros(T, size(c)),
            fill(Inf, size(b)),
            fill(Inf, size(c)),
            T(ρ),
            T(α)
        )
    end
end

struct Log{T}
    iter_time::Union{AbstractVector{T}, Nothing}
    rp::Union{AbstractVector{T}, Nothing}
    rd::Union{AbstractVector{T}, Nothing}
    setup_time::T
    precond_time::T
    solve_time::T
end

struct Result{T}
    obj_val::T
    x::Vector{T}
    log::Log{T}
end

function update_x!(sdp::SDP, solver::S, P) where {T <: Real, S <: Union{CgSolver, Nothing}}
    d = -sdp.data.c + sdp.ρ * (sdp.zk - sdp.uk)
    ν = sdp.data.AAT \ (sdp.data.A*d - sdp.ρ*sdp.data.b)
    sdp.xk .= d .- sdp.data.A'*ν
    return nothing
end

function update_z!(sdp::SDP; relax=true, xhat=nothing)
    if relax
        xhat = sdp.xk
    end
    d, V = eigen(unvec_symm(xhat + sdp.uk), sortby=x->-x)
    nn = count(>(0), d)
    sdp.zk .= vec_symm(V[:, 1:nn] * Diagonal(d[1:nn]) * V[:, 1:nn]')
    return nothing
end

function update_u!(sdp::SDP{T}; relax=true, xhat=nothing) where {T <: Real}
    if relax
        xhat = sdp.xk
    end
    @. sdp.uk += xhat - sdp.zk
    return nothing
end

function update_rho!(sdp::SDP, rp, rd, μ, τ_inc, τ_dec)
    if rp > μ * rd
        sdp.ρ = τ_inc * sdp.ρ
        return sdp.ρ
    elseif rd > μ * rp
        sdp.ρ = sdp.ρ / τ_dec
        return sdp.ρ
    else
        return sdp.ρ
    end
end

function converged(sdp::SDP, eps_abs, eps_rel)
    primal = norm(sdp.rp) ≤ sqrt(sdp.data.m) * eps_abs + eps_rel * max(norm(sdp.xk), norm(sdp.uk), norm(sdp.data.c))
    dual = norm(sdp.rd) ≤ sqrt(sdp.data.n) * eps_abs + sdp.ρ * norm(sdp.uk) * eps_rel
    return primal && dual
end

function solve!(
    sdp::SDP{T};
    relax::Bool=true,
    logging::Bool=false,
    indirect::Bool=false,
    precond::Bool=false,
    eps_abs=1e-5,
    eps_rel=1e-3,
    eps_inf=1e-8,
    max_iters::Int=100,
    print_iter::Int=25,
    cache=nothing
) where {T <: Real}
    setup_time_start = time_ns()
    @printf("Starting setup...")

    # --- parameters ---
    n = sdp.data.n
    m = sdp.data.m
    t = 1
    μ = 10
    τ_inc = 2
    τ_dec = 2
    ρ = sdp.ρ
    α = sdp.α

    # --- enable multithreaded BLAS ---
    BLAS.set_num_threads(Sys.CPU_THREADS)

    # --- allocate memory ---
    if relax
        xhat = copy(sdp.xk)
    end
    if isnothing(cache)
        cache = (
            uk_old = zeros(size(sdp.uk)),
        )
    end

    # --- Precondition ---
    # TODO:

    # --- setup log ---
    # TODO: 

    setup_time = (time_ns() - setup_time_start) / 1e9
    @printf("\nSetup in %6.3fs\n", setup_time)

    # --- Print Headers ---
    headers = ["Iteration", "Objective", "Primal Res", "Dual Res", "ρ", "Time"]
    print_header(headers)
    
    # --------------------------------------------------------------------------
    # --------------------- ITERATIONS -----------------------------------------
    # --------------------------------------------------------------------------
    solve_time_start = time_ns()
    while t <= max_iters && !converged(sdp, eps_abs, eps_rel)
        # --- Update Iterates ---
        # TODO: define solver
        update_x!(sdp, nothing, nothing)
        if relax
            @. xhat = α * sdp.xk + (1-α) * sdp.zk
        end
        update_z!(sdp; relax=relax, xhat=xhat)
        cache.uk_old .= sdp.uk
        update_u!(sdp; relax=relax, xhat=xhat)

        # --- Update ρ ---
        rp = norm(sdp.xk - sdp.zk)
        rd = norm(ρ*(sdp.uk - cache.uk_old))
        ρ = update_rho!(sdp, rp, rd, μ, τ_inc, τ_dec)

        # --- Update objective --
        sdp.obj_val = dot(sdp.data.c, sdp.xk)

        # --- Logging ---
        time_sec = (time_ns() - solve_time_start) / 1e9
        if logging
            #TODO:
        end

        # --- Printing ---
        if t == 1 || t % print_iter == 0
            print_iter_func((
                string(t),
                sdp.obj_val,
                sum(x->x^2, rp),
                sum(x->x^2, rd),
                sdp.ρ,
                time_sec
            ))
        end

        t += 1
    end
end