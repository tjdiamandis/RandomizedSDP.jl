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
    ρ::T                # ADMM param 
    α::T                # over-relaxation param
    cache
    function SDP(c::Vector{T}, A::Matrix{T}, b::Vector{T}; ρ=1.0, α=1.6) where {T <: Real}
        m, n = size(A)
        data = ProblemData(c, A, b)
        return new{T}(
            data,
            zero(T),
            zeros(T, size(c)),
            zeros(T, size(c)),
            zeros(T, size(c)),
            T(ρ),
            T(α)
        )
    end
end

struct Log{T}
    obj_val::Union{AbstractVector{T}, Nothing}
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

function update_x!(
    sdp::SDP,
    AAT_factorization::F,
    indirect::Bool, 
    solver::S, 
    P, 
    cache
) where {S <: Union{CgSolver, Nothing}, F <: Union{Nothing, Cholesky}}
    @. cache.d = -sdp.data.c + sdp.ρ * (sdp.zk - sdp.uk)
    mul!(cache.x_lhs, sdp.data.A, cache.d)
    @. cache.x_lhs -= sdp.ρ * sdp.data.b
    
    if indirect
        cg!(solver, sdp.data.AAT, cache.x_lhs; M=P)
        !issolved(solver) && error("CG failed")
        mul!(cache.d, sdp.data.A', solver.x, -1.0, 1.0)
    else
        ldiv!(cache.ν, AAT_factorization, cache.x_lhs)
        mul!(cache.d, sdp.data.A', cache.ν, -1.0, 1.0)
    end

    sdp.xk .= cache.d
    return nothing
end

function update_z!(sdp::SDP; relax=true, x_relax=nothing)
    if !relax
        x_relax = sdp.xk
    end
    d, V = eigen(unvec_symm(x_relax + sdp.uk), sortby=x->-x)
    nn = count(>(0), d)
    sdp.zk .= vec_symm(V[:, 1:nn] * Diagonal(d[1:nn]) * V[:, 1:nn]')
    return nothing
end

function update_u!(sdp::SDP{T}; relax=true, x_relax=nothing) where {T <: Real}
    if !relax
        x_relax = sdp.xk
    end
    @. sdp.uk += x_relax - sdp.zk
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

function converged(sdp::SDP, rp, rd, eps_abs, eps_rel)
    primal = rp ≤ sqrt(sdp.data.m) * eps_abs + eps_rel * max(norm(sdp.xk), norm(sdp.uk), norm(sdp.data.c))
    dual = rd ≤ sqrt(sdp.data.n) * eps_abs + sdp.ρ * norm(sdp.uk) * eps_rel
    return primal && dual
end

function init_cache(sdp::SDP)
    cache = (
        uk_old = zeros(size(sdp.uk)),
        d = zeros(size(sdp.xk)),
        x_lhs = zeros(size(sdp.data.b)),
        ν = zeros(size(sdp.data.b)),
    )
    return cache
end

function solve!(
    sdp::SDP{T};
    relax::Bool=true,
    logging::Bool=false,
    indirect::Bool=false,
    precondition::Bool=false,
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
    rp, rd = Inf, Inf
    r0 = m ≥ 200 ? 110 : m ÷ 10

    # --- enable multithreaded BLAS ---
    BLAS.set_num_threads(Sys.CPU_THREADS)

    # --- allocate memory ---
    if relax
        x_relax = copy(sdp.xk)
    else
        x_relax = nothing
    end
    if isnothing(cache)
        cache = init_cache(sdp)
    end

    # --- Setup Linear System Solver ---
    AAT_factorization, solver = nothing, nothing
    P = I
    precond_time = 0.0
    if indirect
        solver = CgSolver(m, m, typeof(sdp.xk))
        if precondition
            @printf("\n\tPreconditioning...")
            precond_time_start = time_ns()
            AAT_nys = RP.adaptive_sketch(sdp.data.AAT, r0, RP.NystromSketch; q_norm=20, tol=eps()*m^2)
            P = RP.NystromPreconditionerInverse(AAT_nys, ρ)
            precond_time = (time_ns() - precond_time_start) / 1e9
            r = length(AAT_nys.Λ.diag)
            @printf("\n\tPreconditioned (rank %d) in %6.3fs", r, precond_time)
        end
    else
        AAT_factorization = cholesky(sdp.data.AAT)
    end

    # --- setup log ---
    if logging
        obj_val_log = zeros(max_iters)
        iter_time_log = zeros(max_iters)
        rp_log = zeros(max_iters)
        rd_log = zeros(max_iters)
    else
        obj_val_log = nothing
        iter_time_log = nothing
        rp_log = nothing
        rd_log = nothing
    end

    setup_time = (time_ns() - setup_time_start) / 1e9
    @printf("\nSetup in %6.3fs\n", setup_time)

    # --- Print Headers ---
    headers = ["Iteration", "Objective", "Primal Res", "Dual Res", "ρ", "Time"]
    print_header(headers)
    
    # --------------------------------------------------------------------------
    # --------------------- ITERATIONS -----------------------------------------
    # --------------------------------------------------------------------------
    solve_time_start = time_ns()
    while t <= max_iters && !converged(sdp, rp, rd, eps_abs, eps_rel)
        
        # --- Update Iterates ---
        update_x!(sdp, AAT_factorization, indirect, solver, P, cache)
        if relax
            @. x_relax = α * sdp.xk + (1-α) * sdp.zk
        end
        update_z!(sdp; relax=relax, x_relax=x_relax)
        cache.uk_old .= sdp.uk
        update_u!(sdp; relax=relax, x_relax=x_relax)

        # --- Update ρ ---
        rp = norm(sdp.xk - sdp.zk)
        rd = norm(sdp.ρ*(sdp.uk - cache.uk_old))
        update_rho!(sdp, rp, rd, μ, τ_inc, τ_dec)

        # --- Update objective --
        sdp.obj_val = dot(sdp.data.c, sdp.xk)

        # --- Logging ---
        time_sec = (time_ns() - solve_time_start) / 1e9
        if logging
            obj_val_log[t] = sdp.obj_val
            iter_time_log[t] = time_sec
            rp_log[t] = rp
            rd_log[t] = rd
        end

        # --- Printing ---
        if t == 1 || t % print_iter == 0
            print_iter_func((
                string(t),
                sdp.obj_val,
                rp,
                rd,
                sdp.ρ,
                time_sec
            ))
        end

        t += 1
    end

    # print final iteration if havent done so
    if (t-1) % print_iter != 0 && (t-1) != 1
        rp = norm(sdp.xk - sdp.zk)
        rd = norm(sdp.ρ*(sdp.uk - cache.uk_old))
        print_iter_func((
            string(t-1),
            sdp.obj_val,
            rp,
            rd,
            sdp.ρ,
            (time_ns() - solve_time_start) / 1e9
        ))
    end
    solve_time = (time_ns() - solve_time_start) / 1e9
    @printf("\nSolved in %6.3fs, %d iterations\n", solve_time, t-1)
    @printf("Total time: %6.3fs\n", setup_time + solve_time)
    print_footer()
    
    
    # --- Construct Logs ---
    if logging
        log = Log(
            obj_val_log[1:t-1], 
            iter_time_log[1:t-1],
            rp_log[1:t-1], 
            rd_log[1:t-1],
            setup_time, precond_time, solve_time
        )
    else
        log = Log(
            nothing, nothing, nothing, nothing,
            setup_time, precond_time, solve_time
        )
    end


    # --- Construct Solution ---
    res = Result(
        sdp.obj_val,
        sdp.xk,
        log
    )

    return res
end