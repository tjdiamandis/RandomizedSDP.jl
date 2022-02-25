# Adapted from
# https://github.com/ZIB-IOL/FrankWolfe.jl/blob/master/src/fw_algorithms.jl
function print_header(data)
    @printf(
        "\n─────────────────────────────────────────────────────────────────────────────────────────────────\n"
    )
    @printf(
        "%13s %14s %14s %14s %14s %14s\n",
        data[1],
        data[2],
        data[3],
        data[4],
        data[5],
        data[6]
    )
    @printf(
        "─────────────────────────────────────────────────────────────────────────────────────────────────\n"
    )
end

function print_footer()
    @printf(
        "─────────────────────────────────────────────────────────────────────────────────────────────────\n\n"
    )
end

function print_iter_func(data)
    @printf(
        "%13s %14e %14e %14e %14.3e %13.3f\n",
        data[1],
        Float64(data[2]),
        Float64(data[3]),
        Float64(data[4]),
        data[5],
        data[6]
    )
end


"""
   unvec_symm(x)

Returns a dim-by-dim symmetric matrix corresponding to `x`.
`x` is a vector of length dim*(dim + 1)/2, corresponding to a symmetric matrix
```
X = [ X11     X12/√2 ... X1k/√2
      X21/√2  X22    ... X2k/√2
      ...
      Xk1/√2  Xk2/√2 ... Xkk ],
```
where
`vec(X) = (X11, X12, X22, X13, X23, ..., Xkk)`

Note that the factor √2 preserves inner products:
`x'*c = Tr(unvec_symm(c, dim) * unvec_symm(x, dim))`
"""
function unvec_symm(x::Vector{T}, dim::Int) where {T <: Number}
    X = zeros(T, dim, dim)
    idx = 1
    for i in 1:dim
        for j in 1:i
            if i == j
                X[i,j] = x[idx]
            else
                X[j,i] = X[i,j] = x[idx] / sqrt(2)
            end
            idx += 1
        end
    end
    return X
end

function unvec_symm(x::Vector{T}) where {T <: Number}
    dim = Int( (-1 + sqrt(1 + 8*length(x))) / 2 )
    dim * (dim + 1) ÷ 2 != length(x) && throw(DomainError("invalid vector length"))
    return unvec_symm(x, dim)
end


"""
   vec_symm(X)

Returns a vectorized representation of a symmetric matrix `X`.
`vec(X) = (X11, √2*X12, X22, √2*X13, X23, ..., Xkk)`

Note that the factor √2 preserves inner products:
`x'*c = Tr(unvec_symm(c, dim) * unvec_symm(x, dim))`
"""
function vec_symm(X)
    x_vec = sqrt(2).*X[LinearAlgebra.triu(trues(size(X)))]
    idx = 1
    for i in 1:size(X)[1]
        x_vec[idx] =  x_vec[idx]/sqrt(2)
        idx += i + 1
    end
    return x_vec
end


function build_A(As::Vector{AbstractMatrix})
    m = length(As)
    n_ = size(As[1], 1)
    n = n_ * (n_ + 1) ÷ 2
    # TODO: Handle sparse mats
    A = zeros(m, n)
    for (i, Ai) in enumerate(As)
        A[i,:] .= vec_symm(Ai)
    end
    return A
end