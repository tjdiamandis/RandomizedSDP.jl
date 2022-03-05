
function rand_project_psd_cone(A::AbstractMatrix, r::Int; q::Int=0)
    Q = RP.rangefinder(A, r; q=q)
    C = Q' * A * Q
    @. C = 0.5 * (C + C')
    Λ, V = eigen(C; sortby=x->-real(x))
    V .= real.(V)
    Λ .= real(Λ)
    nn = count(>(0), Λ)
    return Q * V[:, 1:nn] * Diagonal(Λ[1:nn]) * V[:, 1:nn]' * Q'
end