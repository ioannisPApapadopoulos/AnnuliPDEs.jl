"""
Helper functions for computing the solutions errors
"""

function collect_errors(Z::Weighted{T, <:MultivariateOrthogonalPolynomial}, u::PseudoBlockVector, ua::Function, errors=[]) where T

    @assert Z.P isa ZernikeAnnulus || Z.P isa Zernike
    w = weight(Z)
    (a, b) = unweighted(Z).a, unweighted(Z).b

    N = 2*size((ModalTrav(u).matrix),1)-1
    g = AlgebraicCurveOrthogonalPolynomials.grid(Z, Block(N))

    if Z.P isa ZernikeAnnulus
        ρ = unweighted(Z).ρ 
        F = ZernikeAnnulusITransform{T}(N, a, b, 0, ρ) # 0 should be Z.c if implemented.
    else
        F = ZernikeITransform{T}(N, a, b)
    end

    vals = F * u # transform to grid

    vals = w[g].*vals

    p = g -> [g.r, g.θ]
    rθ = map(p, g)
    R = first.(rθ)
    Θ = last.(rθ)
    U = map(ua, R, Θ)
    
    append!(errors, norm(U-vals, ∞))
    return errors
end

function collect_errors(Z::Vector{MultivariateOrthogonalPolynomial{2,T}}, u::Tuple{ModalTrav{T, Matrix{T}}, ModalTrav{T, Matrix{T}}}, ua::Function, scale::Bool=true, errors=[]) where T

    (a, b) = Z[2].a, Z[2].b
    if Z[1] isa Weighted
        (α, β, ρ) = Z[1].P.a, Z[1].P.b, Z[1].P.ρ 
        w = weight(Z[1])
    else
        (α, β, ρ) = Z[1].a, Z[1].b, Z[1].ρ 
    end

    N = 2*size((ModalTrav(u[1]).matrix),1)-1
    gₐ = AlgebraicCurveOrthogonalPolynomials.grid(Z[1], Block(N))
    g = AlgebraicCurveOrthogonalPolynomials.grid(Z[2], Block(N))

    Fₐ = ZernikeAnnulusITransform{T}(N, α, β, 0, ρ) # 0 should be Z.c if implemented.
    F = ZernikeITransform{T}(N, a, b)
    valsₐ = Fₐ * u[1]
    vals = F * u[2]
    
    if Z[1] isa Weighted
        valsₐ = w[gₐ].*valsₐ
    end
    
    p = g -> [g.r, g.θ]
    rθₐ = map(p, gₐ)
    Rₐ = first.(rθₐ)
    Θₐ = last.(rθₐ)
    Uₐ = map(ua, Rₐ, Θₐ)

    rθ = map(p, g)
    R = first.(rθ)
    if scale
        R = ρ*R
    end
    Θ = last.(rθ)
    U = map(ua, R, Θ)
    
    append!(errors, [max(norm(Uₐ-valsₐ, ∞), norm(U-vals, ∞))])
    return errors
end

function collect_errors(U::Adjoint{T, Matrix{T}}, 𝛉::AbstractArray, 𝐫::AbstractArray, ua::Function, errors=[]) where T
    
    R = repeat(𝐫, 1, length(𝛉))
    Θ = repeat(𝛉, 1, length(𝐫))
    Ua = map(ua, R, Θ')
    
    append!(errors, norm(Ua-U, ∞))
    return errors
end

function collect_errors(U::Adjoint{T, Matrix{T}}, Uₐ::Adjoint{T, Matrix{T}}, θ::AbstractArray, r::AbstractArray, rₐ::AbstractArray, ua::Function, errors=[]) where T
    
    R = repeat(r, 1, length(θ))
    Rₐ = repeat(rₐ, 1, length(θ))
    Θ = repeat(θ, 1, length(r))
    Θₐ = repeat(θ, 1, length(rₐ))
    Ua = map(ua, R, Θ')
    Uaₐ = map(ua, Rₐ, Θₐ')
    
    append!(errors, max(norm(Ua-U, ∞),norm(Uaₐ-Uₐ, ∞)))
    return errors
end

function collect_errors(TFρ::Tuple, X::AbstractMatrix, ua::Function, errors=[])
    if length(TFρ) == 3
        (T, F, ρ) = TFρ
        n = size(X,1)-2
        Z = ZernikeAnnulus(ρ,1,1)
        g = AlgebraicCurveOrthogonalPolynomials.grid(Z, Block(n))
        p = g -> [g.r, g.θ]; rθ = map(p, g); rs = first.(rθ)[:,1]; θs = last.(rθ)[1,:]
        # Compute values of the solution on the grid
        Uu = (F[θs,1:n+2]*(T[rs,1:n+2]*X)')'
        return collect_errors(Uu, θs, rs, ua, errors)
    elseif length(TFρ) == 4
        (T, Tₐ, F, ρ) = TFρ
        n = (size(X,1)-2)÷2
        Z = Zernike(0,0)
        Zₐ = ZernikeAnnulus(ρ,0,0)
        
        p = g -> [g.r, g.θ];
        g = AlgebraicCurveOrthogonalPolynomials.grid(Z, Block(n))
        gₐ = AlgebraicCurveOrthogonalPolynomials.grid(Zₐ, Block(n))
        rθ = map(p, g); rs = ρ.*first.(rθ)[:,1]; θs = last.(rθ)[1,:]
        rθₐ = map(p, gₐ); rsₐ = first.(rθₐ)[:,1]; θsₐ = last.(rθₐ)[1,:]
    
        Uu = (F[θs,1:n+1]*(T[rs,1:n+1]*X[n+2:end,1:n+1])')'
        Uuₐ = (F[θsₐ,1:n+1]*(Tₐ[rsₐ,1:n+1]*X[1:n+1,1:n+1])')'
        
        return collect_errors(Uu, Uuₐ, θs, rs, rsₐ, ua, errors)
    else
        error("Collect error not implemented for these arguments.")
    end

end
# function collect_errors2(y, ua, errors=[])
#     Z, c = y.args
#     if Z isa Weighted
#         # bug to do with c not being blocked with a weighted polynomial
#         c = unweighted(y).args[2]
#         w = weight(y.args[1])
#         (a, b, ρ) = unweighted(Z).a, unweighted(Z).b, unweighted(Z).ρ 
#     else
#         (a, b, ρ) = Z.a, Z.b, Z.ρ 
#     end
#     CS = blockcolsupport(c)
#     Bs = last(CS)
#     N = Int(Bs)
#     print(N)
#     g = AlgebraicCurveOrthogonalPolynomials.grid(Z, Bs)

#     F = ZernikeAnnulusITransform{Float64}(N, a, b, 0, ρ) # 0 should be Z.c if implemented.
#     vals = F * c[Block.(OneTo(N))] # transform to grid
#     if Z isa Weighted
#         vals = w[g].*vals
#     end
#     p = g -> [g.r, g.θ]
#     rθ = map(p, g)
#     R = first.(rθ)
#     Θ = last.(rθ)
#     U = map(ua, R, Θ)
    
#     append!(errors, norm(U-vals, ∞))
#     # writedlm("examples/logs/errors-helmholtz.txt", errors)
#     return errors
# end