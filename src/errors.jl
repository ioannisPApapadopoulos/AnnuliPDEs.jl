"""
Helper functions for computing the solutions errors
"""

###
# Collect error routines for the Zernike (annular) basis
###

# Collect errors for the (1-element) weighted Zernike (annular) basis
function collect_errors(Z::Weighted{T, <:MultivariateOrthogonalPolynomial}, u::PseudoBlockVector, ua::Function, errors=[]) where T

    @assert Z.P isa ZernikeAnnulus || Z.P isa Zernike
    w = weight(Z)
    (a, b) = unweighted(Z).a, unweighted(Z).b

    N = 2*size((ModalTrav(u).matrix),1)-1
    g = AlgebraicCurveOrthogonalPolynomials.grid(Z, Block(N))

    if Z.P isa ZernikeAnnulus
        ρ = unweighted(Z).ρ 
        F = ZernikeAnnulusITransform{T}(N, a, b, 0, ρ)
    else
        F = ZernikeITransform{T}(N, a, b)
    end

    vals = F * u # synthesis - transform to grid

    vals = w[g].*vals # add in weight

    p = g -> [g.r, g.θ]
    rθ = map(p, g)
    R = first.(rθ)
    Θ = last.(rθ)
    U = map(ua, R, Θ)
    
    append!(errors, norm(U-vals, ∞))
    return errors
end

# Collect errors for the (2-element) Zernike + Zernike annular basis
function collect_errors(Z::Vector{MultivariateOrthogonalPolynomial{2,T}}, u::Tuple{ModalTrav{T, Matrix{T}}, ModalTrav{T, Matrix{T}}}, ua::Function, scale::Bool=true, errors=[]) where T

    @assert Z[1] isa ZernikeAnnulus && Z[2] isa Zernike

    (α, β, ρ) = Z[1].a, Z[1].b, Z[1].ρ
    (a, b) = Z[2].a, Z[2].b


    N = 2*size((ModalTrav(u[1]).matrix),1)-1
    gₐ = AlgebraicCurveOrthogonalPolynomials.grid(Z[1], Block(N))
    g = AlgebraicCurveOrthogonalPolynomials.grid(Z[2], Block(N))

    Fₐ = ZernikeAnnulusITransform{T}(N, α, β, 0, ρ)
    F = ZernikeITransform{T}(N, a, b)
    valsₐ = Fₐ * u[1] # Synthesis on annulus cell
    vals = F * u[2]   # Synthesis on disk cell

    p = g -> [g.r, g.θ]
    rθₐ = map(p, gₐ)
    Rₐ = first.(rθₐ)
    Θₐ = last.(rθₐ)
    Uₐ = map(ua, Rₐ, Θₐ)

    rθ = map(p, g)
    R = first.(rθ)
    # Scale the disk cell so the outer radius is ρ
    if scale
        R = ρ*R
    end
    Θ = last.(rθ)
    U = map(ua, R, Θ)
    
    append!(errors, [max(norm(Uₐ-valsₐ, ∞), norm(U-vals, ∞))])
    return errors
end

###
# Collect error routines for the Chebyshev/two-band - Fourier series
###

# One disk/annulus cell
function _collect_errors(U::Adjoint{T, Matrix{T}}, 𝛉::AbstractArray, 𝐫::AbstractArray, ua::Function, errors=[]) where T
    
    R = repeat(𝐫, 1, length(𝛉))
    Θ = repeat(𝛉, 1, length(𝐫))
    Ua = map(ua, R, Θ')
    
    append!(errors, norm(Ua-U, ∞))
    return errors
end

# One disk cell + one annulus cell
function _collect_errors(U::Adjoint{T, Matrix{T}}, Uₐ::Adjoint{T, Matrix{T}}, θ::AbstractArray, r::AbstractArray, rₐ::AbstractArray, ua::Function, errors=[]) where T
    
    R = repeat(r, 1, length(θ))
    Rₐ = repeat(rₐ, 1, length(θ))
    Θ = repeat(θ, 1, length(r))
    Θₐ = repeat(θ, 1, length(rₐ))
    Ua = map(ua, R, Θ')
    Uaₐ = map(ua, Rₐ, Θₐ')
    
    append!(errors, max(norm(Ua-U, ∞),norm(Uaₐ-Uₐ, ∞)))
    return errors
end

# Works for both Chebyshev and Two-band
function collect_errors(TFρ::Tuple, X::AbstractMatrix, ua::Function, errors=[])

    # One cell routine
    if length(TFρ) == 3
        (T, F, ρ) = TFρ
        try
            @assert T.parent isa ChebyshevT 
        catch 
            @assert T.P isa TwoBandJacobi
        end
        @assert F isa Fourier
        @assert ρ isa Number

        V = promote_type(eltype(T), eltype(F))

        n = size(X,1)-2
        Z = ZernikeAnnulus{V}(ρ,1,1) # want to measure the errors on the ZernikeAnnulus grid

        g = AlgebraicCurveOrthogonalPolynomials.grid(Z, Block(n))
        p = g -> [g.r, g.θ]; rθ = map(p, g); rs = first.(rθ)[:,1]; θs = last.(rθ)[1,:]
        # Compute values of the solution on the grid
        Uu = (F[θs,1:n+2]*(T[rs,1:n+2]*X)')' # Directly expand the tensor-proudct basis on the grid
        return _collect_errors(Uu, θs, rs, ua, errors)
    # Two cell routine
    elseif length(TFρ) == 4
        (T, Tₐ, F, ρ) = TFρ
        @assert T.parent isa ChebyshevT && Tₐ.parent isa ChebyshevT
        @assert F isa Fourier
        @assert ρ isa Number

        V = promote_type(eltype(T), eltype(Tₐ), eltype(F))

        n = (size(X,1)-2)÷2
        # When doing spectral element, we use these grids instead
        Z = Zernike{V}(0,0)
        Zₐ = ZernikeAnnulus{V}(ρ,0,0)
        
        p = g -> [g.r, g.θ];
        g = AlgebraicCurveOrthogonalPolynomials.grid(Z, Block(n))
        gₐ = AlgebraicCurveOrthogonalPolynomials.grid(Zₐ, Block(n))
        rθ = map(p, g); rs = ρ.*first.(rθ)[:,1]; θs = last.(rθ)[1,:]
        rθₐ = map(p, gₐ); rsₐ = first.(rθₐ)[:,1]; θsₐ = last.(rθₐ)[1,:]
    
        Uu = (F[θs,1:n+1]*(T[rs,1:n+1]*X[n+2:end,1:n+1])')'
        Uuₐ = (F[θsₐ,1:n+1]*(Tₐ[rsₐ,1:n+1]*X[1:n+1,1:n+1])')'
        
        return _collect_errors(Uu, Uuₐ, θs, rs, rsₐ, ua, errors)
    else
        error("Collect error not implemented for these arguments.")
    end

end