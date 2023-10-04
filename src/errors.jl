"""
Helper functions for computing the solutions errors
"""

###
# Collect error routines for the Zernike (annular) basis
###

function collect_errors(Z::Weighted{T, <:MultivariateOrthogonalPolynomial}, u::AbstractVector, U::AbstractMatrix, G::AbstractMatrix, errors=[]; rvals=false) where T

    N = 2*(size(U,1)-1)
    u_ = pad(u, axes(Z,2))[Block.(1:N)]

    if Z.P isa ZernikeAnnulus
        ρ = unweighted(Z).ρ 
        F = ZernikeAnnulusITransform{T}(N, unweighted(Z).a, unweighted(Z).b, 0, ρ)
    else
        F = ZernikeITransform{T}(N, unweighted(Z).a, unweighted(Z).b)
    end

    vals = F * u_ # synthesis - transform to grid

    w = weight(Z)
    vals = w[G].*vals # add in weight
    append!(errors, norm(U-vals, ∞))
    rvals && return (errors, vals)
    return errors
end

function collect_errors(Z::MultivariateOrthogonalPolynomial, u::AbstractVector, U::AbstractMatrix, errors=[]; rvals=false)
    T = eltype(Z)

    N = 2*(size(U,1)-1)
    u_ = pad(u, axes(Z,2))[Block.(1:N)]

    if Z isa ZernikeAnnulus
        ρ = Z.ρ
        F = ZernikeAnnulusITransform{T}(N, Z.a, Z.b, 0, ρ)
    else
        F = ZernikeITransform{T}(N, Z.a, Z.b)
    end

    vals = F * u_ # synthesis - transform to grid
    append!(errors, norm(U-vals, ∞))
    rvals && return (errors, vals)
    return errors
end

# function collect_errors(Z::Weighted{T, <:MultivariateOrthogonalPolynomial}, u::PseudoBlockVector, ua::Function, errors=[]) where T
#     _collect_errors(Z, u, ua, errors)
# end

# function collect_errors(Z::Weighted{T, <:MultivariateOrthogonalPolynomial}, u::BlockVector, ua::Function, errors=[]) where T
#     _collect_errors(Z, u, ua, errors)
# end

# function collect_errors(Z::MultivariateOrthogonalPolynomial, u::PseudoBlockVector, ua::Function, errors=[])
#     _collect_errors(Z, u, ua, errors)
# end

# function collect_errors(Z::MultivariateOrthogonalPolynomial, u::BlockVector, ua::Function, errors=[])
#     _collect_errors(Z, u, ua, errors)
# end

# Collect errors for the (2-element) Zernike + Zernike annular basis
function collect_errors(Z::Vector{MultivariateOrthogonalPolynomial{2,T}}, u::Tuple{ModalTrav{T, Matrix{T}}, ModalTrav{T, Matrix{T}}}, U, errors=[]; rvals=false) where T

    @assert Z[1] isa ZernikeAnnulus && Z[2] isa Zernike

    (α, β, ρ) = Z[1].a, Z[1].b, Z[1].ρ
    (a, b) = Z[2].a, Z[2].b

    N = 2*(size(U[1],1)-1)
    u_ = [pad(u[i], axes(Z[i],2))[Block.(1:N)] for i in 1:lastindex(u)]

    F = [ZernikeAnnulusITransform{T}(N, α, β, 0, ρ), ZernikeITransform{T}(N, a, b)]

    vals = [F[i] * u_[i] for i in 1:lastindex(u_)]   # Synthesis
    
    append!(errors, [max(norm(U[1]-vals[1], ∞), norm(U[2]-vals[2], ∞))])
    # append!(errors, [(norm(Uₐ-valsₐ, ∞), norm(U-vals, ∞))])
    rvals && return (errors, vals)
    return errors
end

# Collect errors for the (2-element) Zernike annular + Zernike annular basis
function collect_errors(Z::Vector{ZernikeAnnulus{T}}, u::Tuple{ModalTrav{T, Matrix{T}}, ModalTrav{T, Matrix{T}}}, U, errors=[]; rvals=false) where T

    @assert Z[1] isa ZernikeAnnulus && Z[2] isa ZernikeAnnulus

    N = 2*(size(U[1],1)-1)
    u_ = [pad(u[i], axes(Z[i],2))[Block.(1:N)] for i in 1:lastindex(u)]

    F = [ZernikeAnnulusITransform{T}(N, Z[i].a, Z[i].b, 0, Z[i].ρ) for i in 1:lastindex(Z)]

    vals = [F[i] * u_[i] for i in 1:lastindex(u_)]   # Synthesis

    append!(errors, [max(norm(U[1]-vals[1], ∞), norm(U[2]-vals[2], ∞))])
    rvals && return (errors, vals)
    return errors
end

###
# Collect error routines for the Chebyshev/two-band - Fourier series
###

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
function collect_errors(TFρ::Tuple, X::AbstractMatrix, U, G, errors=[]; scale=1.0, rvals=false)

    (T, F, ρ) = TFρ
    try
        @assert T.parent isa ChebyshevT
    catch
        @assert T.P isa TwoBandJacobi
    end
    @assert F isa Fourier
    @assert ρ isa Number

    rs = scale*get_rs.(G)[:,1]
    θs = get_θs.(G)[1,:]

    # Compute values of the solution on the grid
    vals = (F[θs,1:size(X,2)]*(T[rs,1:size(X,1)]*X)')' # Directly expand the tensor-product basis on the grid
    append!(errors, norm(U-vals, ∞))
    rvals && return (errors, vals)
    return errors
end

###
# Collect error routines for the Zernike + Chebyshev-Fourier series
###
function collect_errors(TFZρ::Tuple, Xu::Tuple, U, G, errors=[])

    (T, F, Z, ρ) = TFZρ
    (X, u) = Xu

    err1 = collect_errors((T, F, ρ), X, U[1], G[1], [])
    err2 = collect_errors(Z, u, U[2])

    append!(errors, max(err1[1],err2[1]))
end

###
# two cell chebyshev-fourier on an annulus domain
###
function collect_errors(TFαρ::Tuple{<:Any, <:Any, <:Fourier, <:Number, <:Number}, X::AbstractMatrix, U, G, errors=[])

    # Two cell routine
    (T, Tₐ, F, α, ρ) = TFαρ
    @assert T.parent isa ChebyshevT && Tₐ.parent isa ChebyshevT
    @assert F isa Fourier
    @assert α isa Number
    @assert ρ isa Number

    n = (size(X,1)-2)÷2

    rs = [get_rs.(G[1])[:,1]]
    append!(rs, [ρ*get_rs.(G[2])[:,1]])
    θs = [get_θs.(G[i])[1,:] for i in 1:lastindex(G)]

    Uuₐ = (F[θs[1],1:2n-1]*(Tₐ[rs[1],1:n+1]*X[1:n+1,1:2n-1])')'
    Uu = (F[θs[2],1:2n-1]*(T[rs[2],1:n+1]*X[n+2:end,1:2n-1])')'

    return append!(errors, max(norm(Uuₐ-U[1], ∞),norm(Uu-U[2], ∞)))
    # return [Uuₐ, Uu]
end

###
# two cell chebyshev-fourier in inner annulus, ZernikeAnnulus on outer annulus
###
function collect_errors(ZTFαρ::Tuple{<:ZernikeAnnulus, <:Any, <:Fourier, <:Number, <:Number}, uX::Tuple{ModalTrav{TV, Matrix{TV}}, Matrix{TV}}, U, G, errors=[]; rvals=false) where TV

    (Z, T, F, α, ρ) = ZTFαρ
    (u, X) = uX

    err1 = collect_errors((T, F, ρ), X, U[2], G[2], [], scale=ρ, rvals=rvals)
    err2 = collect_errors(Z, u, U[1], rvals=rvals)

    # return append!(errors, [[err2[1], err1[1]]])
    append!(errors, max(err1[1],err2[1]))
    rvals && return (errors, [err2[2], err1[2]])
    return errors
end