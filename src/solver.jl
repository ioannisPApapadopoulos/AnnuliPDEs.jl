"""
Solver routines
"""

# This function splits the Helmholtz solve via weighted Zernike (annular) polys into a series of
# one-dimensional solves per Fourier mode with decreasing size.
function weighted_zernike_modal_solve(f::AbstractBlockArray, b::Int, Δs, Ls, λ::T=0.0, mmode=1:b) where T 

    fs = ModalTrav(f).matrix
    N = size(fs,1)
    us = zeros(N, size(fs,2))
    
    for j in reverse(mmode)
        M = length(j:2:b)

        if iszero(λ)
            A = Δs[j][1:M, 1:M] # view(Δs[j],1:M,1:M)
        else
            A = Δs[j][1:M, 1:M] + (λ*Ls[j])[1:M, 1:M]
        end

        if j == 1
            us[1:M,1] = A \ fs[1:M,1];
        else
            us[1:M,2j-2] = A \ fs[1:M,2j-2];
            us[1:M,2j-1] = A \ fs[1:M,2j-1];
        end
    end
    ModalTrav(us)[Block.(1:b)]
end

# This function splits the Helmholtz solve via Zernike (annular) polys into a series of
# one-dimensional solves per Fourier mode with decreasing size.
function zernike_modal_solve(f::AbstractBlockArray, b::Int, Δs, Ls, Z::ZernikeAnnulus{T}, λ::T=0.0, mmode=1:b, w::Function=(r,m)->r^m; bcs="Dirichlet") where T

    fs = ModalTrav(f).matrix
    N = size(fs,1)
    us = zeros(N, size(fs,2))

    # Used to compute the entries to enforce continuity. Much faster
    # than evaluating ZernikeAnnulus directly.
    P = SemiclassicalJacobi.(inv(one(T)-Z.ρ^2), Z.b, Z.a, 0:b-1)

    if bcs == "mixed"
        Q = SemiclassicalJacobi.(inv(1-Z.ρ^2), Z.b+1, Z.a+1, 1:b)
        Ds = Q .\ [Derivative(axes(Q[1], 1)) * P[j] for j in 1:lastindex(P)];
    end

    for j in mmode
        m = length(j:2:b) # Length of system
        A = zeros(m+2, m+2) # Preallocate space for the matrix. Require m+1 (+1 for tau-method)
        c1 = P[j][begin, 1:m+1] # Boundary condition at r=1

        if bcs == "Dirichlet"
            A[1,1:m+1] = c1  # Dirichlet boundary condition at r = 1
            cρ = w(Z.ρ, j-1) * P[j][end, 1:m+1] # Contiuity condition at r=ρ
            A[2,1:m+1] = cρ  # boundary condition at r = ρ
        elseif bcs == "mixed"
            ρ = Z.ρ
            # Neumann condition at r = ρ
            dcρ = (
                w(ρ, j-1) * 2ρ/(ρ^2-1) *(Q[j][end, 1:m+1]' * Ds[j][1:m+1, 1:m+1])'
                + derivative(r->w(r, j-1), ρ) * P[j][end, 1:m+1]
            )
            A[1,1:m+1] = c1  # Dirichlet boundary condition at r = 1
            A[2,1:m+1] = dcρ # Neumann boundary condition at r = ρ
        else
            error("Choice of bcs=$bcs not recognized.")
        end

        if iszero(λ)
            A[3:end, 1:m+1] = view(Δs[j],1:m,1:m+1)
        else
            A[3:end, 1:m+1] = view(Δs[j],1:m,1:m+1) + λ*view(Ls[j],1:m,1:m+1)
        end

        # tau-method corrections
        # A[2,end] = 1-exp(-(j-1))
        A[end, end] = 1.
        # A[2,end] = 1-exp(-j)
        # A[end,end] = exp(-(j-1))

        # A = A[1:end-1,1:end-1]
        if j == 1
            us[1:m+1,1] = (A \ [zero(T); zero(T); fs[1:m,1]])[1:end-1]
        else
            us[1:m+1,2j-2] = (A \ [zero(T); zero(T);fs[1:m,2j-2]])[1:end-1]
            us[1:m+1,2j-1] = (A \ [zero(T); zero(T);fs[1:m,2j-1]])[1:end-1]
        end
    end
    ModalTrav(us)[Block.(1:b+1)]
end


# This function splits the Helmholtz solve via Zernike (annular) polys into a series of
# one-dimensional solves per Fourier mode with decreasing size. This is adapted for
# the spectral element method and also implements a tau-method for continuity
# across the disk and annulus cell.
function zernike_modal_solve(Z::Vector{<:MultivariateOrthogonalPolynomial{2, <:T}}, f::Vector, b::Int, Δs, Ls, λs::AbstractVector=[], mmode=1:b, w::Function=(r,m)->r^m) where T
    @assert Z[1] isa ZernikeAnnulus
    @assert Z[2] isa Zernike
    @assert f[1] isa AbstractBlockArray
    @assert f[2] isa AbstractBlockArray

    # Decompose into Fourier mode components
    ρ = Z[1].ρ

    # Used to compute the entries to enforce continuity. Much faster
    # than evaluating ZernikeAnnulus directly.
    P = SemiclassicalJacobi.(inv(one(T)-ρ^2), Z[1].b, Z[1].a, 0:b-1)
    Q = SemiclassicalJacobi.(inv(1-ρ^2), Z[1].b+1, Z[1].a+1, 1:b)

    # Break down into Fourier modes
    fs = [ModalTrav(f[j]).matrix for j in 1:2]
    zs = ModalTrav(Z[2][SVector(one(T),zero(T)), Block.(1:b+2)]).matrix

    n = size(fs[1],1)
    us = zeros(2n+2, 4(n+1)-3)

    # For enforcing the radial derivative continuity
    Ds = Q .\ [Derivative(axes(Q[1], 1)) * P[j] for j in 1:lastindex(P)];
    
    ze(r,θ,b) = Z[2][SVector(r*cos(θ)/ρ,r*sin(θ)/ρ), Block.(1:b+2)]
    dze(r,θ,b) = derivative(r->ze(r,θ,b), r)
    dzs = ModalTrav(dze(ρ,0,b)).matrix

    # Loop over Fourier modes
    for j in mmode
        m = length(j:2:b) # Length of system
        A = zeros(2m+3, 2m+3) # Preallocate space for the matrix. Require m+1 for each element +1 for the tau-method
        c1 = P[j][begin, 1:m+1] # Boundary condition at r=1
        cρ = w(ρ, j-1) * P[j][end, 1:m+1] # Contiuity condition at r=ρ

        # Radial derivative continuity condition
        dcρ = (
            w(ρ, j-1) * 2ρ/(ρ^2-1) *(Q[j][end, 1:m+1]' * Ds[j][1:m+1, 1:m+1])'
            + derivative(r->w(r, j-1), ρ) * P[j][end, 1:m+1]
        )

        A[1,1:m+1] = c1                          # boundary condition row
        A[2,1:end-1] = [cρ' -zs[1:m+1,2j-1]']    # Dirichlet element continuity row
        A[3,1:end-1] = [dcρ' -dzs[1:m+1,2j-1]']  # Radial derivative continuity condition row
        
        if iszero(λs)
            A[4:m+3,1:m+1] = Δs[1][j][1:m,1:m+1]
            # Do not forget inv(ρ^2) factor! To do with rescaling the Zernike polys
            A[m+4:end,m+2:end-1] = inv(ρ^2)*view(Δs[2][j], 1:m,1:m+1)
        else
            A[4:m+3,1:m+1] = Δs[1][j][1:m,1:m+1] + λs[1]*Ls[1][j][1:m,1:m+1]
            # Do not forget inv(ρ^2) factor! To do with rescaling the Zernike polys
            A[m+4:end,m+2:end-1] = inv(ρ^2)*view(Δs[2][j], 1:m,1:m+1) + λs[2]*view(Ls[2][j], 1:m,1:m+1)
        end

        A[2, end] = 1-exp(-j) # tau-method stabilisation
        A[m+3,end] = 1. # tau-method stabilisation

        if j == 1
            𝐛 = [0; 0; 0; fs[1][1:m,1]; fs[2][1:m,1]]
            us[:,1] = (A \ 𝐛)[1:end-1]
        else
            𝐛 = [0; 0; 0; fs[1][1:m,2j-2]; fs[2][1:m,2j-2]]
            u_ = A \ 𝐛
            us[1:m+1,2j-2] = u_[1:m+1]; us[n+2:n+m+2,2j-2] = u_[m+2:end-1]

            𝐛 = [0; 0; 0; fs[1][1:m,2j-1]; fs[2][1:m,2j-1]]
            u_ = A \ 𝐛 
            us[1:m+1,2j-1] = u_[1:m+1]; us[n+2:n+m+2,2j-1] = u_[m+2:end-1]
        end
        
    end
    ((ModalTrav(us[1:n+1, :]), ModalTrav(us[n+2:end, :])))
end

#
# SEM of ZernikeAnnular + ZernikeAnnular
#
function zernike_modal_solve(Z::Vector{ZernikeAnnulus{T}}, f::Vector, b::Int, Δs, Ls, λs::AbstractVector=[], mmode=1:b, w::Function=(r,m)->r^m) where T
    @assert Z[1] isa ZernikeAnnulus
    @assert Z[2] isa ZernikeAnnulus
    @assert f[1] isa AbstractBlockArray
    @assert f[2] isa AbstractBlockArray

    ρ = Z[1].ρ

    # Used to compute the entries to enforce continuity. Much faster
    # than evaluating ZernikeAnnulus directly.
    ts = [inv(one(T)-Z[j].ρ^2) for j in 1:2]
    P = [SemiclassicalJacobi.(ts[j], Z[j].b, Z[j].a, 0:b-1) for j in 1:2]
    Q = [SemiclassicalJacobi.(ts[j], Z[j].b+1, Z[j].a+1, 1:b) for j in  1:2]

    # Break down into Fourier modes
    fs = [ModalTrav(f[j]).matrix for j in 1:2]

    n = size(fs[1],1)
    us = zeros(2n+2, 4(n+1)-3)

    # For enforcing the radial derivative continuity
    Ds = [Q[i] .\ [Derivative(axes(Q[i][1], 1)) * P[i][j] for j in 1:lastindex(P[i])] for i in 1:2];

    # Loop over Fourier modes
    for j in mmode
        m = length(j:2:b) # Length of system
        A = zeros(2m+4, 2m+4) # Preallocate space for the matrix. Require m+1 for each element +1 for the tau-method
        c1 = P[1][j][begin, 1:m+1] # Boundary condition at r=1
        cρ = w(ρ, j-1) * P[1][j][end, 1:m+1] # Contiuity condition at r=ρ
        σ1 = P[2][j][begin, 1:m+1]
        σρ = w(Z[2].ρ, j-1) * P[2][j][end, 1:m+1]

        # Radial derivative continuity condition
        dcρ = (
            w(ρ, j-1) * 2ρ/(ρ^2-1) *(Q[1][j][end, 1:m+1]' * Ds[1][j][1:m+1, 1:m+1])'
            + derivative(r->w(r, j-1), ρ) * P[1][j][end, 1:m+1]
        )
        dσ1 = (2 / (ρ*(Z[2].ρ^2 - 1)) * (Q[2][j][begin, 1:m+1]' * Ds[2][j][1:m+1, 1:m+1])'
                + (j-1) / ρ *  P[2][j][begin, 1:m+1])

        A[1,1:m+1] = c1              # boundary condition row
        A[2,1:end-2] = [cρ' -σ1']    # Dirichlet element continuity row
        A[3,1:end-2] = [dcρ' -dσ1']  # Radial derivative continuity condition row
        A[4, m+2:end-2] = σρ

        if iszero(λs)
            A[5:m+4,1:m+1] = Δs[1][j][1:m,1:m+1] # view(Δs[1][j],1:m,1:m+1)
            # Do not forget inv(ρ^2) factor! To do with rescaling the Zernike polys
            A[m+5:end,m+2:end-2] = inv(ρ^2)*Δs[2][j][1:m,1:m+1] #  inv(ρ^2)*(Δs[2][j], 1:m,1:m+1)
        else
            A[5:m+4,1:m+1] = Δs[1][j][1:m,1:m+1] + λs[1]*Ls[1][j][1:m,1:m+1] #view(Δs[1][j],1:m,1:m+1)+ λs[1]*view(Ls[1][j], 1:m,1:m+1)
            # Do not forget inv(ρ^2) factor! To do with rescaling the Zernike polys
            A[m+5:end,m+2:end-2] = inv(ρ^2)*Δs[2][j][1:m,1:m+1] + λs[2]*Ls[2][j][1:m,1:m+1] #inv(ρ^2)*view(Δs[2][j], 1:m,1:m+1) + λs[2]*view(Ls[2][j], 1:m,1:m+1)
        end

        A[2, end-1] = 1-exp(-(j-1)) # tau-method stabilisation
        # A[4, end] = 1-exp(-j)
        A[end,end] = 1.
        A[m+3,end-1] = 1. # tau-method stabilisation

        if j == 1
            𝐛 = [0; 0; 0; 0; fs[1][1:m,1]; fs[2][1:m,1]]
            us[:,1] = (A \ 𝐛)[1:end-2]
        else
            𝐛 = [0; 0; 0; 0; fs[1][1:m,2j-2]; fs[2][1:m,2j-2]]
            u_ = A \ 𝐛
            us[1:m+1,2j-2] = u_[1:m+1]; us[n+2:n+m+2,2j-2] = u_[m+2:end-2]

            𝐛 = [0; 0; 0; 0; fs[1][1:m,2j-1]; fs[2][1:m,2j-1]]
            u_ = A \ 𝐛
            us[1:m+1,2j-1] = u_[1:m+1]; us[n+2:n+m+2,2j-1] = u_[m+2:end-2]
        end

    end
    ((ModalTrav(us[1:n+1, :]), ModalTrav(us[n+2:end, :])))
end

# This function splits the Helmholtz solve via Chebyshev-Fourier into a series of
# one-dimensional solves per Fourier mode.
function chebyshev_fourier_analysis(TF, rhs_xy::Function, nt::Int, nf::Int)

    (T, F) = TF
    @assert T.parent isa ChebyshevT
    @assert F isa Fourier

    # Find coefficient expansion of tensor-product
    𝐫,𝛉 = ClassicalOrthogonalPolynomials.grid(T, 2n),ClassicalOrthogonalPolynomials.grid(F, 2n)
    PT, PF = plan_transform(T, (2n,2n), 1), plan_transform(F, (2n,2n), 2)
    𝐱 = 𝐫 .* cos.(𝛉')
    𝐲 = 𝐫 .* sin.(𝛉')
    Fs = PT * (PF * rhs_xy.(𝐱, 𝐲))

end

# This function splits the Helmholtz solve via Chebyshev-Fourier into a series of
# one-dimensional solves per Fourier mode.
function chebyshev_fourier_helmholtz_modal_solve(TF, LMR, rhs_xy::Function, n::Int, λ::V=0.0; bcs="Dirichlet", Fs=[], rFs=false) where V

    (T, F) = TF
    @assert T.parent isa ChebyshevT
    @assert F isa Fourier

    (L, M, R) = LMR
    # L = (r²*Δ) : ChebyshevT -> Ultraspherical(2)
    # M = I : ChebyshevT -> Ultraspherical(2)
    # R = r : Ultraspherical(2) -> Ultraspherical(2)

    # Find coefficient expansion of tensor-product
    if Fs == []
        𝐫,𝛉 = ClassicalOrthogonalPolynomials.grid(T, 2n),ClassicalOrthogonalPolynomials.grid(F, 2n)
        PT, PF = plan_transform(T, (2n,2n), 1), plan_transform(F, (2n,2n), 2)
        𝐱 = 𝐫 .* cos.(𝛉')
        𝐲 = 𝐫 .* sin.(𝛉')
        Fs = PT * (PF * rhs_xy.(𝐱, 𝐲))
    end
    # Fs[abs.(Fs) .< 10*eps()] .= 0

    # Preallocate space for the coefficients of the solution
    X = zeros(n+2, 2n)
    # multiply RHS by r^2 and convert to C
    S = (R^2*M)[1:n,1:n]

    for j = 1:2n-1
        m = j ÷ 2
        # Add in Fourier part of Laplacian + Helmholtz part
        Δₘ = L - m^2*M + λ*R^2*M
        if bcs == "Dirichlet"
            X[:,j] = [T[[begin,end],:]; Δₘ][1:n+2,1:n+2] \ [0; 0; S*Fs[1:n,j]]
        elseif bcs == "mixed"
            X[:,j] = [(Derivative(axes(T,1))*T)[[begin],1:n+2];T[[end],1:n+2];Δₘ[1:n, 1:n+2]] \ [0; 0; S*Fs[1:n,j]]
        else
            error("Choice of bcs=$bcs not recognized.")
        end
    end
    # X[abs.(X) .< 10*eps()] .= 0
    rFs && return (X, Fs)
    return X
end

function chebyshev_fourier_helmholtz_modal_solve(TF, LMR, Ds, rhs_xy::Function, n::Int, λs::Vector=[])

    (T, Tₐ, F) = TF
    @assert T.parent isa ChebyshevT
    @assert Tₐ.parent isa ChebyshevT
    @assert F isa Fourier

    (L, Lₐ, M, Mₐ, R, Rₐ) = LMR
    # L = (r²*Δ) : ChebyshevT -> Ultraspherical(2) (disk)
    # M = I : ChebyshevT -> Ultraspherical(2) (disk)
    # R = r : Ultraspherical(2) -> Ultraspherical(2) (disk)
    # Lₐ = (r²*Δ) : ChebyshevT -> Ultraspherical(2) (annulus)
    # Mₐ = I : ChebyshevT -> Ultraspherical(2) (annulus)
    # Rₐ = r : Ultraspherical(2) -> Ultraspherical(2) (annulus)

    (D, Dₐ) = Ds
    @assert D isa Derivative
    @assert Dₐ isa Derivative
    
    𝐫,𝛉 = ClassicalOrthogonalPolynomials.grid(T, 2n),ClassicalOrthogonalPolynomials.grid(F, 2n)
    PT,PF = plan_transform(T, (2n,2n), 1),plan_transform(F, (2n,2n), 2)
    𝐫ₐ = ClassicalOrthogonalPolynomials.grid(Tₐ, 2n)
    PTₐ = plan_transform(T, (2n,2n), 1)

    # Coefficients for the disk cell
    𝐱 = 𝐫 .* cos.(𝛉')
    𝐲 = 𝐫 .* sin.(𝛉')
    Fs = PT * (PF * rhs_xy.(𝐱, 𝐲))

    # Coefficients for the annulus cell
    𝐱ₐ = 𝐫ₐ .* cos.(𝛉')
    𝐲ₐ = 𝐫ₐ .* sin.(𝛉')
    Fsₐ = PTₐ * (PF * rhs_xy.(𝐱ₐ, 𝐲ₐ))

    X = zeros(2n+2, 2n)
    A = zeros(2n+3, 2n+3)
    # multiply RHS by r^2 and convert to C
    S = (R^2*M)[1:n,1:n]
    Sₐ = (Rₐ^2*Mₐ)[1:n,1:n]

    for j = 1:2n-1
        # Form matrix to be solved
        m = j ÷ 2
        Δₘ = L - m^2*M + λs[2] * R^2*M
        Δₘₐ = Lₐ - m^2*Mₐ + λs[1] * Rₐ^2*Mₐ

        # Boundary condition at r=1
        A[1,1:n+1] = Tₐ[[end],1:n+1]
        # Dirichlet element continuity row
        A[2, 1:end-1] = [Tₐ[[begin],:][1:n+1]; - T[[end],:][1:n+1]]'
        # Radial derivative continuity row
        A[3, 1:end-1] = [(Dₐ*Tₐ)[[begin],:][1:n+1]; -(D*T)[[end],:][1:n+1]]'
        # Annulus PDE
        A[4:n+3,1:n+1] = Δₘₐ[1:n,1:n+1]
        # Disk PDE
        A[n+4:end,n+2:end-1] = Δₘ[1:n,1:n+1]
        
        A[n+3,end] = 1. # tau-method stabilisation

        𝐛 = [0;0;0; Sₐ*Fsₐ[1:n,j]; S*Fs[1:n,j]]
        X[:,j] = (A \ 𝐛)[1:end-1]
    end

    return (X, Fsₐ, Fs)
end

# This function splits the Helmholtz solve via Two-band-Fourier into a series of
# one-dimensional solves per Fourier mode.
function twoband_fourier_helmholtz_modal_solve(UF, ΔLMR, rhs_xy::Function, n::Int, λ::V=0.0) where V

    (U, F) = UF
    @assert U isa TwoBandJacobi && U.a == U.b == U.c == 0
    @assert F isa Fourier

    (Δᵣ, L, M, R) = ΔLMR
    # Δᵣ = (r²*Δ): HalfWeighted{:ab}(TwoBandJacobi(ρ,1,1,0)) -> TwoBandJacobi(ρ,1,1,0)
    # L = I : TwoBandJacobi(ρ,0,0,0) -> TwoBandJacobi(ρ,1,1,0)
    # M = I : HalfWeighted{:ab}(TwoBandJacobi(ρ,1,1,0)) -> TwoBandJacobi(ρ,1,1,0)
    # R = r : TwoBandJacobi(ρ,1,1,0) -> TwoBandJacobi(ρ,1,1,0)

    # Need to expand via U as (for some reason) via C directly does not work.
    𝐫,𝛉 = ClassicalOrthogonalPolynomials.grid(U, n),ClassicalOrthogonalPolynomials.grid(F, n)
    PU, PF = plan_transform(U, (n,n), 1), plan_transform(F, (n,n), 2)

    𝐱 = 𝐫 .* cos.(𝛉')
    𝐲 = 𝐫 .* sin.(𝛉')

    Fs = PU * (PF * rhs_xy.(𝐱, 𝐲))

    X = zeros(n, n)
    # multiply RHS by r^2 and convert to C
    R² = (R*R*L)[1:n, 1:n]

    for j = 1:n
        m = j ÷ 2
        Δₘ = Δᵣ - m^2*M + λ*R*R*M^2*M
        X[:,j] = Δₘ[1:n,1:n] \ (R² * Fs[:,j])
    end

    return (X, Fs)
end

###
# SEM method of Zernike for the inner disk and Chebyshev-Fourier for the outer annulus.
###
function chebyshev_fourier_zernike_helmholtz_modal_solve(B::Vector, rhs_xy::Function, f::AbstractVector, ρ::TV, n::Int, Δ::Vector, L::Vector=[], λs::AbstractVector=[], mmode=1:n) where TV
    @assert length(B) == 2
    Z = B[2]
    @assert Z isa Zernike
    @assert B[1] isa Tuple
    (T, F) = B[1]
    @assert T.parent isa ChebyshevT
    @assert F isa Fourier

    Δs = Δ[2]
    Ls = L[2]

    @assert Δ[1] isa Tuple
    (Lₜ, M, R, D) = Δ[1]
    @assert D isa Derivative
    𝐫,𝛉 = ClassicalOrthogonalPolynomials.grid(T, 2n),ClassicalOrthogonalPolynomials.grid(F, 2n)
    PT,PF = plan_transform(T, (2n,2n), 1),plan_transform(F, (2n,2n), 2)

    # Coefficients for the annulus cell
    𝐱 = 𝐫 .* cos.(𝛉')
    𝐲 = 𝐫 .* sin.(𝛉')
    Fs = PT * (PF * rhs_xy.(𝐱, 𝐲))

    fs = [Fs, ModalTrav(f).matrix] # coefficients for disk and annulus cell

    # multiply RHS by r^2 and convert to C
    S = (R^2*M)[1:n,1:n]

    # Break down into Fourier modes
    zs = ModalTrav(Z[SVector(one(TV),zero(TV)), Block.(1:n+2)]).matrix

    N = size(fs[2],1) + 1
    us = zeros(N, 4N-3)

    ze(r,θ,n) = Z[SVector(r*cos(θ)/ρ,r*sin(θ)/ρ), Block.(1:n+2)]
    dze(r,θ,n) = derivative(r->ze(r,θ,n), r)
    dzs = ModalTrav(dze(ρ,0,n)).matrix

    X = zeros(n+1, 2n)
    for j in mmode
        # Form matrix to be solved
        mTF = j-1
        Δₘ = Lₜ - mTF^2*M + λs[1] * R^2*M

        m = length(j:2:n) # Length of Zernike system
        # Preallocate space for the matrix. Require n+1 for annulus, m+1 for the disk, +1 for the tau-method
        A = zeros((n+1)+(m+1)+1, (n+1)+(m+1)+1)

        # Boundary condition at r=1
        A[1,1:n+1] = T[[end],1:n+1]
        # Dirichlet element continuity row
        A[2, 1:end-1] = [T[[begin],:][1:n+1]; -zs[1:m+1,2j-1]]'
        # Radial derivative continuity row
        A[3, 1:end-1] = [(D*T)[[begin],:][1:n+1]; -dzs[1:m+1,2j-1]]'

        if iszero(λs)
            A[4:n+3,1:n+1] = Δₘ[1:n,1:n+1]
            # Do not forget inv(ρ^2) factor! To do with rescaling the Zernike polys
            A[n+4:end,n+2:end-1] = inv(ρ^2)*view(Δs[j], 1:m, 1:m+1)
        else
            A[4:n+3,1:n+1] = Δₘ[1:n,1:n+1]
            # Do not forget inv(ρ^2) factor! To do with rescaling the Zernike polys
            A[n+4:end,n+2:end-1] = inv(ρ^2)*view(Δs[j], 1:m, 1:m+1) + λs[2]*view(Ls[j], 1:m, 1:m+1)
        end

        A[2, end] = 1-exp(-j) # tau-method stabilisation
        A[n+3,end] = 1. # tau-method stabilisation

        if j == 1
            𝐛 = [0; 0; 0; S*fs[1][1:n,j]; fs[2][1:m,j]]
            u_ = A \ 𝐛
            X[:,j] = u_[1:n+1];
            us[1:m+1,j] = u_[n+2:end-1];
        else
            𝐛 = [0; 0; 0; S*fs[1][1:n,2j-2]; fs[2][1:m,2j-2]]
            u_ = A \ 𝐛
            X[:,2j-2] = u_[1:n+1]; us[1:m+1,2j-2] = u_[n+2:end-1]

            𝐛 = [0; 0; 0; S*fs[1][1:n,2j-1]; fs[2][1:m,2j-1]]
            u_ = A \ 𝐛
            X[:,2j-1] = u_[1:n+1]; us[1:m+1,2j-1] = u_[n+2:end-1]
        end

    end
    (X, ModalTrav(us))
end

function chebyshev_fourier_annulus_helmholtz_modal_solve(TF, LMR, Ds, rhs_xy::Function, n::Int, λs::Vector=[])

    (T, Tₐ, F) = TF
    @assert T.parent isa ChebyshevT
    @assert Tₐ.parent isa ChebyshevT
    @assert F isa Fourier

    (L, Lₐ, M, Mₐ, R, Rₐ) = LMR
    # L = (r²*Δ) : ChebyshevT -> Ultraspherical(2) (inner annulus)
    # M = I : ChebyshevT -> Ultraspherical(2) (inner annulus)
    # R = r : Ultraspherical(2) -> Ultraspherical(2) (inner annulus)
    # Lₐ = (r²*Δ) : ChebyshevT -> Ultraspherical(2) (outer annulus)
    # Mₐ = I : ChebyshevT -> Ultraspherical(2) (outer annulus)
    # Rₐ = r : Ultraspherical(2) -> Ultraspherical(2) (outer annulus)

    (D, Dₐ) = Ds
    @assert D isa Derivative
    @assert Dₐ isa Derivative

    𝐫,𝛉 = ClassicalOrthogonalPolynomials.grid(T, 2n),ClassicalOrthogonalPolynomials.grid(F, 2n)
    PT,PF = plan_transform(T, (2n,2n), 1),plan_transform(F, (2n,2n), 2)
    𝐫ₐ = ClassicalOrthogonalPolynomials.grid(Tₐ, 2n)
    PTₐ = plan_transform(T, (2n,2n), 1)

    # Coefficients for the inner annulus cell
    𝐱 = 𝐫 .* cos.(𝛉')
    𝐲 = 𝐫 .* sin.(𝛉')
    Fs = PT * (PF * rhs_xy.(𝐱, 𝐲))

    # Coefficients for the outer annulus cell
    𝐱ₐ = 𝐫ₐ .* cos.(𝛉')
    𝐲ₐ = 𝐫ₐ .* sin.(𝛉')
    Fsₐ = PTₐ * (PF * rhs_xy.(𝐱ₐ, 𝐲ₐ))

    X = zeros(2n+2, 2n)
    A = zeros(2n+3, 2n+3)
    # multiply RHS by r^2 and convert to C
    S = (R^2*M)[1:n-1,1:n-1]
    Sₐ = (Rₐ^2*Mₐ)[1:n,1:n]

    for j = 1:2n-1
        # Form matrix to be solved
        m = j ÷ 2
        Δₘ = L - m^2*M + λs[2] * R^2*M
        Δₘₐ = Lₐ - m^2*Mₐ + λs[1] * Rₐ^2*Mₐ

        # Boundary condition at r=1
        A[1,1:n+1] = Tₐ[[end],1:n+1]
        # Dirichlet element continuity row
        A[2, 1:end-1] = [Tₐ[[begin],:][1:n+1]; - T[[end],:][1:n+1]]'
        # Radial derivative continuity row
        A[3, 1:end-1] = [(Dₐ*Tₐ)[[begin],:][1:n+1]; -(D*T)[[end],:][1:n+1]]'
        # Inner bcs at r = α
        A[4,n+2:end-1] = T[[begin],:][1:n+1]
        # Outer annulus PDE
        A[5:n+4,1:n+1] = Δₘₐ[1:n,1:n+1]
        # Inner annulus PDE
        A[n+5:end,n+2:end-1] = Δₘ[1:n-1,1:n+1]

        A[n+3,end] = 1. # tau-method stabilisation

        𝐛 = [0;0;0;0;Sₐ*Fsₐ[1:n,j]; S*Fs[1:n-1,j]]
        X[:,j] = (A \ 𝐛)[1:end-1]
    end

    return (X, Fsₐ, Fs)
end

###
# SEM method of Chebyshev-Fourier for the inner annulus and ZernikeAnnulus for the outer annulus.
###
function zernikeannulus_chebyshev_fourier_helmholtz_modal_solve(B::Vector, rhs_xy::Function, f::AbstractVector, n::Int, Δ::Vector, L::Vector=[], λs::AbstractVector=[], mmode=1:n, w::Function=(r,m)->r^m)
    @assert length(B) == 2
    Z = B[1]
    TV = eltype(Z)
    @assert Z isa ZernikeAnnulus
    @assert B[2] isa Tuple
    (T, F) = B[2]
    @assert T.parent isa ChebyshevT
    @assert F isa Fourier

    Δs = Δ[1]
    Ls = L[1]

    @assert Δ[2] isa Tuple
    (Lₜ, M, R, D) = Δ[2]
    @assert D isa Derivative
    𝐫,𝛉 = ClassicalOrthogonalPolynomials.grid(T, 2n),ClassicalOrthogonalPolynomials.grid(F, 2n)
    PT,PF = plan_transform(T, (2n,2n), 1),plan_transform(F, (2n,2n), 2)

    # Coefficients for the annulus cell
    𝐱 = 𝐫 .* cos.(𝛉')
    𝐲 = 𝐫 .* sin.(𝛉')
    Fs = PT * (PF * rhs_xy.(𝐱, 𝐲))
    # ##### TODO remember to delete this.
    # return 1.0, Fs
    fs = [ModalTrav(f).matrix, Fs] # coefficients for disk and annulus cell

    # multiply RHS by r^2 and convert to C
    S = (R^2*M)[1:n-1,1:n-1]

    N = size(fs[1],1) + 1
    us = zeros(N, 4N-3)

    ρ = Z.ρ

    # Used to compute the entries to enforce continuity. Much faster
    # than evaluating ZernikeAnnulus directly.
    P = SemiclassicalJacobi.(inv(one(TV)-ρ^2), Z.b, Z.a, 0:n-1)
    Q = SemiclassicalJacobi.(inv(one(TV)-ρ^2), Z.b+1, Z.a+1, 1:n)

    Ds = Q .\ [Derivative(axes(Q[1], 1)) * P[j] for j in 1:lastindex(P)];

    X = zeros(n+1, 2n)
    for j in mmode
        # Form matrix to be solved
        mTF = j-1
        Δₘ = Lₜ - mTF^2*M + λs[2] * R^2*M

        m = length(j:2:n) # Length of ZernikeAnnulus system
        # Preallocate space for the matrix. Require m+1 for Zernike annulus, 
        # n+1 for the Chebyshev-Fourier annulus, +1 for the tau-method
        A = zeros((m+1)+(n+1)+1, (m+1)+(n+1)+1)

        c1 = P[j][begin, 1:m+1] # Boundary condition at r=1
        cρ = w(ρ, j-1) * P[j][end, 1:m+1] # Contiuity condition at r=ρ

        # Radial derivative continuity condition
        dcρ = (
            w(ρ, j-1) * 2ρ/(ρ^2-1) *(Q[j][end, 1:m+1]' * Ds[j][1:m+1, 1:m+1])'
            + derivative(r->w(r, j-1), ρ) * P[j][end, 1:m+1]
        )

        # Boundary condition at r=1
        A[1,1:m+1] = c1
        # Dirichlet element continuity row
        A[2, 1:end-1] = [cρ; -T[[end],:][1:n+1]]'
        # Radial derivative continuity row
        A[3, 1:end-1] = [dcρ;-(D*T)[[end],:][1:n+1]]'
        # Boundary condition at r = α
        A[4,m+2:end-1] = T[[begin],:][1:n+1]

        if iszero(λs)
            A[5:m+4,1:m+1] = view(Δs[j],1:m,1:m+1)
            A[m+5:end,m+2:end-1] = Δₘ[1:n-1,1:n+1]
        else
            A[5:m+4,1:m+1] = view(Δs[j], 1:m, 1:m+1) + λs[1]*view(Ls[j], 1:m, 1:m+1)
            A[m+5:end,m+2:end-1] = Δₘ[1:n-1,1:n+1]
        end

        A[2, end] = 1-exp(-sqrt(sqrt(j))) # tau-method stabilisation
        A[m+3,end] = 1. # tau-method stabilisation
        # A[end,end] = 1.

        if j == 1
            𝐛 = [0; 0; 0; 0; fs[1][1:m,j]; S*fs[2][1:n-1,j]]
            u_ = A \ 𝐛
            us[1:m+1,j] = u_[1:m+1];
            X[:,j] = u_[m+2:end-1];
        else
            𝐛 = [0; 0; 0; 0; fs[1][1:m,2j-2]; S*fs[2][1:n-1,2j-2]; ]
            u_ = A \ 𝐛
            X[:,2j-2] = u_[m+2:end-1]; us[1:m+1,2j-2] =u_[1:m+1]

            𝐛 = [0; 0; 0; 0; fs[1][1:m,2j-1]; S*fs[2][1:n-1,2j-1]]
            u_ = A \ 𝐛
            X[:,2j-1] = u_[m+2:end-1]; us[1:m+1,2j-1] = u_[1:m+1]
        end

    end
    (ModalTrav(us), X)
end

# Archived for the future and not used in the examples.
# This solves the Helmholtz problem via least-squares not using a tau-method.
# function modal_solve_LS(P, f, Δs, L, b, ρ, mmode=1:b, w=(x,m)->r^m)
#     fs = [ModalTrav(f[j]).matrix for j in 1:2]

#     zs = ModalTrav(Z[2][SVector(1,0.), Block.(1:b+2)]).matrix

#     # These two calls are slow, so we directly lookup the values via SemiclassicalJacobi
#     # zsₐ1 = ModalTrav(Z[1][SVector(1,0.), Block.(1:b+2)]).matrix
#     # zsₐρ = ModalTrav(Z[1][SVector(ρ,0.), Block.(1:b+2)]).matrix

#     n = size(fs[1],1)
#     us = zeros(2n+2, 4(n+1)-3)

#     # First Fourier mode m = 0
#     c1 = P[1][begin, 1:n+1]
#     cρ = P[1][end, 1:n+1]

#     Q = SemiclassicalJacobi.(inv(1-ρ^2), Z[1].b+1, Z[1].a+1, 1:b);
#     Ds = Q .\ [Derivative(axes(Q[1], 1)) * P[j] for j in 1:lastindex(P)];

#     # zDs = Derivative(axes(Jacobi(1,0),1)) * Normalized(Jacobi(1 , 0:b-1))
#     ze(r,θ,b) = Z[2][SVector(r*cos(θ)/ρ,r*sin(θ)/ρ), Block.(1:b+2)]
#     dze(r,θ,b) = derivative(r->ze(r,θ,b), r)
#     dzs = ModalTrav(dze(ρ,0,b)).matrix

#     # Loop over the rest of the Fourier modes
#     residuals = []
#     for j in mmode
#         # j = 1;
#         m = length(j:2:b)
#         A = zeros(2m+3, 2m+2)
#         c1 = P[j][begin, 1:m+1]
#         cρ = w(ρ, j-1) * P[j][end, 1:m+1]

#         dcρ = (
#             w(ρ, j-1) * 2ρ/(ρ^2-1) *(Q[j][end, 1:m+1]' * Ds[j][1:m+1, 1:m+1])'
#             + derivative(r->w(r, j-1), ρ) * P[j][end, 1:m+1]
#         )

#         A[1,1:m+1] = c1                    # boundary condition
#         A[2,:] = [cρ' -zs[1:m+1,2j-1]']    # Dirichlet element continuity
#         A[3,:] = [dcρ' -dzs[1:m+1,2j-1]']  # Neumann element continuity

#         A[4:m+3,1:m+1] = Δs[1][j][1:m,1:m+1]
#         # Do not forget inv(ρ^2) factor! To do with rescaling the Zernike polys
#         A[m+4:end,m+2:end] = inv(ρ^2)*Δs[2][j][1:m,1:m+1]

#         if j == 1
#             𝐛 = [0; 0; 0; fs[1][1:m,1]; fs[2][1:m,1]]
#             us[:,1] = A \ 𝐛
#             append!(residuals, norm(A*us[:,1]-𝐛))
#         else
#             𝐛 = [0; 0; 0; fs[1][1:m,2j-2]; fs[2][1:m,2j-2]]
#             u_ = A \ 𝐛
#             append!(residuals, norm(A*u_-𝐛))
#             us[1:m+1,2j-2] = u_[1:m+1]; us[n+2:n+m+2,2j-2] = u_[m+2:end]

#             𝐛 = [0; 0; 0; fs[1][1:m,2j-1]; fs[2][1:m,2j-1]]
#             u_ = A \ 𝐛
#             append!(residuals, norm(A*u_-𝐛))
#             us[1:m+1,2j-1] = u_[1:m+1]; us[n+2:n+m+2,2j-1] = u_[m+2:end]
#         end

#     end
#     ((ModalTrav(us[1:n+1, :]), ModalTrav(us[n+2:end, :])), residuals)
# end