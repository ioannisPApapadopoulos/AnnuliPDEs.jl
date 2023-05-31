"""
Solver routines
"""

# This function splits the Helmholtz solve via Zernike (annular) polys into a series of
# one-dimensional solves per Fourier mode with decreasing size.
function helmholtz_modal_solve(f::BlockVector, b::Int, Δ::MultivariateOrthogonalPolynomials.ModalInterlace, L::MultivariateOrthogonalPolynomials.ModalInterlace, λ::T=0.0, mmode=2:b) where T 
    Δs = Δ.ops
    Ls = L.ops
    
    fs = ModalTrav(f).matrix
    N = size(fs,1)
    us = zeros(N, size(fs,2))
    
    us[:,1] = (Δs[1])[1:N,1:N] \ fs[1:N,1]
    for j in mmode
        M = length(j:2:b)
        if λ != 0
            us[1:M,2j-2] = (Δs[j]+λ*Ls[j])[1:M,1:M] \ fs[1:M,2j-2]
            us[1:M,2j-1] = (Δs[j]+λ*Ls[j])[1:M,1:M] \ fs[1:M,2j-1]
        else
            us[1:M,2j-2] = (Δs[j])[1:M,1:M] \ fs[1:M,2j-2]
            us[1:M,2j-1] = (Δs[j])[1:M,1:M] \ fs[1:M,2j-1]
        end
    end
    ModalTrav(us)[Block.(1:b)]
end


# This function splits the Helmholtz solve via Zernike (annular) polys into a series of
# one-dimensional solves per Fourier mode with decreasing size. This is adapted for
# the spectral element method and also implements a tau-method for continuity
# across the disk and annulus cell.
function helmholtz_modal_solve(f::Vector, b::Int, ρ::T, Δ::Vector{ModalInterlace}, L::Vector{ModalInterlace}=[], λ::T=[], mmode=1:b, w::Function=(x,m)->r^m) where T
    Δs = Δ.ops
    Ls = L.ops
    
    P = SemiclassicalJacobi.(inv(one(T)-ρ^2), Z[1].b, Z[1].a, 0:b-1)

    fs = [ModalTrav(f[j]).matrix for j in 1:2]

    zs = ModalTrav(Z[2][SVector(1,0.), Block.(1:b+2)]).matrix

    n = size(fs[1],1)
    us = zeros(2n+2, 4(n+1)-3)

    # First Fourier mode m = 0
    c1 = P[1][begin, 1:n+1]
    cρ = P[1][end, 1:n+1]

    Q = SemiclassicalJacobi.(inv(1-ρ^2), Z[1].b+1, Z[1].a+1, 1:b);
    Ds = Q .\ [Derivative(axes(Q[1], 1)) * P[j] for j in 1:lastindex(P)];
    
    # zDs = Derivative(axes(Jacobi(1,0),1)) * Normalized(Jacobi(1 , 0:b-1))
    ze(r,θ,b) = Z[2][SVector(r*cos(θ)/ρ,r*sin(θ)/ρ), Block.(1:b+2)]
    dze(r,θ,b) = derivative(r->ze(r,θ,b), r)
    dzs = ModalTrav(dze(ρ,0,b)).matrix

    # Loop over the rest of the Fourier modes
    for j in mmode
        # j = b-3;
        m = length(j:2:b)
        A = zeros(2m+3, 2m+3)
        c1 = P[j][begin, 1:m+1]
        cρ = w(ρ, j-1) * P[j][end, 1:m+1]

        dcρ = (
            w(ρ, j-1) * 2ρ/(ρ^2-1) *(Q[j][end, 1:m+1]' * Ds[j][1:m+1, 1:m+1])'
            + derivative(r->w(r, j-1), ρ) * P[j][end, 1:m+1]
        )

        A[1,1:m+1] = c1                    # boundary condition
        A[2,1:end-1] = [cρ' -zs[1:m+1,2j-1]']    # Dirichlet element continuity
        A[3,1:end-1] = [dcρ' -dzs[1:m+1,2j-1]']  # Neumann element continuity
        
        A[4:m+3,1:m+1] = Δs[1][j][1:m,1:m+1]
        # Do not forget inv(ρ^2) factor! To do with rescaling the Zernike polys
        A[m+4:end,m+2:end-1] = inv(ρ^2)*Δs[2][j][1:m,1:m+1]
        
        A[2, end] = 1-exp(-j)
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

# Archived for the future and not used in the examples.
# This solves the Helmholtz problem via least-squares not using a tau-method.
function modal_solve_LS(P, f, Δs, L, b, ρ, mmode=1:b, w=(x,m)->r^m)    
    fs = [ModalTrav(f[j]).matrix for j in 1:2]

    zs = ModalTrav(Z[2][SVector(1,0.), Block.(1:b+2)]).matrix

    # These two calls are slow, so we directly lookup the values via SemiclassicalJacobi
    # zsₐ1 = ModalTrav(Z[1][SVector(1,0.), Block.(1:b+2)]).matrix
    # zsₐρ = ModalTrav(Z[1][SVector(ρ,0.), Block.(1:b+2)]).matrix

    n = size(fs[1],1)
    us = zeros(2n+2, 4(n+1)-3)

    # First Fourier mode m = 0
    c1 = P[1][begin, 1:n+1]
    cρ = P[1][end, 1:n+1]

    Q = SemiclassicalJacobi.(inv(1-ρ^2), Z[1].b+1, Z[1].a+1, 1:b);
    Ds = Q .\ [Derivative(axes(Q[1], 1)) * P[j] for j in 1:lastindex(P)];
    
    # zDs = Derivative(axes(Jacobi(1,0),1)) * Normalized(Jacobi(1 , 0:b-1))
    ze(r,θ,b) = Z[2][SVector(r*cos(θ)/ρ,r*sin(θ)/ρ), Block.(1:b+2)]
    dze(r,θ,b) = derivative(r->ze(r,θ,b), r)
    dzs = ModalTrav(dze(ρ,0,b)).matrix

    # Loop over the rest of the Fourier modes
    residuals = []
    for j in mmode
        # j = 1;
        m = length(j:2:b)
        A = zeros(2m+3, 2m+2)
        c1 = P[j][begin, 1:m+1]
        cρ = w(ρ, j-1) * P[j][end, 1:m+1]

        dcρ = (
            w(ρ, j-1) * 2ρ/(ρ^2-1) *(Q[j][end, 1:m+1]' * Ds[j][1:m+1, 1:m+1])'
            + derivative(r->w(r, j-1), ρ) * P[j][end, 1:m+1]
        )

        A[1,1:m+1] = c1                    # boundary condition
        A[2,:] = [cρ' -zs[1:m+1,2j-1]']    # Dirichlet element continuity
        A[3,:] = [dcρ' -dzs[1:m+1,2j-1]']  # Neumann element continuity
        
        A[4:m+3,1:m+1] = Δs[1][j][1:m,1:m+1]
        # Do not forget inv(ρ^2) factor! To do with rescaling the Zernike polys
        A[m+4:end,m+2:end] = inv(ρ^2)*Δs[2][j][1:m,1:m+1]
        
        if j == 1
            𝐛 = [0; 0; 0; fs[1][1:m,1]; fs[2][1:m,1]]
            us[:,1] = A \ 𝐛
            append!(residuals, norm(A*us[:,1]-𝐛))
        else
            𝐛 = [0; 0; 0; fs[1][1:m,2j-2]; fs[2][1:m,2j-2]]
            u_ = A \ 𝐛
            append!(residuals, norm(A*u_-𝐛))
            us[1:m+1,2j-2] = u_[1:m+1]; us[n+2:n+m+2,2j-2] = u_[m+2:end]

            𝐛 = [0; 0; 0; fs[1][1:m,2j-1]; fs[2][1:m,2j-1]]
            u_ = A \ 𝐛 
            append!(residuals, norm(A*u_-𝐛))
            us[1:m+1,2j-1] = u_[1:m+1]; us[n+2:n+m+2,2j-1] = u_[m+2:end]
        end
        
    end
    ((ModalTrav(us[1:n+1, :]), ModalTrav(us[n+2:end, :])), residuals)
end

# This function splits the Helmholtz solve via Chebyshev-Fourier into a series of
# one-dimensional solves per Fourier mode.
function chebyshev_fourier_helmholtz_modal_solve(TF, LMR, rhs_xy::Function, n::Int, λ::V=0.0) where V  

    (T, F) = TF
    (L, M, R) = LMR

    𝐫,𝛉 = ClassicalOrthogonalPolynomials.grid(T, n),ClassicalOrthogonalPolynomials.grid(F, n)
    PT, PF = plan_transform(T, (n,n), 1), plan_transform(F, (n,n), 2)

    𝐱 = 𝐫 .* cos.(𝛉')
    𝐲 = 𝐫 .* sin.(𝛉')

    Fs = PT * (PF * rhs_xy.(𝐱, 𝐲))

    X = zeros(n+2, n+2)
    # multiply RHS by r^2 and convert to C
    S = (R^2*M)[1:n,1:n]

    for j = 1:n
        m = j ÷ 2
        Δₘ = L - m^2*M + λ*R^2*M
        X[:,j] = [T[[begin,end],:]; Δₘ][1:n+2,1:n+2] \ [0; 0; S*Fs[:,j]]
    end

    return (X, Fs)
end

# This function splits the Helmholtz solve via Two-band-Fourier into a series of
# one-dimensional solves per Fourier mode.
function twoband_fourier_helmholtz_modal_solve(UF, ΔLMR, rhs_xy::Function, n::Int, λ::V=0.0) where V

    (U, F) = UF
    (Δᵣ, L, M, R) = ΔLMR

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