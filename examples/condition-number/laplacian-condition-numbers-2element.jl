using AnnuliPDEs, ClassicalOrthogonalPolynomials, LinearAlgebra
using AnnuliOrthogonalPolynomials, SemiclassicalOrthogonalPolynomials
using Plots, LaTeXStrings

"""
This script plots the condition number of the 2-element Laplacian
and matrix sizes for increasing Fourier mode at truncation degree N=100 
for Chebyshev-Fourier series and Zernike annular polynomials.
"""

ρ = 0.5; n = 101; ms = 0:10:100;

T,C,Tₐ,Cₐ,F = chebyshevt(0..ρ),ultraspherical(2, 0..ρ),chebyshevt(ρ..1),ultraspherical(2, ρ..1),Fourier()
r,rₐ = axes(T,1), axes(Tₐ,1)
D,Dₐ = Derivative(r), Derivative(rₐ)

L = C \ (r.^2 .* (D^2 * T)) + C \ (r .* (D * T)) # r^2 * ∂^2 + r*∂
M = C\T # Identity
R = C \ (r .* C) # mult by r

Lₐ = Cₐ \ (rₐ.^2 .* (Dₐ^2 * Tₐ)) + Cₐ \ (rₐ .* (Dₐ * Tₐ)) # r^2 * ∂^2 + r*∂
Mₐ = Cₐ\Tₐ # Identity
Rₐ = Cₐ \ (rₐ .* Cₐ) # mult by r


cs = []
for m in ms
# m = 0
    Δₘ = L - m^2*M #+ λs[2] * R^2*M
    Δₘₐ = Lₐ - m^2*Mₐ #+ λs[1] * Rₐ^2*Mₐ
    A = zeros(2n+3, 2n+3)
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
    dg = Diagonal(inv.(diag(A))); dg[dg .== Inf] .= 1
    append!(cs, [cond(A)])
end


## Zernike 2-element condition number
T = Float64
Z = [ZernikeAnnulus(ρ, 0, 0), Zernike(0,0)]
Zd = [ZernikeAnnulus(ρ, 2, 2), Zernike(0,2)]

Δ =  [Zd[1] \ (Laplacian(axes(Z[1],1)) * Z[1]), 
        Zd[2]\ (Laplacian(axes(Z[2],1)) * Z[2])
];
Δs = [Δ[j].ops for j in 1:2]


ρ = Z[1].ρ
b=101;
# Used to compute the entries to enforce continuity. Much faster
# than evaluating ZernikeAnnulus directly.
P = SemiclassicalJacobi.(inv(one(T)-ρ^2), Z[1].b, Z[1].a, 0:b-1)
Q = SemiclassicalJacobi.(inv(1-ρ^2), Z[1].b+1, Z[1].a+1, 1:b)

# Break down into Fourier modes
zs = ModalTrav(Z[2][SVector(one(T),zero(T)), Block.(1:b+2)]).matrix

# For enforcing the radial derivative continuity
Ds = Q .\ [Derivative(axes(Q[1], 1)) * P[j] for j in 1:lastindex(P)];

ze(r,θ,b) = Z[2][SVector(r*cos(θ)/ρ,r*sin(θ)/ρ), Block.(1:b+2)]
dze(r,θ,b) = derivative(r->ze(r,θ,b), r)
dzs = ModalTrav(dze(ρ,0,b)).matrix
w(r,m) = r^m

ds = []
js = []
# Loop over Fourier modes
for j in (ms.+1)
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
    
    A[4:m+3,1:m+1] = Δs[1][j][1:m,1:m+1]
    # Do not forget inv(ρ^2) factor! To do with rescaling the Zernike polys
    A[m+4:end,m+2:end-1] = inv(ρ^2)*Δs[2][j][1:m,1:m+1]
    A[2, end] = 1-exp(-j) # tau-method stabilisation
    A[m+3,end] = 1. # tau-method stabilisation
    append!(ds, [cond(A)])
    append!(js, [size(A,1)])
end


Plots.plot(ms, [cs ds],
    label=["Chebyshev-Fourier" "Zernike annular"],
    linewidth=2,
    markershape=[:circle :dtriangle],
    markersize=5,

    legend=:topright,
    ylabel="Condition number",
    xlabel=L"$m$",
    yscale=:log10,
    xtickfontsize=10, ytickfontsize=10,xlabelfontsize=15,ylabelfontsize=15,
    yticks=[1e5,1e7,1e9,1e11],
    # ylim=[1e1,1e9]
)

Plots.savefig("laplacian-condition-numbers-2-element.pdf")

Plots.plot(ms, [(2n+2)*ones(length(ms)) js.^1],
    label=["Chebyshev-Fourier" "Zernike annular"],
    linewidth=2,
    markershape=[:circle :dtriangle],
    markersize=5,

    legend=:bottomleft,
    ylabel="Matrix size",
    xlabel=L"$m$",
    yscale=:log10,
    xtickfontsize=10, ytickfontsize=10,xlabelfontsize=15,ylabelfontsize=15,
    yticks=[1e0, 1e1, 1e2, 1e3],
    # ylim=[3e0,5e2]
)

Plots.savefig("laplacian-matrix-size-2-element.pdf")