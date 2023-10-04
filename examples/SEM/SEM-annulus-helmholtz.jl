using AnnuliPDEs, ClassicalOrthogonalPolynomials, AnnuliOrthogonalPolynomials
using PyPlot, Plots, LaTeXStrings
using DelimitedFiles
"""
This script implements the Helmholtz example of

"Discontinuous variable coefficients and data on an annulus" (section 7.4).

We are solving (Δ + λ(x,y)) u(x,y) = ∑ᵢ dᵢ exp(−aᵢ((x−bᵢ)²+(y−cᵢ)²)) * (−4aᵢ * (−aᵢ(bᵢ² − 2bᵢx + cᵢ² − 2cᵢy + x² + y²) + 1)))
on a annulus with inradius 0.1.

Here we have a right-hand side that has a jump in the radial direction at r=1/2.

The solution is continuous. Hence we use a spectral element method to
split the annulus domain into an inner annulus cell and an outer annulus cell with
inradii at r=1/10 and r=1/2, respectively.

We consider three methods
1. Zernike annular discretisation on both cells.
2. Chebyshev-Fourier on the inner cell & Zernike annular on the outer cell.
3. Chebyshev-Fourier on both cells.
"""

# ρ is the inner radius on the annulus domain
# α is the radius where the two cells intersect and where the radial discontinuity occurs.
# κ₀ is the jump coefficient on the inner cell
# κ₁ is the jump coefficient on the outer cell
ρ, α, κ₀, κ₁ = 0.1, 0.5, 1e2, 1e0

# Gaussian bumps
θs = [0, π/2, π/3, 5π/4]
c = zeros(length(θs)+1,4)
for i in 1:lastindex(θs) c[i,:] = [1.0 -10*i α*cos(θs[i]) α*sin(θs[i])] end
c[end,:] = [1e1 -80 0.95*cos(π) 0.95*sin(π)] # 80 for Poisson

function gbump(r, θ)
    x = r*cos(θ); y = r*sin(θ)
    s = 0;
    for j in axes(c,1)
        s += c[j,1] * exp(c[j,2]*((c[j,3] - x)^2 + (c[j,4] - y)^2))
    end
    s
end

# Coefficients for the RHS to form a known continuous solution
c1 = ((κ₀ - κ₁) * α^2 / 4 + (κ₁ - κ₀*ρ^2) / 4) / log(ρ) + (κ₁ - κ₀) * α^2 / 2 * (log(α) - log(ρ)) / log(ρ)
c0 = (κ₁ - κ₀) * α^2 /2 + c1
d0 = -κ₀*ρ^2 / 4 - c0*log(ρ)
d1 = -κ₁/4

jump(r) = r ≤ α ? κ₀*r^2 / 4 + c0*log(r) + d0 : κ₁*r^2 / 4 + c1*log(r) + d1

# Exact solution
ua(r,θ) = gbump(r,θ) * jump(r)

# Variable Hemholtz coefficient
κ(r) = r < α ? κ₀ : κ₁
λs = [κ(α+0.1); κ(α-0.1)]

# Use ForwardDiff to compute the RHS
rhs_(r,θ) =  (derivative(r->derivative(r->ua(r,θ), r),r) 
    + derivative(r->ua(r,θ), r)/r 
    + derivative(θ->derivative(θ->ua(r,θ), θ),θ)/r^2
    + κ(r) * ua(r,θ)
)

# RHS in Cartesian coordinates
function rhs_xy(x, y)
    r = sqrt(x^2 + y^2); θ = atan(y, x)
    rhs_(r,θ)
end

# Scaled RHS for the inner annulus element
function rhs_xy_scale(x, y)
    r = sqrt(x^2 + y^2); θ = atan(y, x)
    r̃ = α * r
    rhs_(r̃,θ)
end

###
# (2-element) Zernike annular on both cells
###

# Cells listed from outwards to inwards
Z = [ZernikeAnnulus(α, 0, 0), ZernikeAnnulus(ρ/α, 0, 0)]
Zd = [ZernikeAnnulus(α, 2, 2), ZernikeAnnulus(ρ/α, 2, 2)]

Δ =  [Zd[1] \ (Laplacian(axes(Z[1],1)) * Z[1]), 
        Zd[2]\ (Laplacian(axes(Z[2],1)) * Z[2])
];

L =  [Zd[1] \ Z[1], 
        Zd[2]\ Z[2]
];

Δs = [Δ[i].ops for i in 1:2];
Ls = [L[i].ops for i in 1:2];

# Preallocating speeds things up
[Δs[i][170] for i in 1:2];
[Ls[i][170] for i in 1:2];

xyₐ = axes(Z[1],1); xₐ,yₐ = first.(xyₐ),last.(xyₐ)
xy = axes(Z[2],1); x,y = first.(xy),last.(xy)
x = [xₐ, x];
y = [yₐ, y];

# Oversampled grid for computing pointwise errors on.
Nn = 1000
G = AnnuliOrthogonalPolynomials.grid.(Z, Block(Nn))
# Exact solution evaluated on the grid
U = [ua.(get_rs.(G[1]), get_θs.(G[1]))]
append!(U, [ua.(α*get_rs.(G[2]), get_θs.(G[2]))])

errors_Z_2 = []
u = []
f = []
for n in 11:10:201
    f = []
    # Expand RHS in Zernike annular polynomials for annulus element
    append!(f,[Zd[1][:, Block.(1:n)] \ rhs_xy.(x[1], y[1])])
    # Expand RHS in Zernike polynomials for the disk element
    append!(f, [Zd[2][:, Block.(1:n)] \ rhs_xy_scale.(x[2], y[2])])

    # Solve by breaking down into solves for each Fourier mode.
    # We utilise a tau-method to enforce the boundary conditions
    # and continuity.
    @time u = zernike_modal_solve(Z, f, n, Δs, Ls, λs)

    print("Computed coefficients for n=$n \n")
    collect_errors(Z, u, U, errors_Z_2)
end
writedlm("errors_Z_2.log", errors_Z_2)

# Plot the right-hand side
plot_solution(Zd, (ModalTrav(f[1]), ModalTrav(f[2])), inner_radial=ρ, ttl=L"$f(x,y)$")
PyPlot.savefig("spectral-element-f.pdf")
# Plot the solution
plot_solution(Z, u, inner_radial=ρ)
PyPlot.savefig("spectral-element-u.pdf")


###
# (2-element) Chebyshev-Fourier series on both cells
###
T,C,Tₐ,Cₐ,F = chebyshevt(ρ..α),ultraspherical(2, ρ..α),chebyshevt(α..1),ultraspherical(2, α..1),Fourier()
r,rₐ = axes(T,1), axes(Tₐ,1)
D,Dₐ = Derivative(r), Derivative(rₐ)

L = C \ (r.^2 .* (D^2 * T)) + C \ (r .* (D * T)) # r^2 * ∂^2 + r*∂
M = C\T # Identity
R = C \ (r .* C) # mult by r

Lₐ = Cₐ \ (rₐ.^2 .* (Dₐ^2 * Tₐ)) + Cₐ \ (rₐ .* (Dₐ * Tₐ)) # r^2 * ∂^2 + r*∂
Mₐ = Cₐ\Tₐ # Identity
Rₐ = Cₐ \ (rₐ .* Cₐ) # mult by r

errors_TF_2 = []
for n in 11:10:111
    # Compute coefficients of solution to Helmholtz problem with Chebyshev-Fourier series
    X, _ = chebyshev_fourier_annulus_helmholtz_modal_solve((T, Tₐ, F), (L, Lₐ, M, Mₐ, R, Rₐ), (D, Dₐ), rhs_xy, n, λs) 
    print("Computed coefficients for n=$n \n")
    collect_errors((T,Tₐ,F,ρ,α), X, U, G, errors_TF_2)
end
writedlm("errors_TF_2.log", errors_TF_2)

###
# (2-element) Chebyshev-Fourier on inner cell, Zernike annular on outer cell.
###
T,C,F = chebyshevt(ρ..α),ultraspherical(2, ρ..α),Fourier()
r = axes(T,1)
D = Derivative(r)

Lₜ = C \ (r.^2 .* (D^2 * T)) + C \ (r .* (D * T)) # r^2 * ∂^2 + r*∂
M = C\T # Identity
R = C \ (r .* C) # mult by r

Z = ZernikeAnnulus(α,0,0)
Zd = ZernikeAnnulus(α,2,2)

Δ =   Zd \ (Laplacian(axes(Z,1)) * Z);
L =  Zd \ Z;

Δs = Δ.ops;
Ls = L.ops;
Δs[111]; # Pre-allocation speeds things up
Ls[111];

xy = axes(Z,1); x,y = first.(xy),last.(xy)

errors_TF_Z = []
for n in 11:10:111
    # Expand RHS in Zernike annular polynomials for the outer annulus element
    f = Zd[:, Block.(1:n)] \ rhs_xy.(x, y)
    # Solve by breaking down into solves for each Fourier mode.
    # We utilise a tau-method to enforce the boundary conditions
    # and continuity.
    u, X = zernikeannulus_chebyshev_fourier_helmholtz_modal_solve([Z, (T,F)], rhs_xy, f, n, [Δs, (Lₜ, M, R, D)], [Ls, []], λs)
    print("Computed coefficients for n=$n \n")
    collect_errors((Z,T,F,ρ,α), (u,X), U, G, errors_TF_Z)

end
writedlm("errors_TF_Z.log", errors_TF_Z)

###
# Convergence plots
###

errors_Z_2 = readdlm("errors_Z_2.log")
errors_TF_Z = readdlm("errors_TF_Z.log")
errors_TF_2 = readdlm("errors_TF_2.log")

tfns = [2*(2b-1)*(b+1) for b in 11:10:111]
Plots.plot(tfns, maximum.(errors_TF_2),
    label=L"\mathrm{Chebyshev} \otimes \mathrm{Fourier/Chebyshev} \otimes \mathrm{Fourier \,\, (2 \,\, elements)}",
    linewidth=2,
    markershape=:dtriangle,
    markersize=5
)

tfns = [(2b-1)*(b+1) + sum(1:b) for b in 11:10:111]
Plots.plot!(tfns, maximum.(errors_TF_Z),
    label=L"\mathrm{Chebyshev} \otimes \mathrm{Fourier/Zernike \,\, annular \,\, (2 \,\, elements)}",
    linewidth=2,
    markershape=:diamond,
    markersize=5,

    ylabel=L"$l^\infty\mathrm{-norm \;\; error}$",
    xlabel=L"$\# \mathrm{Basis \; functions}$",
    ylim=[1e-15, 9e0],
    xlim = [0, 6e4],
    legend=:topright,
    xtickfontsize=10, ytickfontsize=10,xlabelfontsize=15,ylabelfontsize=15,
    yscale=:log10,
    yticks=[1e-15, 1e-10, 1e-5, 1e0],
    gridlinewidth=2
)

ns = [2*sum(1:b) for b in 11:10:211]
Plots.plot!(ns, errors_Z_2,
    label=L"\mathrm{Zernike \,\, annular /Zernike \,\, annular \,\, (2 \,\,elements)}",
    linewidth=2,
    marker=:dot,
    markersize=5,

    ylabel=L"$l^\infty\mathrm{-norm \;\; error}$",
    xlabel=L"$\# \mathrm{Basis \; functions}$",
    ylim=[1e-15, 5e0],
    xlim = [0, 5.5e4],
    legend=:topright,
    xtickfontsize=10, ytickfontsize=10,xlabelfontsize=15,ylabelfontsize=15,
    yscale=:log10,
    yticks=[1e-15, 1e-10, 1e-5, 1e0],
    gridlinewidth=2
)

Plots.savefig("SEM-annulus-zernike-zernike-poisson.pdf")