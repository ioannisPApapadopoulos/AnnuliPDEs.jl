using AnnuliPDEs, ClassicalOrthogonalPolynomials, AlgebraicCurveOrthogonalPolynomials, MultivariateOrthogonalPolynomials
using PyPlot, Plots, LaTeXStrings

"""
This script implements the Helmholtz example of

"Discontinuous variable coefficients and data" (section 7.5).

We are solving (Δ+κ(r)) u(x,y) = exp(−a((x−b)²+(y−c)²)) * (−4a * (−a(b² − 2bx + c² − 2cy + x² + y²) + 1)) + κ(r))
on a disk.

Here we have a variable coefficient where κ(r) has a jump in the radial direction at r=1/2.

Hence, the right-hand side also has a jump in the radial direction at r=1/2.

The solution is continuous. Hence we use a spectral element method to
split the disk domain into an inner disk cell and an outer annulus cell with
inradius at r=1/2. This performs particularly well.

"""

ρ = 0.5; κ₀ = 1e2; κ₁ = 1e0; 
# Exact solution

# Gaussian bump
c = [-20 0 ρ]
function gbump(r, θ)
    x = r*cos(θ); y = r*sin(θ)
    s = 0;
    for j in axes(c,1)
        s += exp(c[j,1]*((c[j,2] - x)^2 + (c[j,3] - y)^2))
    end
    s
end

# jump in radial direction, κ(r) in the manuscript
coeff_jump(r, θ) = r ≤ ρ ? κ₀ : κ₁
# continuous jump
jump(r,θ) = r ≤ ρ ? (κ₀*r^2/4 + (κ₁ - κ₀)*ρ^2/4 - κ₁/4 + (κ₀ - κ₁)/2*ρ^2*log(ρ)) : (κ₁*r^2/4 - κ₁/4 + (κ₀ - κ₁)/2*ρ^2*log(r))

# Exact solution
ua(r,θ) = gbump(r,θ) * jump(r,θ)

# Variable discontinuous coefficient
λ(r) = coeff_jump(r,0)

# Use ForwardDiff to compute the RHS
rhs_(r,θ) =  (derivative(r->derivative(r->ua(r,θ), r),r) 
    + derivative(r->ua(r,θ), r)/r 
    + derivative(θ->derivative(θ->ua(r,θ), θ),θ)/r^2
    + λ(r)*ua(r,θ)
)

# RHS in Cartesian coordinates
function rhs_xy(x, y)
    r = sqrt(x^2 + y^2); θ = atan(y, x)
    rhs_(r,θ)
end

# Scaled RHS for the disk element
function rhs_xy_scale(x, y)
    r = sqrt(x^2 + y^2); θ = atan(y, x)
    r̃ = ρ * r
    rhs_(r̃,θ)
end

λs = [λ(0.8); λ(0.2)]

###
# Spectral element Chebyshev-Fourier series discretisation
###
T,C,Tₐ,Cₐ,F = chebyshevt(0..ρ),ultraspherical(2, 0..ρ),chebyshevt(ρ..1),ultraspherical(2, ρ..1),Fourier()
r,rₐ = axes(T,1), axes(Tₐ,1)
D,Dₐ = Derivative(r), Derivative(rₐ)

L = C \ (r.^2 .* (D^2 * T)) + C \ (r .* (D * T)) # r^2 * ∂^2 + r*∂
M = C\T # Identity
R = C \ (r .* C) # mult by r

Lₐ = Cₐ \ (rₐ.^2 .* (Dₐ^2 * Tₐ)) + Cₐ \ (rₐ .* (Dₐ * Tₐ)) # r^2 * ∂^2 + r*∂
Mₐ = Cₐ\Tₐ # Identity
Rₐ = Cₐ \ (rₐ .* Cₐ) # mult by r

errors_TF = []
for n in 11:10:111
    # Compute coefficients of solution to Helmholtz problem with Chebyshev-Fourier series
    X, _ = chebyshev_fourier_helmholtz_modal_solve((T, Tₐ, F), (L, Lₐ, M, Mₐ, R, Rₐ), (D, Dₐ), rhs_xy, n, λs) 
    print("Computed coefficients for n=$n \n")

    collect_errors((T,Tₐ,F,ρ), X, ua, errors_TF)
end

###
# Spectral element Zernike discretisation
###
Z = [ZernikeAnnulus(ρ, 0, 0), Zernike(0,0)]
Zd = [ZernikeAnnulus(ρ, 2, 2), Zernike(0,2)]

Δ =  [Zd[1] \ (Laplacian(axes(Z[1],1)) * Z[1]), 
        Zd[2]\ (Laplacian(axes(Z[2],1)) * Z[2])
];

L =  [Zd[1] \ Z[1], 
        Zd[2]\ Z[2]
];

xyₐ = axes(Z[1],1); xₐ,yₐ = first.(xyₐ),last.(xyₐ)
xy = axes(Z[2],1); x,y = first.(xy),last.(xy)
x = [xₐ, x];
y = [yₐ, y];

errors_Z = []
u = []
f = []
for n in 11:10:111
    f = []
    # Expand RHS in Zernike annular polynomials for annulus element
    append!(f,[Zd[1][:, Block.(1:n)] \ rhs_xy.(x[1], y[1])])
    # Expand RHS in Zernike polynomials for the disk element
    append!(f, [Zd[2][:, Block.(1:n)] \ rhs_xy_scale.(x[2], y[2])])


    # Solve by breaking down into solves for each Fourier mode.
    # We utilise a tau-method to enforce the boundary conditions
    # and continuity.
    u = helmholtz_modal_solve(Z, f, n, Δ, L, λs)
    
    print("Computed coefficients for n=$n \n")

    collect_errors(Z, u, ua, true, errors_Z)
end

# Plot the right-hand side
plot_solution(Zd, (ModalTrav(f[1]), ModalTrav(f[2])))
PyPlot.savefig("spectral-element-helmholtz-f.pdf")
# Plot the solution
plot_solution(Z, u)
PyPlot.savefig("spectral-element-u.pdf")

###
# Convergence plot
###

ns = [2*sum(1:10*b) for b in 1:length(errors_Z)]
Plots.plot(ns, errors_Z,
    label=L"\mathrm{Zernike/Zernike \,\, annular \,\, (2 \,\,elements)}",
    ylabel=L"$\infty\mathrm{-norm \;\; error}$",
    xlabel=L"$\# \mathrm{Basis \; functions}$",
    linewidth=2,
    marker=:dot,
    markersize=5,
    color=3
)

tfns = [2b*(2b+2) for b in 11:10:111]
Plots.plot!(tfns, errors_TF,
    label=L"\mathrm{Chebyshev} \otimes \mathrm{Fourier \,\, (2 \,\, elements)}",
    linewidth=2,
    markershape=:dtriangle,
    markersize=5,

    ylabel=L"$l^\infty\mathrm{-norm \;\; error}$",
    xlabel=L"$\# \mathrm{Basis \; functions}$",
    ylim=[1e-15, 1e2],
    xlim = [0, 2.8e4],
    legend=:topright,
    xtickfontsize=10, ytickfontsize=10,xlabelfontsize=15,ylabelfontsize=15,
    yscale=:log10,
    yticks=[1e-15, 1e-10, 1e-5, 1e0],
    color=4
)