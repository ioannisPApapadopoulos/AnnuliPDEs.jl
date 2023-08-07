using AnnuliPDEs, ClassicalOrthogonalPolynomials, AnnuliOrthogonalPolynomials, MultivariateOrthogonalPolynomials
using PyPlot, Plots, LaTeXStrings

"""
This script implements the Poisson example of

"Discontinuous variable coefficients and data" (section 7.4).

We are solving Δ u(x,y) = exp(−a((x−b)²+(y−c)²)) * (−4a * (−a(b² − 2bx + c² − 2cy + x² + y²) + 1)))
on a disk.

Here we have a right-hand side that has a jump in the radial direction at r=1/2.

The solution is continuous. Hence we use a spectral element method to
split the disk domain into an inner disk cell and an outer annulus cell with
inradius at r=1/2. This performs particularly well.

"""

ρ = 0.5; κ₀ = 1e2; κ₁ = 1e0; 
# Exact solution

# Gaussian bump
θs = [0, π/2, π/3, 5π/4]
c = zeros(length(θs)+1,3)
for i in 1:lastindex(θs) c[i,:] = [-10*i ρ*cos(θs[i]) ρ*sin(θs[i])] end
c[end,:] = [-50 0.9*cos(3π/4) 0.9*sin(3π/4)]

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

# Use ForwardDiff to compute the RHS
rhs_(r,θ) =  (derivative(r->derivative(r->ua(r,θ), r),r) 
    + derivative(r->ua(r,θ), r)/r 
    + derivative(θ->derivative(θ->ua(r,θ), θ),θ)/r^2
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

λs = [0; 0]

###
# (1-element) Chebyshev-Fourier series discretisation
###

T,C,F = chebyshevt(0..1),ultraspherical(2, 0..1),Fourier()
r = axes(T,1)
D = Derivative(r)

L = C \ (r.^2 .* (D^2 * T)) + C \ (r .* (D * T)) # r^2 * ∂^2 + r*∂ = (r²*Δ)
M = C\T # Identity
R = C \ (r .* C) # mult by r

errors_TF = []
for n in 11:10:231
    # Compute coefficients of solution to Helmholtz problem with Chebyshev-Fourier series
    X, _ = chebyshev_fourier_helmholtz_modal_solve((T, F), (L, M, R), rhs_xy, n, 0.0) 
    print("Computed coefficients for n=$n \n")

    collect_errors((T,F,0.0), X, ua, errors_TF)
end

###
# (2-element) Spectral element Chebyshev-Fourier series discretisation
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

errors_TF_2 = []
for n in 11:10:111
    # Compute coefficients of solution to Helmholtz problem with Chebyshev-Fourier series
    X, _ = chebyshev_fourier_helmholtz_modal_solve((T, Tₐ, F), (L, Lₐ, M, Mₐ, R, Rₐ), (D, Dₐ), rhs_xy, n, λs) 
    print("Computed coefficients for n=$n \n")

    collect_errors((T,Tₐ,F,ρ), X, ua, errors_TF_2)
end

###
# (1 element) Zernike discretisation
###
Z = Zernike(0, 1)
wZ = Weighted(Z)

Δ = Z \ (Laplacian(axes(Z,1)) * wZ);
L = Z \ wZ;

xy = axes(Z,1); x,y = first.(xy),last.(xy)
errors_Z = []
for n in 11:10:311
    # Expand RHS in Zernike annular polynomials
    f = Z[:, Block.(1:n)] \ rhs_xy.(x, y)

    # Solve by breaking down into solves for each Fourier mode
    u = helmholtz_modal_solve(f, n, Δ, L)
    
    print("Computed coefficients for n=$n \n")

    collect_errors(wZ, u, ua, errors_Z)
end

###
# (2-element) Spectral element Zernike discretisation
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

errors_Z_2 = []
u = []
f = []
for n in 11:10:151
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

    collect_errors(Z, u, ua, true, errors_Z_2)
end

# Plot the right-hand side
plot_solution(Zd, (ModalTrav(f[1]), ModalTrav(f[2])))
PyPlot.savefig("spectral-element-f.pdf")
# Plot the solution
plot_solution(Z, u)
PyPlot.savefig("spectral-element-u.pdf")

###
# Convergence plot
###

ns = [sum(1:b) for b in 11:10:311]
Plots.plot(ns, errors_Z,
    label=L"\mathrm{Zernike \,\, (1 \,\, element)}",
    linewidth=2,
    markershape=:diamond,
    markersize=5,
)

tfns = [(2b-1)*(b+2) for b in 11:10:231]
Plots.plot!(tfns, errors_TF,
    label=L"\mathrm{Chebyshev}\otimes \mathrm{Fourier\,\, (1 \,\, element)}",
    linewidth=2,
    markershape=:xcross,
    markersize=5,
)

ns = [2*sum(1:b) for b in 11:10:151]
Plots.plot!(ns, errors_Z_2,
    label=L"\mathrm{Zernike/Zernike \,\, annular \,\, (2 \,\,elements)}",
    linewidth=2,
    marker=:dot,
    markersize=5,
)

tfns = [(2b-1)*(2b+2) for b in 11:10:111]
Plots.plot!(tfns, errors_TF_2,
    label=L"\mathrm{Chebyshev} \otimes \mathrm{Fourier \,\, (2 \,\, elements)}",
    linewidth=2,
    markershape=:dtriangle,
    markersize=5,

    ylabel=L"$l^\infty\mathrm{-norm \;\; error}$",
    xlabel=L"$\# \mathrm{Basis \; functions}$",
    ylim=[1e-15, 5e0],
    xlim = [0, 6.3e4],
    legend=:right,
    xtickfontsize=10, ytickfontsize=10,xlabelfontsize=15,ylabelfontsize=15,
    yscale=:log10,
    yticks=[1e-15, 1e-10, 1e-5, 1e0]
)
Plots.savefig("spectral-element-convergence.pdf")