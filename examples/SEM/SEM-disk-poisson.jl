using AnnuliPDEs, ClassicalOrthogonalPolynomials, AnnuliOrthogonalPolynomials, MultivariateOrthogonalPolynomials
using PyPlot, Plots, LaTeXStrings
using DelimitedFiles

"""
This script implements the Poisson example of

"Discontinuous variable coefficients and data on a disk" (section 7.3).

We are solving Δu(x,y) = ∑ᵢ dᵢ exp(−aᵢ((x−bᵢ)²+(y−cᵢ)²)) * (−4aᵢ * (−aᵢ(bᵢ² − 2bᵢx + cᵢ² − 2cᵢy + x² + y²) + 1)))
on a disk.

Here we have a right-hand side that has a jump in the radial direction at r=1/2.

The solution is continuous. Hence we use a spectral element method to
split the disk domain into an inner disk cell and an outer annulus cell with
inradius at r=1/2. This performs particularly well.

"""
# ρ is the inner radius on the annulus cell
# κ₀ is the jump coefficient on the inner cell
# κ₁ is the jump coefficient on the outer cell
ρ = 0.5; κ₀ = 1e2; κ₁ = 1e0; 

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

Nn = 1000 # Poly degree of over-sampled grid for error collection.
λs = [0; 0]

###
# (1 element) Zernike discretisation
###
Z = Zernike(0, 1)
wZ = Weighted(Z)

Δ = Z \ (Laplacian(axes(Z,1)) * wZ);
L = Z \ wZ;
Δs, Ls = Δ.ops, L.ops; # decompose into Fourier modes
Δs[311]; # pre-allocation speeds things up
Ls[311];

xy = axes(Z,1); x,y = first.(xy),last.(xy)

# Oversampled grid
G = AnnuliOrthogonalPolynomials.grid(Z, Block(Nn))
# Exact solution on oversampled grid
U = ua.(get_rs.(G), get_θs.(G))
xy = axes(Z,1); x,y = first.(xy),last.(xy)
errors_Z = []
for n in 11:10:311
    # Expand RHS in Zernike annular polynomials
    f = Z[:, Block.(1:n)] \ rhs_xy.(x, y)

    # Solve by breaking down into solves for each Fourier mode
    u = weighted_zernike_modal_solve(f, n, Δs, Ls)
    collect_errors(wZ, u, U, G, errors_Z)

    print("Computed coefficients for n=$n \n")
end
writedlm("errors_Z.log", errors_Z)

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

# Split into Fourier mode components
Δs = [Δ[i].ops for i in 1:2];
Ls = [L[i].ops for i in 1:2];

[Δs[i][151] for i in 1:2]; # pre-allocation speeds things up
[Ls[i][151] for i in 1:2];

xyₐ = axes(Z[1],1); xₐ,yₐ = first.(xyₐ),last.(xyₐ)
xy = axes(Z[2],1); x,y = first.(xy),last.(xy)
x = [xₐ, x];
y = [yₐ, y];

# Oversampled grid
G = AnnuliOrthogonalPolynomials.grid.(Z, Block(Nn))
# Exact solution on oversampled grid
U = [ua.(get_rs.(G[1]), get_θs.(G[1]))]
append!(U, [ua.(ρ*get_rs.(G[2]), get_θs.(G[2]))])

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
    u = zernike_modal_solve(Z, f, n, Δs, Ls, λs)
    collect_errors(Z, u, U, errors_Z_2)

    print("Computed coefficients for n=$n \n")
end
writedlm("errors_Z_2.log", errors_Z_2)

# Plot the right-hand side
plot_solution(Zd, (ModalTrav(f[1]), ModalTrav(f[2])), ttl=latexstring(L"$f_1(x,y)$"))
PyPlot.savefig("spectral-element-f.png", dpi=700)
# Plot the solution
plot_solution(Z, u)
PyPlot.savefig("spectral-element-u.png", dpi=700)


###
# (2-element) Spectral element Zernike + Chebyshev-Fourier series discretisation
###
T,C,F = chebyshevt(ρ..1),ultraspherical(2, ρ..1),Fourier()
r = axes(T,1)
D = Derivative(r)

Lₜ = C \ (r.^2 .* (D^2 * T)) + C \ (r .* (D * T)) # r^2 * ∂^2 + r*∂
M = C\T # Identity
R = C \ (r .* C) # mult by r

Z = Zernike(0,0)
Zd = Zernike(0,2)

Δ =  Zd \ (Laplacian(axes(Z,1)) * Z);
Δs = Δ.ops; # split into Fourier modes
Δs[111]; # pre-allocation speeds things up


xy = axes(Z,1); x,y = first.(xy),last.(xy)

errors_TF_Z = []
for n in 11:10:111
    # Expand RHS in Zernike polynomials for the disk element
    f = Zd[:, Block.(1:n)] \ rhs_xy_scale.(x, y)

    # Solve by breaking down into solves for each Fourier mode.
    # We utilise a tau-method to enforce the boundary conditions
    # and continuity.
    X, u = chebyshev_fourier_zernike_helmholtz_modal_solve([(T,F), Z], rhs_xy, f, ρ, n, [(Lₜ, M, R, D), Δs], [[], []], λs)
    
    print("Computed coefficients for n=$n \n")
    collect_errors((T,F,Z,ρ), (X,u), U, G, errors_TF_Z)
end
writedlm("errors_TF_Z.log", errors_TF_Z)

###
# Convergence plot
###
errors_Z = readdlm("errors_Z.log")
errors_Z_2 = readdlm("errors_Z_2.log")
errors_TF_Z = readdlm("errors_TF_Z.log")

ns = [sum(1:b) for b in 11:10:311]
Plots.plot(ns, errors_Z,
    label=L"\mathrm{Zernike \,\, (1 \,\, element)}",
    linewidth=2,
    markershape=:diamond,
    markersize=5,
)

tfns = [(2b-1)*(b+1) + sum(1:b) for b in 11:10:111]
Plots.plot!(tfns, errors_TF_Z,
    label=L"\mathrm{Zernike/Chebyshev} \otimes \mathrm{Fourier \,\, (2 \,\, elements)}",
    linewidth=2,
    markershape=:dtriangle,
    markersize=5
)

ns = [2*sum(1:b) for b in 11:10:151]
Plots.plot!(ns, errors_Z_2,
    label=L"\mathrm{Zernike/Zernike \,\, annular \,\, (2 \,\,elements)}",
    linewidth=2,
    marker=:dot,
    markersize=5,

    ylabel=L"$l^\infty\mathrm{-norm \;\; error}$",
    xlabel=L"$\# \mathrm{Basis \; functions}$",
    ylim=[1e-15, 5e0],
    # xlim = [0, 6.3e4],
    xlim = [0, 5.5e4],
    legend=:right,
    xtickfontsize=10, ytickfontsize=10,xlabelfontsize=15,ylabelfontsize=15,
    yscale=:log10,
    yticks=[1e-15, 1e-10, 1e-5, 1e0],
    gridlinewidth=2
)

Plots.savefig("SEM-disk-poisson.pdf")