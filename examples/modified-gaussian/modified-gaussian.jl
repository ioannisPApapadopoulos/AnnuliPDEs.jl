using AnnuliPDEs, ClassicalOrthogonalPolynomials, AnnuliOrthogonalPolynomials, SemiclassicalOrthogonalPolynomials
using PyPlot, Plots, LaTeXStrings

"""
This script implements the "Modified Gaussian bump" example (section 7.2).

We are solving Δu(x,y) = −4a * exp(−a((x−b)²+(y−c)²)) * (−a(b² − 2bx + c² − 2cy + x² + y²) + 1)

where, if a ≫ 1, the exact solution is u(x, y) ≈ exp(−a((x−b)²+(y−c)²)).

on the annulus with inradius ρ = 0.2.

"""

ρ = 0.2
# Exact solution in Cartesian coordinates
c1 = -250; c2 = 0; c3=0.6
ua_xy(x, y) = exp(c1*(x^2 + (y-c3)^2))

# Exact solution in polar coordinates
function ua(r,θ)
    x = r*cos(θ); y = r*sin(θ)
    ua_xy(x, y)
end

# RHS
function rhs_xy(x, y)
    4*c1*exp(c1*((c2 - x)^2 + (c3 - y)^2))*(c1*(c2^2 - 2*c2*x + c3^2 - 2*c3*y + x^2 + y^2) + 1)
end

###
# Chebyshev-Fourier series discretisation
###
T,C,F = chebyshevt(ρ..1),ultraspherical(2, ρ..1),Fourier()
r = axes(T,1)
D = Derivative(r)

L = C \ (r.^2 .* (D^2 * T)) + C \ (r .* (D * T)) # r^2 * ∂^2 + r*∂ = (r²*Δ)
M = C\T # Identity
R = C \ (r .* C) # mult by r

errors_TF = []
for n in 10:10:250
    # Compute coefficients of solution to Helmholtz problem with Chebyshev-Fourier series
    X, _ = chebyshev_fourier_helmholtz_modal_solve((T, F), (L, M, R), rhs_xy, n, 0.0) 
    print("Computed coefficients for n=$n \n")

    collect_errors((T,F,ρ), X, ua, errors_TF)
end

###
# Two-band-Fourier series discretisation
###
# T,U,C,F = HalfWeighted{:ab}(TwoBandJacobi(ρ,1,1,0)),TwoBandJacobi(ρ,0,0,0),TwoBandJacobi(ρ,1,1,0),Fourier()
# r = axes(T,1)
# D = Derivative(r)
# R = jacobimatrix(C) # mult by r
# r∂ = R * (C \ (D*T)) # r*∂
# ∂² = (C \ (D^2 * T))
# r²∂² = R * R * ∂²
# Δᵣ = r²∂² + r∂
# L = C \ U
# M = L*(U \ T) # Identity

# errors_TBF = []
# for n in 8:8:300
#     # Compute coefficients of solution to Helmholtz problem with Chebyshev-Fourier series
#     X, _ = twoband_fourier_helmholtz_modal_solve((U, F), (Δᵣ, L, M, R), rhs_xy, n, 0.0) 
#     print("Computed coefficients for n=$n \n")

#     collect_errors((T,F,ρ), X, ua, errors_TBF)
# end

###
# Zernike annular discretisation
###
Z = ZernikeAnnulus(ρ,1,1)
wZ = Weighted(Z)

Δ = Z \ (Laplacian(axes(Z,1)) * wZ);
L = Z \ wZ;

xy = axes(Z,1); x,y = first.(xy),last.(xy)
errors_Z = []
u = []
for n in 10:10:200

    # Expand RHS in Zernike annular polynomials
    f = Z[:, Block.(1:n)] \ rhs_xy.(x, y)

    # Solve by breaking down into solves for each Fourier mode
    u = helmholtz_modal_solve(f, n, Δ, L)
    
    print("Computed coefficients for n=$n \n")

    collect_errors(wZ, u, ua, errors_Z)
end

# Plot the solution
plot_solution(wZ, u)
PyPlot.savefig("modified-gaussian.pdf")

###
# Convergence plot
###
bs = [sum(1:b) for b in 10:10:200]
Plots.plot(bs, errors_Z,
    label=L"$\mathrm{Zernike \,\, annular}$",
    linewidth=2,
    marker=:dot,
    markersize=5,
)
# ns = [b^2/2 for b in 8:8:100]
# Plots.plot!(ns, errors_TBF,
#     label=L"$\mathrm{Two}$-$\mathrm{band} \otimes \mathrm{Fourier}$",
#     linewidth=2,
#     markershape=:diamond,
#     markersize=5,
# )

ns = [(2n-1)*(n+2) for n = 10:10:250]
Plots.plot!(ns[1:12], errors_TF[1:12],
    label=L"$\mathrm{Chebyshev}(r_\rho) \otimes \mathrm{Fourier}$",
    linewidth=2,
    markershape=:dtriangle,
    markersize=5,

    yscale=:log10,
    xtickfontsize=10, ytickfontsize=10,xlabelfontsize=15,ylabelfontsize=15,
    ylim=[1e-15, 1e1],
    yticks=[1e-15,1e-10,1e-5,1e0],
    xlim=[0, 3.2e4],
    legend=:topright,
    ylabel=L"$l^\infty\mathrm{-norm \;\; error}$",
    xlabel=L"$\# \mathrm{Basis \, \,functions}$",
)
Plots.savefig("gaussian-bump-convergence.pdf")