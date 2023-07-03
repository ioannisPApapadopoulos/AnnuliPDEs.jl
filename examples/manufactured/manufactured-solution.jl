using AnnuliPDEs, ClassicalOrthogonalPolynomials, AlgebraicCurveOrthogonalPolynomials, SemiclassicalOrthogonalPolynomials
using PyPlot, Plots, LaTeXStrings

"""
This script implements the "Manufactured solution" example (section 7.3).

We are solving Δu(x,y) = f(x,y)

where the exact solution is u(r, θ) = (1 − r²)(r² − ρ²)r⁶(exp(sin(2θ)) + exp(sin(20θ)))

on the annulus with inradius ρ = 1/2.

"""

ρ = 0.5
# Exact manafactured solution
function ua(r,θ)
    if θ > π
        θ = θ - 2π
    end
    (1-r^2)*(r^2-ρ^2)*r^6*(exp(sin(2θ))+exp(sin(20θ)))
end

# Use ForwardDiff to compute the RHS
function rhs(r,θ)
    if θ > π
        θ = θ - 2π
    end
    # uᵣᵣ(r,θ) + uᵣ(r,θ)/r + uθθ(r,θ)/r^2
    derivative(r->derivative(r->ua(r,θ),r),r) + derivative(r->ua(r,θ),r)/r + derivative(θ->derivative(θ->ua(r,θ),θ),θ)/r^2
end
# RHS in Cartesian coordinates
function rhs_xy(x, y)
    r = sqrt(x^2 + y^2); θ = atan(y, x)
    rhs(r,θ)
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
for n in 10:10:320
    # Compute coefficients of solution to Helmholtz problem with Chebyshev-Fourier series
    X, _ = chebyshev_fourier_helmholtz_modal_solve((T, F), (L, M, R), rhs_xy, n, 0.0) 
    print("Computed coefficients for n=$n \n")

    collect_errors((T,F,ρ), X, ua, errors_TF)
end
errors_TF

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
# for n in 10:10:320
#     # Compute coefficients of solution to Helmholtz problem with Chebyshev-Fourier series
#     X, _ = twoband_fourier_helmholtz_modal_solve((U, F), (Δᵣ, L, M, R), rhs_xy, n, 0.0) 
#     print("Computed coefficients for n=$n \n")

#     collect_errors((T,F,ρ), X, ua, errors_TBF)
# end
# errors_TBF

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
for n in 10:10:320

    # Expand RHS in Zernike annular polynomials
    f = Z[:, Block.(1:n)] \ rhs_xy.(x, y)

    # Solve by breaking down into solves for each Fourier mode
    u = helmholtz_modal_solve(f, n, Δ, L)
    
    print("Computed coefficients for n=$n \n")

    collect_errors(wZ, u, ua, errors_Z)
end

# Plot the solution
plot_solution(wZ, u)
PyPlot.savefig("manafactured-solution.pdf")

###
# Convergence plot
###
bs = [sum(1:10*b) for b in 1:length(errors_Z)]
Plots.plot(bs, errors_Z,
    label=L"$\mathrm{Zernike \,\, annular}$",
    linewidth=2,
    markershape=:circle,
    markersize=5)

# ns = [n^2/2 for n in 10:10:320]
# Plots.plot!(bs, errors_TBF,
#     label=L"$\mathrm{Two}$-$\mathrm{band} \otimes \mathrm{Fourier}$",
#     linewidth=2,
#     markershape=:dtriangle,
#     markersize=5)

ns = [(n+2)*(2n-1) for n in 10:10:320]
Plots.plot!(ns[1:20], errors_TF[1:20],
    label=L"$\mathrm{Chebyshev}(r_\rho)\otimes\mathrm{Fourier}$",
    linewidth=2,
    markershape=:diamond,
    markersize=5,
    

    legend=:topright,
    yscale=:log10,
    xtickfontsize=10, ytickfontsize=10,xlabelfontsize=15,ylabelfontsize=15,
    ylabel=L"$l^\infty\mathrm{-norm \;\; error}$",
    xlabel=L"$\# \mathrm{Basis \,\, functions}$")

Plots.savefig("manfactured-solution-convergence.pdf")