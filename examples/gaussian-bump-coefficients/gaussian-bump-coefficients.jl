using AnnuliPDEs, ClassicalOrthogonalPolynomials, AlgebraicCurveOrthogonalPolynomials, SemiclassicalOrthogonalPolynomials, LinearAlgebra
using PyPlot, Plots, LaTeXStrings

"""
This script implements the "Gaussian bump" example.

We are solving Δu(x,y) = exp(a((x−b)²+(y−c)²) with a=25, b=0, and c = 0.99. 

on two annuli with inradius ρ = 0.2 and 0.8.

"""

ρs = [0.2, 0.8]

# RHS in Cartesian coordinates
c = [-25 0 0.99]
function rhs_xy(x, y)
    s = 0;
    for j in 1:size(c,1)
        s += exp(c[j,1]*((c[j,2] - x)^2 + (c[j,3] - y)^2))
    end
    s
end

###
# Chebyshev-Fourier series discretisation
###
n = 250
coeff_TF_f = [[], []]
coeff_TF_u = [[], []]
for (ρ, i) in zip(ρs, 1:2)
    T,C,F = chebyshevt(ρ..1),ultraspherical(2, ρ..1),Fourier()
    r = axes(T,1)
    D = Derivative(r)

    L = C \ (r.^2 .* (D^2 * T)) + C \ (r .* (D * T)) # r^2 * ∂^2 + r*∂ = (r²*Δ)
    M = C\T # Identity
    R = C \ (r .* C) # mult by r

    # Compute coefficients of solution to Helmholtz problem with Chebyshev-Fourier series
    (X, Fs) = chebyshev_fourier_helmholtz_modal_solve((T, F), (L, M, R), rhs_xy, n, 0.0) 

    # Want to count the coefficients in a zigzag manner
    # so that we are comparing the same number of coefficients
    # with Zernike annular polynomials.
    Xs = X[1:end-2, 1:end-1]
    Xs_0 = (Xs[:,1:2:end])[:,end:-1:1]
    Xs_1 = (Xs[:,2:2:end])[:,end:-1:1]
    Fss_0 = (Fs[1:n,1:2:end])[:,end:-1:1]
    Fss_1 = (Fs[1:n,2:2:end])[:,end:-1:1]

    for ns = 2:n
        append!(coeff_TF_u[i],norm(vcat(diag(Xs_0, ns-1), diag(Xs_1, ns-2)), ∞))
        append!(coeff_TF_f[i],norm(vcat(diag(Fss_0, ns-1),diag(Fss_1, ns-2)), ∞))
    end
    coeff_TF_u[i] = coeff_TF_u[i][end:-1:1]
    coeff_TF_f[i] = coeff_TF_f[i][end:-1:1]

end

###
# Two-band-Fourier series discretisation
###
# n = 250
# coeff_TBF_f = [[], []]
# coeff_TBF_u = [[], []]
# for (ρ, i) in zip(ρs, 1:2)
#     T,U,C,F = HalfWeighted{:ab}(TwoBandJacobi(ρ,1,1,0)),TwoBandJacobi(ρ,0,0,0),TwoBandJacobi(ρ,1,1,0),Fourier()
#     r = axes(T,1)
#     D = Derivative(r)
#     R = jacobimatrix(C) # mult by r
#     r∂ = R * (C \ (D*T)) # r*∂
#     ∂² = (C \ (D^2 * T))
#     r²∂² = R * R * ∂²
#     Δᵣ = r²∂² + r∂
#     L = C \ U
#     M = L*(U \ T) # Identity

#     errors_TBF = []
#     # Compute coefficients of solution to Helmholtz problem with Chebyshev-Fourier series
#     (X, Fs) = twoband_fourier_helmholtz_modal_solve((U, F), (Δᵣ, L, M, R), rhs_xy, n, 0.0) 
    
#     # Want to count the coefficients in a zigzag manner
#     # so that we are comparing the same number of coefficients
#     # with Zernike annular polynomials.
#     Xs = X[:,end:-1:1]
#     Fss = Fs[:,end:-1:1]

#     for ns = 1:n
#         append!(coeff_TBF_u[i],norm(diag(Xs, ns-1), ∞))
#         append!(coeff_TBF_f[i],norm(diag(Fss, ns-1), ∞))
#     end
#     coeff_TBF_u[i] = coeff_TBF_u[i][end:-1:1]
#     coeff_TBF_f[i] = coeff_TBF_f[i][end:-1:1]
# end

###
# Zernike annular discretisation
###
n = 250
coeff_Z_f = [[], []]
coeff_Z_u = [[], []]
us = []
for (ρ, i) in zip(ρs, 1:2)
    Z = ZernikeAnnulus(ρ,1,1)
    wZ = Weighted(Z)

    Δ = Z \ (Laplacian(axes(Z,1)) * wZ);
    L = Z \ wZ;

    xy = axes(Z,1); x,y = first.(xy),last.(xy)

    # Expand RHS in Zernike annular polynomials
    f = Z[:, Block.(1:n)] \ rhs_xy.(x, y)

    # Solve by breaking down into solves for each Fourier mode
    u = helmholtz_modal_solve(f, n, Δ, L)
    append!(us, [u])
    for ns in 1:250
        append!(coeff_Z_f[i], norm(f[Block(ns)], ∞))
        append!(coeff_Z_u[i], norm(u[Block(ns)], ∞))
    end
end

# Plot the solutions

plot_solution(Weighted(ZernikeAnnulus(ρs[1],1,1)), us[1])
PyPlot.savefig("gaussian-bump-0-2.pdf")

plot_solution(Weighted(ZernikeAnnulus(ρs[2],1,1)), us[2])
PyPlot.savefig("gaussian-bump-0-8.pdf")

###
# Convergence plots
###
n = 250

# Right-hand side coefficients
Plots.plot(0:n-1, [coeff_Z_f[1] coeff_Z_f[2]],
    label=[L"$\rho = 0.2, \; \mathrm{Zernike \,\, annular}$" L"$\rho = 0.8, \; \mathrm{Zernike \,\, annular}$"],
    linewidth=2,
    linestyle=[:dot :dash],
    markersize=5,
)

# Plots.plot!(1:2:n-1, [coeff_TBF_f[1][1:2:n] coeff_TBF_f[2][1:2:n]],
#     label=[L"$\rho = 0.2, \; \mathrm{Two}$-$\mathrm{band}  \otimes \mathrm{Fourier}$" L"$\rho = 0.8, \; \mathrm{Two}$-$\mathrm{band}  \otimes \mathrm{Fourier}$"],
#     linewidth=2,
#     markersize=5,
#     linestyle=[:dashdotdot :solid],
# )

Plots.plot!(1:n-1, [coeff_TF_f[1] coeff_TF_f[2]],

    label=[L"$\rho = 0.2, \; \mathrm{Chebyshev}(r_\rho) \otimes \mathrm{Fourier}$" L"$\rho = 0.8, \; \mathrm{Chebyshev}(r_\rho) \otimes \mathrm{Fourier}$"],
    linewidth=2,
    markersize=5,
    linestyle=[:dashdot :solid],

    yscale=:log10,
    xtickfontsize=10, ytickfontsize=10,xlabelfontsize=15,ylabelfontsize=15,
    yticks=[1e-20, 1e-15,1e-10,1e-5,1e0],
    legend=:topright,
    ylabel=L"$l^\infty\mathrm{-norm}$",
    xlabel=L"$k$",
)
Plots.savefig("gaussian-bump-coefficient-f.pdf")


# Solution coefficients
Plots.plot(0:n-1, [coeff_Z_u[1] coeff_Z_u[2]],
    label=[L"$\rho = 0.2, \; \mathrm{Zernike \,\, annular}$" L"$\rho = 0.8, \; \mathrm{Zernike \,\, annular}$"],
    linewidth=2,
    linestyle=[:dot :dash],
    markersize=5,
)

# Plots.plot!(1:2:n-1, [coeff_TBF_u[1][1:2:n] coeff_TBF_u[2][1:2:n]],
#     label=[L"$\rho = 0.2, \; \mathrm{Two}$-$\mathrm{band}  \otimes \mathrm{Fourier}$" L"$\rho = 0.8, \; \mathrm{Two}$-$\mathrm{band}  \otimes \mathrm{Fourier}$"],
#     linewidth=2,
#     markersize=5,
#     linestyle=[:dashdotdot :solid],
# )

Plots.plot!(1:n-1, [coeff_TF_u[1] coeff_TF_u[2]],

    label=[L"$\rho = 0.2, \; \mathrm{Chebyshev}(r_\rho) \otimes \mathrm{Fourier}$" L"$\rho = 0.8, \; \mathrm{Chebyshev}(r_\rho) \otimes \mathrm{Fourier}$"],
    linewidth=2,
    markersize=5,
    linestyle=[:dashdot :solid],

    yscale=:log10,
    xtickfontsize=10, ytickfontsize=10,xlabelfontsize=15,ylabelfontsize=15,
    yticks=[1e-20, 1e-15,1e-10,1e-5,1e0],
    legend=:topright,
    ylabel=L"$l^\infty\mathrm{-norm}$",
    xlabel=L"$k$",
)
Plots.savefig("gaussian-bump-coefficient-u.pdf")