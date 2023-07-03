using ClassicalOrthogonalPolynomials, LinearAlgebra
using AlgebraicCurveOrthogonalPolynomials
using Plots, LaTeXStrings

"""
This script plots the preconditioned condition number of the Laplacian
and matrix sizes for increasing Fourier mode at truncation degree N=100 
for Chebyshev-Fourier series and Zernike annular polynomials.
"""

ρ = 0.5
T,C,F = chebyshevt(ρ..1),ultraspherical(2, ρ..1),Fourier()
r = axes(T,1)
D = Derivative(r)

L = C \ (r.^2 .* (D^2 * T)) + C \ (r .* (D * T)) # r^2 * ∂^2 + r*∂
M = C\T # Identity
R = C \ (r .* C) # mult by r

n = 101;
cs = []
for m in 0:5:100
    Δₘ = L - m^2*M
    X = [T[[begin,end],:]; Δₘ][1:n+2,1:n+2]
    append!(cs, [cond(X * Diagonal(inv.(diag(X))))])
end

Z = ZernikeAnnulus(ρ,1,1)
wZ = Weighted(Z)
Δs = Z \ (Laplacian(axes(Z,1))*wZ)

ds = []
js = []
for m in 1:5:101
    j = length(m:2:n)
    Y = Δs.ops[m][1:j,1:j]
    append!(js, [j])
    append!(ds, [cond(Y * Diagonal(inv.(diag(Y))))])
end

ms = 0:5:100
Plots.plot(ms, [cs ds],
    label=["Chebyshev-Fourier" "Zernike annular"],
    linewidth=2,
    markershape=[:circle :dtriangle],
    markersize=5,

    legend=:topleft,
    ylabel="Condition number",
    xlabel=L"$m$",
    yscale=:log10,
    xtickfontsize=10, ytickfontsize=10,xlabelfontsize=15,ylabelfontsize=15,
    yticks=[1e1,1e3,1e5,1e7],
    ylim=[0.1,1e8]
)

Plots.savefig("laplacian-condition-numbers.pdf")

Plots.plot(ms, [(2n-1)^2*ones(length(ms)) js.^2],
    label=["Chebyshev-Fourier" "Zernike annular"],
    linewidth=2,
    markershape=[:circle :dtriangle],
    markersize=5,

    legend=:topleft,
    ylabel="Condition number",
    xlabel=L"$m$",
    yscale=:log10,
    xtickfontsize=10, ytickfontsize=10,xlabelfontsize=15,ylabelfontsize=15,
    yticks=[1e1,1e3,1e5,1e7],
    ylim=[1e0,1e7]
)

Plots.savefig("laplacian-matrix-size.pdf")