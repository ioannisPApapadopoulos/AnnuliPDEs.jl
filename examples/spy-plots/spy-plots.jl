using ClassicalOrthogonalPolynomials, LinearAlgebra, Plots
using AlgebraicCurveOrthogonalPolynomials
import AlgebraicCurveOrthogonalPolynomials: Laplacian

"""
This script gives spy plots of the Laplacian and Helmholtz operators
as applied to the Zernike annular polynomials.
"""

ρ = 0.5

Z = ZernikeAnnulus(ρ,1,1)
wZ = Weighted(Z)
Δs = Z \ (Laplacian(axes(Z,1))*wZ)

Plots.spy(Δs[Block.(1:21),Block.(1:21)], markersize=2)
Plots.savefig("spy-Laplacian.pdf")

Plots.spy(Δs.ops[1][1:21,1:21], markersize=5)
Plots.savefig("0-mode-spy-Laplacian.pdf")

Bs = Z \ wZ

Plots.spy(Δs[Block.(1:21),Block.(1:21)]+Bs[Block.(1:21),Block.(1:21)], markersize=2)
Plots.savefig("spy-helmholtz.pdf")

Plots.spy(Δs.ops[1][1:21,1:21]+Bs.ops[1][1:21,1:21], markersize=5)
Plots.savefig("0-mode-spy-Helmholtz.pdf")   