using ClassicalOrthogonalPolynomials, LinearAlgebra, Plots
using SemiclassicalOrthogonalPolynomials

"""
This script showcases computational complexity for the Q-based QR decomposition method for the computation of semiclassical Jacobi polynomials via their Jacobi matrix.
"""

t = 1.1
Q = SemiclassicalJacobi(t, 1, 1, 20)
Q.X[5000,5000] # pre-cache the Jacobi matrix for Q to avoid polluting the computational cost

cpuelapsed = zeros(10)
@inbounds for j = 1:10
    P = SemiclassicalJacobi(t, 1, 1, 22, Q) # re-generate P to avoid caching for subsequent steps
    cpuelapsed[j] = @elapsed P.X[400*j,400*j]
end

xticks = Vector(range(400,4000,10))
scatter(xticks,cpuelapsed, xlabel="N", ylabel="CPU time [s]", label="Q-based QR method") # plot linear cost increase with Jacobi matrix size